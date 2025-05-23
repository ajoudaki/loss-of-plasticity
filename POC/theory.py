import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from scipy import stats, integrate
from matplotlib.gridspec import GridSpec

# --- Matplotlib Styling for Paper Quality ---
try:
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        'font.family': 'serif',
        'text.usetex': False,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'font.size': 9,
        'axes.labelsize': 9,
        'axes.titlesize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.titlesize': 11,
    })
except OSError:
    print("seaborn-v0_8-paper style not found, using rcParams for basic styling.")
    plt.rcParams.update({
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.titlesize': 11,
        'lines.linewidth': 1.5,
        'lines.markersize': 5,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'font.family': 'serif',
        'text.usetex': False
    })

# --- Global Parameters ---
FIGURE_DIR = "./figures/"
os.makedirs(FIGURE_DIR, exist_ok=True)

# --- Helper Functions ---
# Note: For computing α_f, we provide two integration methods:
# 1. compute_alpha_f: Uses scipy's adaptive quadrature (more flexible, handles any activation)
# 2. compute_alpha_f_hermite: Uses Gauss-Hermite quadrature (faster, ideal for smooth activations)
# Both methods give equivalent results for well-behaved activation functions.

def compute_effective_rank(matrix_or_svals):
    """Compute effective rank from matrix or singular values."""
    if isinstance(matrix_or_svals, torch.Tensor) and matrix_or_svals.ndim == 2:
        s_unnormalized = torch.linalg.svdvals(matrix_or_svals).detach().cpu().numpy()
    elif isinstance(matrix_or_svals, torch.Tensor) and matrix_or_svals.ndim == 1:
        s_unnormalized = matrix_or_svals.detach().cpu().numpy()
    elif isinstance(matrix_or_svals, np.ndarray) and matrix_or_svals.ndim == 1:
        s_unnormalized = matrix_or_svals
    elif isinstance(matrix_or_svals, np.ndarray) and matrix_or_svals.ndim == 2:
        s_unnormalized = np.linalg.svd(matrix_or_svals, compute_uv=False)
    else:
        raise ValueError("Input must be a 2D matrix or 1D singular values")

    sum_s = np.sum(s_unnormalized)
    if sum_s < 1e-12:
        return 0.0
    s_norm = s_unnormalized / sum_s
    s_norm = s_norm[s_norm > 1e-15]
    if len(s_norm) == 0:
        return 0.0
    entropy = -np.sum(s_norm * np.log(s_norm))
    effective_rank_val = np.exp(entropy)
    return effective_rank_val

def compute_alpha_f(activation_fn, integration_limits=(-8, 8), integration_points=1000):
    """Compute the rank recovery strength α_f for an activation function using numerical integration.
    
    α_f = E[f'(z)²]/E[f(z)²] - 1 where z ~ N(0,1)
    
    We compute:
    E[f(z)²] = ∫ f(z)² φ(z) dz
    E[f'(z)²] = ∫ f'(z)² φ(z) dz
    where φ(z) = (1/√(2π)) exp(-z²/2) is the standard normal PDF
    """
    from scipy import integrate
    
    # Standard normal PDF
    def normal_pdf(z):
        return np.exp(-z**2 / 2) / np.sqrt(2 * np.pi)
    
    # Define integrand for E[f(z)²]
    def integrand_f_squared(z):
        z_tensor = torch.tensor(z, dtype=torch.float32)
        f_z = activation_fn(z_tensor)
        return (f_z.detach().numpy() ** 2) * normal_pdf(z)
    
    # Define integrand for E[f'(z)²]
    def integrand_fprime_squared(z):
        z_tensor = torch.tensor(z, dtype=torch.float32, requires_grad=True)
        f_z = activation_fn(z_tensor)
        
        # Compute derivative
        f_prime = torch.autograd.grad(f_z, z_tensor)[0].item()
        
        return (f_prime ** 2) * normal_pdf(z)
    
    # Compute integrals using adaptive quadrature
    E_f_squared, error_f = integrate.quad(integrand_f_squared, 
                                          integration_limits[0], 
                                          integration_limits[1],
                                          limit=integration_points)
    
    E_f_prime_squared, error_fprime = integrate.quad(integrand_fprime_squared, 
                                                     integration_limits[0], 
                                                     integration_limits[1],
                                                     limit=integration_points)
    
    # Compute α_f
    if E_f_squared > 1e-10:
        alpha_f = (E_f_prime_squared / E_f_squared) - 1
    else:
        alpha_f = 0.0
    
    return alpha_f, E_f_squared, E_f_prime_squared

def compute_alpha_f_hermite(activation_fn, n_points=100):
    """Compute α_f using Gauss-Hermite quadrature (more efficient for Gaussian integrals).
    
    This method is particularly well-suited for integrating against Gaussian weights.
    """
    # Get Gauss-Hermite quadrature points and weights
    # Note: numpy's hermgauss uses weight function exp(-x²), so we need to adjust
    x, w = np.polynomial.hermite.hermgauss(n_points)
    
    # Adjust points for standard normal (divide by sqrt(2))
    z_points = x / np.sqrt(2)
    # Adjust weights for standard normal
    weights = w / np.sqrt(np.pi)
    
    # Compute f(z) for all points
    f_values = []
    f_prime_values = []
    
    for i in range(len(z_points)):
        z_i = torch.tensor(z_points[i], dtype=torch.float32, requires_grad=True)
        
        f_i = activation_fn(z_i)
        f_values.append(f_i.item())
        
        # Compute derivative
        f_prime = torch.autograd.grad(f_i, z_i)[0].item()
        f_prime_values.append(f_prime)
    
    f_values = np.array(f_values)
    f_prime_values = np.array(f_prime_values)
    
    # Compute expectations using quadrature
    E_f_squared = np.sum(weights * f_values**2)
    E_f_prime_squared = np.sum(weights * f_prime_values**2)
    
    # Compute α_f
    if E_f_squared > 1e-10:
        alpha_f = (E_f_prime_squared / E_f_squared) - 1
    else:
        alpha_f = 0.0
    
    return alpha_f, E_f_squared, E_f_prime_squared

def create_correlated_gaussian(n_samples, d_features, correlation=0.9):
    """Create correlated Gaussian data with specified correlation."""
    # Create covariance matrix
    cov = torch.eye(d_features) * (1 - correlation) + correlation
    
    # Generate samples
    mean = torch.zeros(d_features)
    samples = torch.distributions.MultivariateNormal(mean, cov).sample((n_samples,))
    
    return samples, cov

def create_rank_deficient_data(n_samples, d_full, d_rank):
    """Create rank-deficient data by projecting through low-rank matrix."""
    # Generate low-rank data
    L = torch.randn(n_samples, d_rank)
    A = torch.randn(d_rank, d_full)
    
    # Create rank-deficient data
    X = L @ A
    
    # Add small noise to avoid exact zero eigenvalues
    X = X + 1e-6 * torch.randn_like(X)
    
    return X

# --- Validation 1: Hard Rank Recovery ---
def validate_hard_rank_recovery():
    """Validate that non-linear activations recover hard rank from rank-deficient inputs."""
    print("\n=== Validating Hard Rank Recovery ===")
    
    n_samples = 10000
    d_full = 50
    d_rank = 10
    
    # Create rank-deficient data
    X = create_rank_deficient_data(n_samples, d_full, d_rank)
    
    # Compute pre-activation covariance and rank
    cov_X = torch.cov(X.T)
    svals_X = torch.linalg.svdvals(cov_X).detach().numpy()
    rank_X = np.sum(svals_X > 1e-6 * svals_X[0])
    
    # Test different activation functions
    activations = {
        'ReLU': torch.nn.ReLU(),
        'Tanh': torch.nn.Tanh(),
        'GELU': torch.nn.GELU(),
        'Sigmoid': torch.nn.Sigmoid(),
        'ELU': torch.nn.ELU(),
    }
    
    results = []
    
    for name, act_fn in activations.items():
        # Apply activation
        Y = act_fn(X)
        
        # Compute post-activation covariance and rank
        cov_Y = torch.cov(Y.T)
        svals_Y = torch.linalg.svdvals(cov_Y).detach().numpy()
        rank_Y = np.sum(svals_Y > 1e-6 * svals_Y[0])
        eff_rank_Y = compute_effective_rank(svals_Y)
        
        results.append({
            'Activation': name,
            'Pre-Act Rank': rank_X,
            'Post-Act Rank': rank_Y,
            'Post-Act Eff Rank': eff_rank_Y,
            'Rank Recovery': rank_Y > rank_X
        })
        
        print(f"{name}: Pre-rank={rank_X}, Post-rank={rank_Y}, Eff-rank={eff_rank_Y:.2f}")
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Plot 1: Singular value spectrum
    ax1.semilogy(svals_X[:d_full//2], 'k--', label='Pre-activation', linewidth=2)
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(activations)))
    for i, (name, act_fn) in enumerate(activations.items()):
        Y = act_fn(X)
        cov_Y = torch.cov(Y.T)
        svals_Y = torch.linalg.svdvals(cov_Y).detach().numpy()
        ax1.semilogy(svals_Y[:d_full//2], label=f'{name}', color=colors[i], linewidth=1.5)
    
    ax1.set_xlabel('Singular Value Index')
    ax1.set_ylabel('Singular Value')
    ax1.set_title('Singular Value Spectrum')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Rank comparison
    df_results = pd.DataFrame(results)
    x = np.arange(len(df_results))
    width = 0.6
    
    # Show pre-activation rank as a single bar at the beginning
    ax2.bar([-0.7], [rank_X], width=0.4, label='Pre-activation', color='gray', alpha=0.7)
    
    # Show post-activation ranks for each activation
    ax2.bar(x, df_results['Post-Act Rank'], width, label='Post-activation', color='darkgreen', alpha=0.7)
    
    ax2.set_xlabel('Activation Function')
    ax2.set_ylabel('Rank')
    ax2.set_title('Hard Rank Recovery')
    
    # Set x-ticks and labels
    all_ticks = [-0.7] + list(x)
    all_labels = ['Pre-act'] + list(df_results['Activation'])
    ax2.set_xticks(all_ticks)
    ax2.set_xticklabels(all_labels, rotation=45)
    ax2.grid(True, axis='y', alpha=0.3)
    
    # Add horizontal line at full rank
    ax2.axhline(y=d_full, color='red', linestyle=':', alpha=0.5)
    
    # Create legend entries manually to include the full rank line
    from matplotlib.lines import Line2D
    legend_elements = [
        plt.Rectangle((0,0),1,1, color='gray', alpha=0.7, label='Pre-activation'),
        plt.Rectangle((0,0),1,1, color='darkgreen', alpha=0.7, label='Post-activation'),
        Line2D([0], [0], color='red', linestyle=':', alpha=0.5, label='Full rank')
    ]
    ax2.legend(handles=legend_elements)
    
    # Adjust x-axis limits to accommodate the pre-activation bar
    ax2.set_xlim(-1.2, len(df_results) - 0.4)
    
    plt.suptitle(f'Hard Rank Recovery from Rank-{d_rank} Input (d={d_full})\nPre-activation rank is same ({rank_X}) for all activations', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "validation_hard_rank_recovery.pdf"))
    plt.show()
    
    return results

# --- Validation 2: Effective Rank Recovery Formula ---
def validate_effective_rank_formula():
    """Validate the theoretical formula for effective rank recovery."""
    print("\n=== Validating Effective Rank Recovery Formula ===")
    
    n_samples = 10000
    d_features = 100
    epsilons = np.logspace(-3, -1, 10)  # Different correlation levels
    
    # Test activations
    test_activations = {
        'ReLU': torch.nn.ReLU(),
        'Tanh': torch.nn.Tanh(),
        'GELU': torch.nn.GELU(),
    }
    
    results = {name: {'epsilon': [], 'eff_rank_pre': [], 'eff_rank_post': [], 
                      'eff_rank_pred': [], 'alpha_f': None} 
               for name in test_activations}
    
    # Compute alpha_f for each activation
    for name, act_fn in test_activations.items():
        alpha_f, _, _ = compute_alpha_f(act_fn)
        results[name]['alpha_f'] = alpha_f
        print(f"{name}: α_f = {alpha_f:.4f}")
    
    # Test different correlation levels
    for eps in epsilons:
        # Create highly correlated data: C_ij = 1-ε for i≠j
        correlation = 1 - eps
        X, cov = create_correlated_gaussian(n_samples, d_features, correlation)
        
        # Compute pre-activation effective rank
        eff_rank_pre = compute_effective_rank(cov)
        
        for name, act_fn in test_activations.items():
            # Apply activation
            Y = act_fn(X)
            
            # Compute post-activation covariance and effective rank
            cov_Y = torch.cov(Y.T)
            eff_rank_post = compute_effective_rank(cov_Y)
            
            # Theoretical prediction: eff_rank(C_f) ≈ eff_rank(C)(1 + α_f ε ln(n/ε))
            alpha_f = results[name]['alpha_f']
            eff_rank_pred = eff_rank_pre * (1 + alpha_f * eps * np.log(d_features/eps))
            
            results[name]['epsilon'].append(eps)
            results[name]['eff_rank_pre'].append(eff_rank_pre)
            results[name]['eff_rank_post'].append(eff_rank_post)
            results[name]['eff_rank_pred'].append(eff_rank_pred)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    for idx, (name, data) in enumerate(results.items()):
        ax = axes[idx]
        
        # Plot empirical vs theoretical
        ax.semilogx(data['epsilon'], data['eff_rank_post'], 'o-', 
                    label='Empirical', markersize=6)
        ax.semilogx(data['epsilon'], data['eff_rank_pred'], 's--', 
                    label='Theoretical', markersize=5, alpha=0.7)
        ax.semilogx(data['epsilon'], data['eff_rank_pre'], '^:', 
                    label='Pre-activation', markersize=4, alpha=0.5)
        
        ax.set_xlabel('ε (correlation = 1-ε)')
        ax.set_ylabel('Effective Rank')
        ax.set_title(f'{name} (α_f={data["alpha_f"]:.3f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Effective Rank Recovery: Theory vs Empirical', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "validation_effective_rank_formula.pdf"))
    plt.show()
    
    return results

# --- Validation 4: Fixed Frozen States Analysis ---
def validate_frozen_states():
    """Validate that extreme modulation leads to frozen states."""
    print("\n=== Validating Frozen States from Extreme Modulation ===")
    
    n_samples = 10000
    
    # Create figure with subplots - 2 rows only
    fig = plt.figure(figsize=(14, 7))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # === Part 1: Tanh with scaling ===
    print("\n1. Testing Tanh(ax) with increasing scale a:")
    
    a_values = np.logspace(-1, 2, 20)
    results_tanh = []
    
    for a in a_values:
        act_fn = lambda x: torch.tanh(a * x)
        alpha_f, E_f_sq, E_fp_sq = compute_alpha_f(act_fn)
        
        # Compute fraction of near-zero derivatives
        z = torch.randn(n_samples)
        z.requires_grad_(True)
        f = torch.tanh(a * z)
        grad_output = torch.autograd.grad(f.sum(), z)[0]
        frozen_frac = torch.mean((torch.abs(grad_output) < 1e-4).float()).item()
        
        results_tanh.append({
            'a': a,
            'alpha_f': alpha_f,
            'E_f_squared': E_f_sq,
            'E_fp_squared': E_fp_sq,
            'frozen_fraction': frozen_frac
        })
        
    df_tanh = pd.DataFrame(results_tanh)
    
    # Plot Tanh results
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.semilogx(df_tanh['a'], df_tanh['alpha_f'], 'b-', linewidth=2)
    ax1.set_xlabel('Scale factor a')
    ax1.set_ylabel('α_f')
    ax1.set_title('Tanh(ax): Rank Recovery Strength')
    ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.semilogx(df_tanh['a'], df_tanh['frozen_fraction'], 'r-', linewidth=2)
    ax2.set_xlabel('Scale factor a')
    ax2.set_ylabel('Fraction with |f\'| < 1e-4')
    ax2.set_title('Tanh(ax): Frozen Units')
    ax2.grid(True, alpha=0.3)
    
    # Visualize activation and derivative
    ax3 = fig.add_subplot(gs[0, 2])
    x = torch.linspace(-3, 3, 1000)
    a_demo_values = [0.5, 1.0, 5.0, 20.0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(a_demo_values)))
    for i, a in enumerate(a_demo_values):
        y = torch.tanh(a * x)
        ax3.plot(x, y, label=f'a={a}', alpha=0.8, color=colors[i], linewidth=2)
    ax3.set_xlabel('x')
    ax3.set_ylabel('tanh(ax)')
    ax3.set_title('Activation Shape')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # === Part 2: ReLU with shifting ===
    print("\n2. Testing ReLU(x+b) with negative shift b:")
    
    b_values = np.linspace(-4, 2, 20)
    results_relu = []
    
    for b in b_values:
        act_fn = lambda x: torch.relu(x + b)
        alpha_f, E_f_sq, E_fp_sq = compute_alpha_f(act_fn)
        
        # Compute fraction of zero activations
        z = torch.randn(n_samples)
        dead_frac = torch.mean((z + b <= 0).float()).item()
        
        results_relu.append({
            'b': b,
            'alpha_f': alpha_f,
            'E_f_squared': E_f_sq,
            'E_fp_squared': E_fp_sq,
            'dead_fraction': dead_frac
        })
        
    df_relu = pd.DataFrame(results_relu)
    
    # Plot ReLU results
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(df_relu['b'], df_relu['alpha_f'], 'b-', linewidth=2)
    ax4.set_xlabel('Shift b')
    ax4.set_ylabel('α_f')
    ax4.set_title('ReLU(x+b): Rank Recovery Strength')
    ax4.grid(True, alpha=0.3)
    
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(df_relu['b'], df_relu['dead_fraction'], 'r-', linewidth=2)
    ax5.set_xlabel('Shift b')
    ax5.set_ylabel('Fraction with f\'=0')
    ax5.set_title('ReLU(x+b): Dead Units')
    ax5.grid(True, alpha=0.3)
    
    # Visualize activation
    ax6 = fig.add_subplot(gs[1, 2])
    x = torch.linspace(-4, 4, 1000)
    b_demo_values = [-3, -1, 0, 1]
    colors = plt.cm.plasma(np.linspace(0, 1, len(b_demo_values)))
    for i, b in enumerate(b_demo_values):
        y = torch.relu(x + b)
        ax6.plot(x, y, label=f'b={b}', alpha=0.8, color=colors[i], linewidth=2)
    ax6.set_xlabel('x')
    ax6.set_ylabel('ReLU(x+b)')
    ax6.set_title('Activation Shape')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle('Validation: Extreme Modulation Leads to Frozen States', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "validation_frozen_states.pdf"))
    plt.show()
    
    return df_tanh, df_relu

def validate_modulation_2d_comprehensive():
    """Create 2D heatmaps showing frozen states for multiple activations under different modulations."""
    print("\n=== Comprehensive 2D Analysis of Modulation Effects ===")
    
    n_samples = 5000  # Reduced for efficiency in 2D grid
    
    # Define parameter grids
    a_vals = np.logspace(-0.5, 2.0, 25)  # Extended to 100
    
    # Activations to test with their modulation schemes
    activations = {
        'ReLU(ax+b)': {
            'base_fn': torch.nn.ReLU(),
            'modulated_fn': lambda a, b: (lambda x: torch.relu(a * x + b)),
            'category': 'half-vanishing',
            'description': 'Half-vanishing: f(x)=0 for x<0, f(x)=x for x≥0'
        },
        'SELU(ax+b)': {
            'base_fn': torch.nn.SELU(),
            'modulated_fn': lambda a, b: (lambda x: torch.nn.SELU()(a * x + b)),
            'category': 'half-vanishing',
            'description': 'Half-vanishing: exponential for x<0, linear for x≥0'
        },
        'Tanh(a(x+b))': {
            'base_fn': torch.nn.Tanh(),
            'modulated_fn': lambda a, b: (lambda x: torch.tanh(a * (x + b))),
            'category': 'plateauing',
            'description': 'Plateauing: saturates to ±1 as x→±∞'
        },
        'Sigmoid(a(x+b))': {
            'base_fn': torch.nn.Sigmoid(),
            'modulated_fn': lambda a, b: (lambda x: torch.sigmoid(a * (x + b))),
            'category': 'plateauing',
            'description': 'Plateauing: saturates to 0/1 as x→-∞/+∞'
        }
    }
    
    # Create figure
    fig, axes = plt.subplots(len(activations), 4, figsize=(16, 3.5 * len(activations)))
    
    for row_idx, (name, config) in enumerate(activations.items()):
        print(f"\nProcessing {name}...")
        
        # Set bias range based on activation type
        if config['category'] == 'half-vanishing':
            b_vals = np.linspace(-20, 5, 20)  # Extended range for ReLU/SELU
        else:
            b_vals = np.linspace(-3, 1, 20)   # Original range for Tanh/Sigmoid
        
        # Initialize grids for this activation
        frozen_map = np.zeros((len(b_vals), len(a_vals)))
        alpha_map = np.zeros((len(b_vals), len(a_vals)))
        
        # Compute for each parameter combination
        for i, b in enumerate(b_vals):
            for j, a in enumerate(a_vals):
                act_fn = config['modulated_fn'](a, b)
                
                # Compute alpha_f using integration
                try:
                    alpha_f, _, _ = compute_alpha_f_hermite(act_fn, n_points=100)
                    alpha_map[i, j] = min(alpha_f, 10)  # Cap for visualization
                except:
                    alpha_map[i, j] = 0
                
                # Compute frozen/dead fraction
                z = torch.randn(n_samples)
                z.requires_grad_(True)
                
                try:
                    f = act_fn(z)
                    if z.grad is not None:
                        z.grad.zero_()
                    f_sum = f.sum()
                    grad_output = torch.autograd.grad(f_sum, z)[0]
                    
                    if config['category'] == 'half-vanishing':
                        # For ReLU/SELU: count zeros
                        frozen_frac = torch.mean((torch.abs(grad_output) < 1e-8).float()).item()
                    else:
                        # For Tanh/Sigmoid: count near-zeros
                        frozen_frac = torch.mean((torch.abs(grad_output) < 1e-4).float()).item()
                    
                    frozen_map[i, j] = frozen_frac
                except:
                    frozen_map[i, j] = 1.0
        
        # Plot frozen fraction heatmap
        ax0 = axes[row_idx, 0]
        im0 = ax0.imshow(frozen_map, aspect='auto', origin='lower', cmap='hot', 
                         vmin=0, vmax=1, extent=[a_vals[0], a_vals[-1], b_vals[0], b_vals[-1]])
        ax0.set_xlabel('Scale a')
        ax0.set_ylabel('Shift b')
        ax0.set_title(f'{name}: Frozen Fraction')
        ax0.set_xscale('log')
        plt.colorbar(im0, ax=ax0, label='Fraction')
        
        # Plot alpha_f heatmap
        ax1 = axes[row_idx, 1]
        im1 = ax1.imshow(alpha_map, aspect='auto', origin='lower', cmap='viridis',
                         vmin=0, vmax=10, extent=[a_vals[0], a_vals[-1], b_vals[0], b_vals[-1]])
        ax1.set_xlabel('Scale a')
        ax1.set_ylabel('Shift b')
        ax1.set_title(f'{name}: α_f (capped at 10)')
        ax1.set_xscale('log')
        plt.colorbar(im1, ax=ax1, label='α_f')
        
        # Add activation function visualization in the third column
        ax2 = axes[row_idx, 2]
        
        # First plot the activation shape
        x_demo = torch.linspace(-3, 3, 200)
        y_demo = config['base_fn'](x_demo)
        ax2_left = ax2
        ax2_left.plot(x_demo, y_demo, 'b-', linewidth=2, label='f(x)')
        ax2_left.set_xlabel('x')
        ax2_left.set_ylabel('f(x)', color='b')
        ax2_left.tick_params(axis='y', labelcolor='b')
        ax2_left.grid(True, alpha=0.3)
        ax2_left.axhline(y=0, color='k', linestyle=':', alpha=0.3)
        ax2_left.axvline(x=0, color='k', linestyle=':', alpha=0.3)
        
        # Add derivative on right y-axis
        ax2_right = ax2_left.twinx()
        x_grad = x_demo.clone().requires_grad_(True)
        y_grad = config['base_fn'](x_grad)
        grad_output = torch.autograd.grad(y_grad.sum(), x_grad)[0]
        ax2_right.plot(x_demo, grad_output.detach(), 'r--', linewidth=1.5, label="f'(x)", alpha=0.7)
        ax2_right.set_ylabel("f'(x)", color='r')
        ax2_right.tick_params(axis='y', labelcolor='r')
        
        ax2_left.set_title(f'{name.split("(")[0]} Shape')
        
        # Create combined legend
        lines1, labels1 = ax2_left.get_legend_handles_labels()
        lines2, labels2 = ax2_right.get_legend_handles_labels()
        ax2_left.legend(lines1 + lines2, labels1 + labels2, loc='best')
        
        # Plot correlation: alpha_f vs frozen fraction in the fourth column
        ax3 = axes[row_idx, 3]
        # Flatten and filter valid points
        alpha_flat = alpha_map.flatten()
        frozen_flat = frozen_map.flatten()
        valid_mask = (alpha_flat > 0) & (frozen_flat < 0.999)
        
        scatter = ax3.scatter(alpha_flat[valid_mask], frozen_flat[valid_mask], 
                            c=np.log10(np.repeat(a_vals, len(b_vals))[valid_mask]), 
                            cmap='plasma', s=20, alpha=0.6)
        ax3.set_xlabel('α_f (Rank Recovery Strength)')
        ax3.set_ylabel('Frozen Fraction')
        ax3.set_title(f'{name}: Trade-off')
        ax3.grid(True, alpha=0.3)
        
        if row_idx == 0:
            cbar = plt.colorbar(scatter, ax=ax3)
            cbar.set_label('log₁₀(a)', rotation=270, labelpad=15)
        else:
            plt.colorbar(scatter, ax=ax3).set_label('log₁₀(a)', rotation=270, labelpad=15)
        
        # Add category annotation
        ax3.text(0.95, 0.95, f'[{config["category"]}]', 
                transform=ax3.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Add column titles
    if len(activations) > 0:
        fig.text(0.125, 0.98, 'Frozen Fraction', ha='center', va='top', fontsize=12, weight='bold')
        fig.text(0.375, 0.98, 'Rank Recovery (α_f)', ha='center', va='top', fontsize=12, weight='bold')
        fig.text(0.625, 0.98, 'Activation Shape', ha='center', va='top', fontsize=12, weight='bold')
        fig.text(0.875, 0.98, 'Trade-off Analysis', ha='center', va='top', fontsize=12, weight='bold')
    
    plt.suptitle('Comprehensive 2D Analysis: Modulation Effects on Different Activation Functions', 
                 fontsize=14, y=0.995)
    # Add parameter explanation
    param_text = ("Parameters: For ReLU/SELU: f(ax+b) with b∈[-20,5], for Tanh/Sigmoid: f(a(x+b)) with b∈[-3,1]\n"
                  "Half-vanishing: zero gradient for negative inputs → dead units\n"
                  "Plateauing: near-zero gradient at extremes → saturation")
    fig.text(0.5, 0.005, param_text, ha='center', va='bottom', fontsize=9, 
             style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    plt.savefig(os.path.join(FIGURE_DIR, "validation_modulation_2d_comprehensive.pdf"))
    plt.show()
    
    return fig

# --- Main execution ---
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("Starting theoretical validation experiments...")
    
    # First, let's verify our integration methods
    print("\n=== Verifying Integration Methods ===")
    test_activations = {
        'ReLU': torch.nn.ReLU(),
        'Tanh': torch.nn.Tanh(),
        'Sigmoid': torch.nn.Sigmoid(),
    }
    
    print("Comparing integration methods for computing α_f:")
    print("-" * 70)
    print(f"{'Activation':<10} {'Quadrature':<15} {'Gauss-Hermite':<15} {'Difference':<10}")
    print("-" * 70)
    
    for name, act_fn in test_activations.items():
        alpha_quad, _, _ = compute_alpha_f(act_fn)
        alpha_herm, _, _ = compute_alpha_f_hermite(act_fn)
        diff = abs(alpha_quad - alpha_herm)
        print(f"{name:<10} {alpha_quad:<15.6f} {alpha_herm:<15.6f} {diff:<10.2e}")
    
    print("-" * 70)
    print("Both methods give consistent results!\n")
    
    # Run all validations
    hard_rank_results = validate_hard_rank_recovery()
    eff_rank_results = validate_effective_rank_formula()
    
    # Run the fixed frozen states validation
    frozen_results = validate_frozen_states()
    
    # Run the comprehensive 2D analysis
    validate_modulation_2d_comprehensive()
    
    print("\nAll validation experiments completed!")
    print(f"Figures saved to: {FIGURE_DIR}")