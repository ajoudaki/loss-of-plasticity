import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# --- Matplotlib Styling for Paper Quality ---
try:
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        'font.family': 'serif',
        'text.usetex': False, # Set to True if LaTeX is installed and you want LaTeX fonts
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

# --- Global Parameters & Figure Saving ---
D_FULL_GLOBAL = 20
K_RANK_GLOBAL = 5
N_SAMPLES_GLOBAL = 10000
ABS_TOL_SVD_RANK_GLOBAL = 1e-7
FIGURE_DIR = "./figures/"
os.makedirs(FIGURE_DIR, exist_ok=True)

# --- Helper Functions ---
def compute_effective_rank(matrix_or_svals):
    if isinstance(matrix_or_svals, torch.Tensor) and matrix_or_svals.ndim == 2:
        s_unnormalized = torch.linalg.svdvals(matrix_or_svals).detach().cpu().numpy()
    elif isinstance(matrix_or_svals, torch.Tensor) and matrix_or_svals.ndim == 1:
        s_unnormalized = matrix_or_svals.detach().cpu().numpy()
    elif isinstance(matrix_or_svals, np.ndarray) and matrix_or_svals.ndim == 1:
        s_unnormalized = matrix_or_svals
    elif isinstance(matrix_or_svals, np.ndarray) and matrix_or_svals.ndim == 2:
        s_unnormalized = np.linalg.svd(matrix_or_svals, compute_uv=False)
    else:
        raise ValueError("Input must be a 2D matrix or 1D singular values (torch.Tensor or np.ndarray)")

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

def get_matrix_analysis(matrix_torch, name="Matrix", tol_abs=None):
    svals_unnormalized_torch = torch.linalg.svdvals(matrix_torch)
    svals_unnormalized_np = svals_unnormalized_torch.detach().cpu().numpy()

    if tol_abs is None:
        tol_rank_calc = svals_unnormalized_np.max() * max(matrix_torch.shape) * np.finfo(svals_unnormalized_np.dtype).eps
    else:
        tol_rank_calc = tol_abs
    rank_svd_threshold = np.sum(svals_unnormalized_np > tol_rank_calc).item()

    sum_svals_unnormalized = np.sum(svals_unnormalized_np)
    svals_normalized_for_plot_np = np.zeros_like(svals_unnormalized_np)
    if sum_svals_unnormalized > 1e-12:
        svals_normalized_for_plot_np = svals_unnormalized_np / sum_svals_unnormalized

    eff_rank = compute_effective_rank(svals_unnormalized_np)
    return rank_svd_threshold, svals_normalized_for_plot_np, eff_rank, svals_unnormalized_np

def generate_base_features(n_samples=N_SAMPLES_GLOBAL, k_rank=K_RANK_GLOBAL, d_full=D_FULL_GLOBAL):
    L_base = torch.randn(n_samples, k_rank)
    A_coeffs_base = torch.randn(d_full, k_rank)
    A_row_norms_for_corr_structure = torch.linalg.norm(A_coeffs_base, dim=1, keepdim=True)
    A_normalized_for_corr_structure = A_coeffs_base / (A_row_norms_for_corr_structure + 1e-9)
    z_base = L_base @ A_normalized_for_corr_structure.T
    return z_base

# --- Figure 1.1: Impact of tanh(az) ---
def generate_tanh_az_data(z_base):
    a_params = [0.01, 0.1, 1.0, 5.0, 20.0, 100.0]
    results = []

    cov_input = torch.cov(z_base.T)
    r_in, s_in, er_in, _ = get_matrix_analysis(cov_input, "Cov(z_input)")
    results.append({'a_label': 'Input ($z$)', 'a_val': 0.005, 'is_ref': True, 
                    'rank': r_in, 'svals_norm': s_in, 'eff_rank': er_in})

    for a in a_params:
        h = torch.tanh(a * z_base)
        cov_h = torch.cov(h.T)
        rank_h, svals_norm_h, eff_rank_h, _ = get_matrix_analysis(cov_h, f"Cov(tanh({a}x))", tol_abs=ABS_TOL_SVD_RANK_GLOBAL)
        results.append({'a_label': f"{a}", 'a_val': a, 'is_ref': False,
                        'rank': rank_h, 'svals_norm': svals_norm_h, 'eff_rank': eff_rank_h})

    h_sign = torch.sign(z_base)
    h_sign += torch.randn_like(h_sign) * 1e-6 
    cov_h_sign = torch.cov(h_sign.T)
    rank_sign, svals_norm_sign, eff_rank_sign, _ = get_matrix_analysis(cov_h_sign, "Cov(sign(x))", tol_abs=ABS_TOL_SVD_RANK_GLOBAL)
    results.append({'a_label': 'Sign($z$)', 'a_val': 200.0, 'is_ref': True, 
                    'rank': rank_sign, 'svals_norm': svals_norm_sign, 'eff_rank': eff_rank_sign})
    
    return pd.DataFrame(results)

def plot_tanh_az_figure(df_tanh_az):
    fig, axs = plt.subplots(1, 2, figsize=(7, 3.0)) 
    
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(df_tanh_az))) 
    markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X']
    
    for i, (_, row) in enumerate(df_tanh_az.iterrows()):
        label_str = row['a_label']
        axs[0].plot(np.arange(1, len(row['svals_norm']) + 1), row['svals_norm'], marker=markers[i % len(markers)], 
                    linestyle='-', label=label_str, color=colors[i], markersize=4)
    axs[0].set_title(f'Spectra (Input k={K_RANK_GLOBAL})')
    axs[0].set_xlabel("Eigenvalue Index")
    axs[0].set_ylabel("Norm. Eigenvalue")
    axs[0].set_yscale('log')
    axs[0].legend(title="Tanh($az$), $a$:", loc='upper right')
    axs[0].grid(True, which="both", ls=":", alpha=0.6)

    numeric_df = df_tanh_az[~df_tanh_az['is_ref']].copy()
    numeric_df['a_val'] = numeric_df['a_val'].astype(float)
    numeric_df = numeric_df.sort_values(by='a_val')
    axs[1].plot(numeric_df['a_val'], numeric_df['eff_rank'], marker='s', linestyle='-', color='teal', label='Tanh($az$) ER')
    
    ref_df = df_tanh_az[df_tanh_az['is_ref']].copy()
    for _, ref_row in ref_df.iterrows():
         axs[1].scatter(ref_row['a_val'], ref_row['eff_rank'], marker='*', s=60, label=f"{ref_row['a_label']} ER", 
                        color=colors[df_tanh_az[df_tanh_az['a_label']==ref_row['a_label']].index[0]], zorder=5)

    axs[1].set_title('Effective Rank vs. "$a$"')
    axs[1].set_xlabel('Scaling Factor "$a$"')
    axs[1].set_ylabel('Effective Rank')
    axs[1].set_xscale('log')
    axs[1].legend(loc='center right')
    axs[1].grid(True, which="both", ls=":", alpha=0.6)

    fig.suptitle(f'Impact of Non-linearity Strength in Tanh($az$) ($d_{{full}}$={D_FULL_GLOBAL})', fontsize=plt.rcParams['figure.titlesize'])
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(FIGURE_DIR, "theory_tanh_az_rank.pdf"))
    print(f"Saved: {os.path.join(FIGURE_DIR, 'theory_tanh_az_rank.pdf')}")
    plt.show()

# --- Figure 1.2: Impact of ReLU(z+b) ---
def generate_relu_zb_data(z_base):
    b_params = [-2.0, -1.0, -0.5, 0.0, 0.5, 2.0, 5.0]
    results = []

    cov_input = torch.cov(z_base.T)
    r_in, s_in, er_in, _ = get_matrix_analysis(cov_input, "Cov(z_input)")
    results.append({'b_label': 'Input ($z$)', 'b_val': min(b_params)-2.0, 'is_ref':True, 
                    'rank': r_in, 'svals_norm': s_in, 'eff_rank': er_in})

    for b in b_params:
        h = torch.relu(z_base + b)
        cov_h = torch.cov(h.T)
        rank_h, svals_norm_h, eff_rank_h, _ = get_matrix_analysis(cov_h, f"Cov(ReLU(z+{b}))", tol_abs=ABS_TOL_SVD_RANK_GLOBAL)
        results.append({'b_label': f"{b}", 'b_val': b, 'is_ref':False,
                        'rank': rank_h, 'svals_norm': svals_norm_h, 'eff_rank': eff_rank_h})
    return pd.DataFrame(results)

def plot_relu_zb_figure(df_relu_zb):
    fig, axs = plt.subplots(1, 2, figsize=(7, 3.0))
    indices = np.arange(1, D_FULL_GLOBAL + 1)
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(df_relu_zb)))
    markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X']

    for i, (_, row) in enumerate(df_relu_zb.iterrows()):
        label_str = row['b_label']
        axs[0].plot(indices, row['svals_norm'], marker=markers[i % len(markers)], linestyle='-', 
                    label=label_str, color=colors[i], markersize=4)
    axs[0].set_title(f'Spectra (Input k={K_RANK_GLOBAL})')
    axs[0].set_xlabel("Eigenvalue Index")
    axs[0].set_ylabel("Norm. Eigenvalue")
    axs[0].set_yscale('log')
    axs[0].legend(title="ReLU($z+b$), $b$:", loc='upper right')
    axs[0].grid(True, which="both", ls=":", alpha=0.6)

    numeric_df = df_relu_zb[~df_relu_zb['is_ref']].copy()
    numeric_df['b_val'] = numeric_df['b_val'].astype(float)
    numeric_df = numeric_df.sort_values(by='b_val')
    axs[1].plot(numeric_df['b_val'], numeric_df['eff_rank'], marker='s', linestyle='-', color='teal', label='ReLU($z+b$) ER')
    
    input_row = df_relu_zb[df_relu_zb['b_label'] == 'Input ($z$)']
    if not input_row.empty:
        axs[1].scatter(input_row['b_val'].iloc[0], input_row['eff_rank'].iloc[0], marker='*', s=60, 
                       label=f"Input ($z$) ER", color=colors[df_relu_zb[df_relu_zb['b_label']=='Input ($z$)'].index[0]], zorder=5)

    axs[1].set_title('Effective Rank vs. "$b$"')
    axs[1].set_xlabel('Shift Factor "$b$"')
    axs[1].set_ylabel('Effective Rank')
    axs[1].legend(loc='lower right')
    axs[1].grid(True, which="both", ls=":", alpha=0.6)
    
    fig.suptitle(f'Impact of Input Regime in ReLU($z+b$) ($d_{{full}}$={D_FULL_GLOBAL})', fontsize=plt.rcParams['figure.titlesize'])
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(FIGURE_DIR, "theory_relu_zb_rank.pdf"))
    print(f"Saved: {os.path.join(FIGURE_DIR, 'theory_relu_zb_rank.pdf')}")
    plt.show()

# --- Figure 1.3: Benefit of Joint Normalization and Activation ---
def generate_joint_norm_act_data(z_base_input, activation_name='ReLU'):
    d_full = z_base_input.shape[1]
    feature_scales_a = torch.rand(d_full) * 2.0  # a_j ~ U(0, 2)
    feature_shifts_b = torch.rand(d_full) * 1.0   # b_j ~ U(0, 1)
    z_perturbed = z_base_input * feature_scales_a + feature_shifts_b

    act_fn = torch.relu if activation_name == 'ReLU' else torch.tanh
    results = []

    cov_z_perturbed = torch.cov(z_perturbed.T)
    r_zp, s_zp, er_zp, _ = get_matrix_analysis(cov_z_perturbed, "Cov(z_perturbed)", tol_abs=ABS_TOL_SVD_RANK_GLOBAL)
    results.append({'type': r'$z_{\text{pert}}$ (Pre-Act)', 'rank': r_zp, 'svals_norm': s_zp, 'eff_rank': er_zp})

    h_direct = act_fn(z_perturbed)
    cov_h_direct = torch.cov(h_direct.T)
    r_hd, s_hd, er_hd, _ = get_matrix_analysis(cov_h_direct, f"Cov({activation_name}(z_perturbed))", tol_abs=ABS_TOL_SVD_RANK_GLOBAL)
    results.append({'type': fr'$\sigma(z_{{\text{{pert}}}})$ (Direct)', 'rank': r_hd, 'svals_norm': s_hd, 'eff_rank': er_hd})

    mean_zp = torch.mean(z_perturbed, dim=0, keepdim=True)
    std_zp = torch.std(z_perturbed, dim=0, keepdim=True)
    z_bn = (z_perturbed - mean_zp) / (std_zp + 1e-7)
    
    h_bn_act = act_fn(z_bn)
    cov_h_bn_act = torch.cov(h_bn_act.T)
    r_hbn, s_hbn, er_hbn, _ = get_matrix_analysis(cov_h_bn_act, f"Cov({activation_name}(BN(z_perturbed)))", tol_abs=ABS_TOL_SVD_RANK_GLOBAL)
    results.append({'type': fr'$\sigma(\text{{BN}}(z_{{\text{{pert}}}}))$ (BN+Act)', 'rank': r_hbn, 'svals_norm': s_hbn, 'eff_rank': er_hbn})
    
    return pd.DataFrame(results)

def plot_joint_norm_act_figure(df_relu_joint, df_tanh_joint):
    fig, axs = plt.subplots(1, 2, figsize=(7, 3.0), sharey=True)
    indices = np.arange(1, D_FULL_GLOBAL + 1)
    activation_dfs = {'ReLU': df_relu_joint, 'Tanh': df_tanh_joint}
    
    for plot_idx, (act_name, df_joint) in enumerate(activation_dfs.items()):
        ax = axs[plot_idx]
        colors = plt.cm.Dark2(np.linspace(0, 0.9, len(df_joint))) # Using Dark2 for distinct colors
        markers = ['o', 's', '^']
        for i, (_, row) in enumerate(df_joint.iterrows()):
            ax.plot(indices, row['svals_norm'], marker=markers[i % len(markers)], linestyle='-', 
                    label=f"{row['type']} (ER: {row['eff_rank']:.1f})", color=colors[i], markersize=4)
        # Corrected title line
        ax.set_title(f'{act_name} ($\\sigma$)') # Use f-string with escaped sigma
        ax.set_xlabel("Eigenvalue Index")
        if plot_idx == 0:
            ax.set_ylabel("Norm. Eigenvalue")
        ax.set_yscale('log')
        ax.legend(loc='upper right')
        ax.grid(True, which="both", ls=":", alpha=0.6)
        
    fig.suptitle(f'Synergy of Normalization and Activation ($k_{{in}}$={K_RANK_GLOBAL}, $d_{{full}}$={D_FULL_GLOBAL})', fontsize=plt.rcParams['figure.titlesize'])
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.text(0.5, 0.01, r"$z_{\text{pert}}$: per-feature $a_j x_j + b_j$ with $a_j \sim U[0,2], b_j \sim U[0,1]$", ha='center', va='bottom', fontsize=7)
    plt.savefig(os.path.join(FIGURE_DIR, "theory_joint_norm_activation_rank.pdf"))
    print(f"Saved: {os.path.join(FIGURE_DIR, 'theory_joint_norm_activation_rank.pdf')}")
    plt.show()

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    z_base_main = generate_base_features()
    
    print("\n--- Figure 1.1: Tanh(az) ---")
    df_tanh_az_data = generate_tanh_az_data(z_base_main.clone())
    plot_tanh_az_figure(df_tanh_az_data)

    print("\n--- Figure 1.2: ReLU(z+b) ---")
    df_relu_zb_data = generate_relu_zb_data(z_base_main.clone())
    plot_relu_zb_figure(df_relu_zb_data)

    print("\n--- Figure 1.3: Joint Normalization & Activation ---")
    df_relu_joint_data = generate_joint_norm_act_data(z_base_main.clone(), activation_name='ReLU')
    df_tanh_joint_data = generate_joint_norm_act_data(z_base_main.clone(), activation_name='Tanh')
    plot_joint_norm_act_figure(df_relu_joint_data, df_tanh_joint_data)

    print(f"\nAll theoretical figures saved to {FIGURE_DIR}")