import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# Simulation Parameters
D_FEATURES = 1000  # d: feature size / rows of W / columns of W
N_SAMPLES = 1000   # n: batch size
LEARNING_RATE = 1e-5 # MODIFIED: Reduced Learning rate
N_STEPS = 100     # T: number of gradient descent steps
EPSILON_NORM_VAR = 1e-3    # Epsilon for stable division in BN/LN variance terms
EPSILON_RMS_DIV = 1e-8     # MODIFIED: Smaller epsilon for RMS W normalization division
SEED = 40         # For reproducibility

# Set seed for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)

# Device
if torch.cuda.is_available():
    device = torch.device("cuda")
# elif torch.backends.mps.is_available(): # For Apple Silicon GPUs
#     device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# Data (fixed throughout the simulation)
X = torch.randn(N_SAMPLES, D_FEATURES, device=device)

# --- Helper Functions ---

def initialize_weights_and_params():
    W = torch.randn(D_FEATURES, D_FEATURES, device=device) / np.sqrt(D_FEATURES)
    W.requires_grad_(True)
    a = torch.ones(D_FEATURES, device=device, requires_grad=True)
    b = torch.zeros(D_FEATURES, device=device, requires_grad=True)
    return W, a, b

def get_all_pairwise_cosine_similarities(W_matrix):
    d = W_matrix.shape[0]
    if d <= 1:
        return torch.empty(0, device=W_matrix.device)
    W_norm_rows = F.normalize(W_matrix, p=2, dim=1)
    sim_matrix = torch.matmul(W_norm_rows, W_norm_rows.transpose(0, 1))
    indices = torch.triu_indices(d, d, offset=1, device=W_matrix.device)
    if indices.numel() == 0:
        return torch.empty(0, device=W_matrix.device)
    pairwise_sims = sim_matrix[indices[0], indices[1]]
    return pairwise_sims

def global_rms_normalize_W(W_matrix): # Uses new EPSILON_RMS_DIV
    rms = torch.sqrt(torch.mean(W_matrix**2))
    if rms < EPSILON_RMS_DIV: # Check against the smaller division epsilon
        return W_matrix 
    return W_matrix / (rms + EPSILON_RMS_DIV)

# --- Model Configurations ---

def forward_vanilla(X_batch, W_matrix):
    Y = torch.matmul(X_batch, W_matrix)
    return Y

def forward_bn(X_batch, W_matrix, a_param, b_param): # Uses EPSILON_NORM_VAR
    Z = torch.matmul(X_batch, W_matrix)
    mu_feat = Z.mean(dim=0)
    var_feat = Z.var(dim=0, unbiased=False)
    Z_hat = (Z - mu_feat) / torch.sqrt(var_feat + EPSILON_NORM_VAR)
    Y = a_param * Z_hat + b_param
    return Y

def forward_ln(X_batch, W_matrix, a_param, b_param): # Uses EPSILON_NORM_VAR
    Z = torch.matmul(X_batch, W_matrix)
    mu_sample = Z.mean(dim=1, keepdim=True)
    var_sample = Z.var(dim=1, unbiased=False, keepdim=True)
    Z_hat = (Z - mu_sample) / torch.sqrt(var_sample + EPSILON_NORM_VAR)
    Y = a_param * Z_hat + b_param
    return Y

def forward_wn(X_batch, W_matrix, a_param, b_param):
    # F.normalize has its own internal epsilon, typically small (like 1e-12)
    W_tilde_rows = F.normalize(W_matrix, p=2, dim=1) 
    Z = torch.matmul(X_batch, W_tilde_rows)
    Y = a_param * Z + b_param
    return Y

# --- Simulation Loop ---

configurations = ["Vanilla", "Batch Norm", "Layer Norm", "Weight Norm"]
results_mean_cosine_similarity = {config: [] for config in configurations}
results_std_cosine_similarity = {config: [] for config in configurations}

params_dict = {}
for config in configurations:
    W, a, b = initialize_weights_and_params()
    params_dict[config] = {'W': W, 'a': a, 'b': b}
    initial_pairwise_sims = get_all_pairwise_cosine_similarities(W.detach())
    if initial_pairwise_sims.numel() > 0:
        results_mean_cosine_similarity[config].append(initial_pairwise_sims.mean().item())
        results_std_cosine_similarity[config].append(initial_pairwise_sims.std().item())
    else:
        results_mean_cosine_similarity[config].append(np.nan)
        results_std_cosine_similarity[config].append(np.nan)

failed_configs = set()

for t in range(N_STEPS):
    print(f"\n--- Step {t+1}/{N_STEPS} ---")
    for config_name in configurations:
        if config_name in failed_configs:
            results_mean_cosine_similarity[config_name].append(np.nan)
            results_std_cosine_similarity[config_name].append(np.nan)
            print(f"  Config: {config_name:<12} | SKIPPING (previously failed due to NaN/Inf)")
            continue

        W_curr = params_dict[config_name]['W']
        a_curr = params_dict[config_name]['a']
        b_curr = params_dict[config_name]['b']

        if W_curr.grad is not None: W_curr.grad.zero_()
        if a_curr.grad is not None: a_curr.grad.zero_()
        if b_curr.grad is not None: b_curr.grad.zero_()

        if config_name == "Vanilla":
            Y_pred = forward_vanilla(X, W_curr)
        elif config_name == "Batch Norm":
            Y_pred = forward_bn(X, W_curr, a_curr, b_curr)
        elif config_name == "Layer Norm":
            Y_pred = forward_ln(X, W_curr, a_curr, b_curr)
        elif config_name == "Weight Norm":
            Y_pred = forward_wn(X, W_curr, a_curr, b_curr)
        
        objective = torch.sum(Y_pred**2)
        current_objective_val = objective.item()
        
        objective_is_problematic = np.isnan(current_objective_val) or np.isinf(current_objective_val)

        if not objective_is_problematic:
            # Clear grads again just before backward if they were somehow populated by Y_pred checks
            if W_curr.grad is not None: W_curr.grad.zero_()
            if a_curr.grad is not None: a_curr.grad.zero_()
            if b_curr.grad is not None: b_curr.grad.zero_()
            objective.backward()
        
        W_next = W_curr 
        W_next_is_problematic = False

        with torch.no_grad():
            if not objective_is_problematic:
                grad_W = W_curr.grad if W_curr.grad is not None else torch.zeros_like(W_curr)
                # Check if gradient itself is problematic
                if torch.isnan(grad_W).any() or torch.isinf(grad_W).any():
                    W_next_is_problematic = True # Mark W_next as problematic if grad is bad
                    print(f"    WARNING: Problematic gradient for W in {config_name}.")
                else:
                    W_cand = W_curr + LEARNING_RATE * grad_W
                    W_next = global_rms_normalize_W(W_cand)
                    params_dict[config_name]['W'].data = W_next
                
                if config_name != "Vanilla":
                    grad_a = a_curr.grad if a_curr.grad is not None else torch.zeros_like(a_curr)
                    grad_b = b_curr.grad if b_curr.grad is not None else torch.zeros_like(b_curr)

                    if torch.isnan(grad_a).any() or torch.isinf(grad_a).any() or \
                       torch.isnan(grad_b).any() or torch.isinf(grad_b).any():
                        W_next_is_problematic = True # Also flag if a/b grads are bad
                        print(f"    WARNING: Problematic gradient for a/b in {config_name}.")
                    else:
                        if not W_next_is_problematic: # Only update a,b if W update was okay
                            a_next = a_curr + LEARNING_RATE * grad_a
                            b_next = b_curr + LEARNING_RATE * grad_b
                            params_dict[config_name]['a'].data = a_next
                            params_dict[config_name]['b'].data = b_next
            else: 
                W_next = params_dict[config_name]['W'].data 

            W_next_is_problematic = W_next_is_problematic or torch.isnan(W_next).any().item() or torch.isinf(W_next).any().item()


        print(f"  Config: {config_name:<12} | Objective (||Y||^2): {current_objective_val:10.3e} "
              f"| W_next Problematic: {str(W_next_is_problematic):<5}")

        if objective_is_problematic or W_next_is_problematic:
            print(f"    WARNING: Problem (NaN/Inf) detected for {config_name}. Skipping future steps.")
            failed_configs.add(config_name)
            results_mean_cosine_similarity[config_name].append(np.nan)
            results_std_cosine_similarity[config_name].append(np.nan)
        else:
            all_pairwise_sims = get_all_pairwise_cosine_similarities(W_next.detach())
            if all_pairwise_sims.numel() > 0:
                results_mean_cosine_similarity[config_name].append(all_pairwise_sims.mean().item())
                results_std_cosine_similarity[config_name].append(all_pairwise_sims.std().item())
            else:
                results_mean_cosine_similarity[config_name].append(np.nan)
                results_std_cosine_similarity[config_name].append(np.nan)

# --- Plotting Results ---
plt.figure(figsize=(14, 9))
x_axis = np.arange(N_STEPS + 1)

for config_name in configurations:
    mean_values = np.array(results_mean_cosine_similarity[config_name])
    std_values = np.array(results_std_cosine_similarity[config_name])
    
    valid_indices = ~np.isnan(mean_values) & ~np.isnan(std_values)
    
    if np.any(valid_indices):
        label_name = config_name
        if config_name in failed_configs:
             label_name += " (encountered NaN/Inf)" # Updated label for clarity
        plt.plot(x_axis[valid_indices], mean_values[valid_indices], label=label_name)
        plt.fill_between(x_axis[valid_indices], 
                         mean_values[valid_indices] - std_values[valid_indices], 
                         mean_values[valid_indices] + std_values[valid_indices], 
                         alpha=0.2)
    else:
        print(f"Plotting: No valid (non-NaN) data to plot for {config_name}")

# MODIFIED: Corrected LaTeX for \pm and formatting for LR
title_str = (f'Mean Pairwise Row Cosine Similarity of W (d={D_FEATURES}, n={N_SAMPLES}, lr={LEARNING_RATE:.1e})'
             f' with $\\pm$1 Std Dev') # Use \\pm or r"$\pm$"
plt.title(title_str)
plt.xlabel('Gradient Ascent Step (t)')
plt.ylabel('Mean Cosine Similarity $E[c(t)]$')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("mean_std_cosine_similarity_W_rows_adjusted.png")
plt.show()

print("Simulation finished. Plot saved as mean_std_cosine_similarity_W_rows_adjusted.png")