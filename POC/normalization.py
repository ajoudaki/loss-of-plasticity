import numpy as np
import torch
import matplotlib.pyplot as plt

class GradientAnalysis:
    """Theoretical analysis of gradient statistics for different configurations"""
    
    def __init__(self, d, mu=None, sigma=None):
        self.d = d
        
        # Set default mu and sigma if not provided
        if mu is None:
            self.mu = np.zeros(d)
        else:
            self.mu = mu
            
        if sigma is None:
            # Create PSD matrix with unit diagonals
            A = np.random.randn(d, d)
            self.sigma = A @ A.T
            # Normalize to have unit diagonals
            diag_sqrt = np.sqrt(np.diag(self.sigma))
            self.sigma = self.sigma / np.outer(diag_sqrt, diag_sqrt)
        else:
            self.sigma = sigma
            
    def vanilla_gradient_stats(self, W, n):
        """
        Compute mean and covariance of gradients for vanilla configuration
        dL/dW = -2(WX^T + b1^T)X = -2WX^TX - 2b1^TX
        """
        # E[X^TX] = n(Σ + μμ^T)
        E_XTX = n * (self.sigma + np.outer(self.mu, self.mu))
        
        # Mean gradient (assuming b=0)
        mean_grad_W = -2 * W @ E_XTX
        
        # For covariance, we need to compute E[(dL/dW)(dL/dW)^T]
        # This involves fourth moments of X
        # Simplified: assuming Gaussian, we use Isserlis' theorem
        
        # Variance of each gradient element
        # Var[dL/dW_ij] ≈ 4n * Var[sum_k W_ik X_kj X_lj]
        
        return mean_grad_W, E_XTX
    
    def batch_norm_gradient_stats(self, W, n):
        """
        Compute gradient statistics for batch norm
        More complex due to normalization
        """
        # After BN, activations have zero mean and unit variance
        # This decorrelates the gradients
        
        # Mean activations per feature
        mu_z = W @ self.mu  # d x 1
        
        # Variance of activations per feature  
        var_z = np.diag(W @ self.sigma @ W.T)  # d x 1
        
        # BN normalizes each feature to have zero mean and unit variance
        # This affects the gradient flow
        
        # Simplified analysis: gradient magnitudes are stabilized
        # E[dL/dW] involves the Jacobian of BN operation
        
        return mu_z, var_z
    
    def weight_norm_gradient_stats(self, W):
        """
        Compute gradient statistics for weight normalization
        Gradient is projected to be orthogonal to current weight direction
        """
        # Compute row norms
        row_norms = np.linalg.norm(W, axis=1, keepdims=True)
        W_normalized = W / (row_norms + 1e-8)
        
        # The gradient w.r.t. W has special structure:
        # It's orthogonal to the current weight direction
        # This creates a flow on the sphere
        
        return W_normalized, row_norms
    
    def compute_cosine_similarity_evolution(self, W, config='vanilla', steps=100, lr=0.01, n=100):
        """
        Theoretically predict how cosine similarity evolves
        """
        cos_sims = []
        W_current = W.copy()
        
        for step in range(steps):
            # Compute gradient based on configuration
            if config == 'vanilla':
                mean_grad, _ = self.vanilla_gradient_stats(W_current, n)
                # Simplified update
                W_current = W_current - lr * mean_grad
                
            elif config == 'weight_norm':
                W_norm, row_norms = self.weight_norm_gradient_stats(W_current)
                # Weight norm preserves angles better
                # Gradient is constrained to sphere
                grad_direction = np.random.randn(*W_current.shape)  # Simplified
                # Project gradient
                for i in range(W_current.shape[0]):
                    w_i = W_current[i]
                    grad_i = grad_direction[i]
                    # Remove component along w_i
                    grad_direction[i] = grad_i - (np.dot(grad_i, w_i) / np.dot(w_i, w_i)) * w_i
                
                W_current = W_current - lr * grad_direction
            
            # Normalize by RMS
            rms = np.sqrt(np.mean(W_current**2))
            W_current = W_current / rms
            
            # Compute average cosine similarity
            cos_sim = self.compute_avg_cosine_similarity(W_current)
            cos_sims.append(cos_sim)
            
        return cos_sims
    
    def compute_avg_cosine_similarity(self, W):
        """Compute average pairwise cosine similarity of rows"""
        # Normalize rows
        W_norm = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-8)
        
        # Compute all pairwise cosine similarities
        cos_sim_matrix = W_norm @ W_norm.T
        
        # Average upper triangular part (excluding diagonal)
        d = W.shape[0]
        upper_indices = np.triu_indices(d, k=1)
        avg_cos_sim = np.mean(cos_sim_matrix[upper_indices])
        
        return avg_cos_sim
    
    def plot_theoretical_predictions(self):
        """Plot theoretical predictions for different configurations"""
        d = self.d
        W_init = np.random.randn(d, d) / np.sqrt(d)
        
        configs = ['vanilla', 'weight_norm']
        plt.figure(figsize=(10, 6))
        
        for config in configs:
            cos_sims = self.compute_cosine_similarity_evolution(
                W_init, config=config, steps=200, lr=0.01, n=100
            )
            plt.plot(cos_sims, label=f'{config} (theoretical)', linestyle='--', linewidth=2)
        
        plt.xlabel('Gradient Step')
        plt.ylabel('Average Cosine Similarity')
        plt.title('Theoretical Prediction of Weight Evolution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

# Demonstrate the analysis
if __name__ == "__main__":
    # Create analyzer
    d = 50
    analyzer = GradientAnalysis(d)
    
    # Initial weight matrix
    W = np.random.randn(d, d) / np.sqrt(d)
    
    # Analyze vanilla gradients
    print("Vanilla Configuration Analysis:")
    mean_grad, E_XTX = analyzer.vanilla_gradient_stats(W, n=100)
    print(f"  Mean gradient norm: {np.linalg.norm(mean_grad):.4f}")
    print(f"  E[X^TX] max eigenvalue: {np.max(np.linalg.eigvals(E_XTX)):.4f}")
    
    # Analyze batch norm
    print("\nBatch Norm Analysis:")
    mu_z, var_z = analyzer.batch_norm_gradient_stats(W, n=100)
    print(f"  Mean activation norm: {np.linalg.norm(mu_z):.4f}")
    print(f"  Average activation variance: {np.mean(var_z):.4f}")
    
    # Analyze weight norm
    print("\nWeight Norm Analysis:")
    W_norm, row_norms = analyzer.weight_norm_gradient_stats(W)
    print(f"  Average row norm: {np.mean(row_norms):.4f}")
    print(f"  Row norm std: {np.std(row_norms):.4f}")
    
    # Plot theoretical predictions
    analyzer.plot_theoretical_predictions()