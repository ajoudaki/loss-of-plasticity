from sympy import false
import torch
import torch.nn as nn
from .layers import get_activation, get_normalization

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class MLP(nn.Module):
    def __init__(
        self,
        input_size=784,
        hidden_sizes=[512, 256, 128],
        output_size=10,
        activation="relu",
        dropout_p=0.0,
        normalization=None,
        norm_after_activation=False,
        bias=True,
        normalization_affine=True,
        use_gated_ffn=False,
        gated_ffn_activation="relu",
        eigenval_reg_lambda: float = 0.000001,
        eigenval_reg_momentum: float = 0.99,
        **kwargs
    ):
        """Fully connected MLP with customizable architecture."""
        super(MLP, self).__init__()

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.norm_after_activation = norm_after_activation
        self.use_gated_ffn = use_gated_ffn
        self.gated_ffn_activation = gated_ffn_activation
        self.eigenval_reg_lambda: float = eigenval_reg_lambda
        self.eigenval_reg_momentum: float = eigenval_reg_momentum

        if self.eigenval_reg_lambda > 0:
            self.register_buffer("running_cov_input", torch.eye(input_size))

        self.layers = nn.ModuleDict()
        in_features = input_size
        for i, hidden_size in enumerate(hidden_sizes):
            if use_gated_ffn and i > 0:
                self.layers[f"gated_ffn_block_{i}"] = GatedFFNBlock(
                    in_features,
                    hidden_size,
                    hidden_size,
                    activation=gated_ffn_activation,
                    bias=bias,
                )
            else:
                self.layers[f"fc_{i}"] = nn.Linear(in_features, hidden_size, bias=bias)
            
            if self.eigenval_reg_lambda > 0:
                self.register_buffer(f"running_cov_fc_{i}", torch.eye(hidden_size))

            if norm_after_activation:
                self.layers[f"act_{i}"] = get_activation(activation)
                if normalization:
                    self.layers[f"norm_{i}"] = get_normalization(normalization, hidden_size, affine=normalization_affine)
            else:
                if normalization:
                    self.layers[f"norm_{i}"] = get_normalization(normalization, hidden_size, affine=normalization_affine)
                self.layers[f"act_{i}"] = get_activation(activation)

            if dropout_p > 0:
                self.layers[f"drop_{i}"] = nn.Dropout(dropout_p)

            in_features = hidden_size

        self.layers["out"] = nn.Linear(in_features, output_size, bias=bias)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        if self.training and self.eigenval_reg_lambda > 0:
            centered_x = x - x.mean(dim=0)
            batch_cov_input = (centered_x.T @ centered_x) / (x.size(0) - 1)
            self.running_cov_input = self.eigenval_reg_momentum * self.running_cov_input + (1 - self.eigenval_reg_momentum) * batch_cov_input.detach()

        for name, layer in self.layers.items():
            x = layer(x)
            if self.training and self.eigenval_reg_lambda > 0 and name.startswith("fc_"):
                centered_h = x - x.mean(dim=0)
                batch_cov = (centered_h.T @ centered_h) / (x.size(0) - 1)
                setattr(self, f"running_cov_{name}", 
                        self.eigenval_reg_momentum * getattr(self, f"running_cov_{name}") + (1 - self.eigenval_reg_momentum) * batch_cov.detach())

        return x


def compute_cov_eigenval_regularization(model):
    reg_loss = 0.0
    for name, module in model.layers.items():
        if name.startswith("fc_"):
            layer_index = int(name.split("_")[1])  # e.g., "fc_0" -> 0
            W = module.weight

            if layer_index == 0:
                C_prev = model.running_cov_input
            else:
                C_prev = getattr(model, f"running_cov_fc_{layer_index-1}")
            
            C_curr_est = W @ C_prev @ W.t()
            
            # Spectral norm of C_curr_est (largest eigenvalue)
            # eigenvalues = torch.linalg.eigvalsh(C_curr_est)
            # lambda_max = eigenvalues.max()
            lambda_max = torch.linalg.norm(C_curr_est, ord=2)
            
            # Regularization matrix difference: (||C_l||_2 * I - C_curr_est)
            diff = lambda_max * torch.eye(C_curr_est.size(0), device=C_curr_est.device) - C_curr_est
            reg_loss += (diff**2).sum() / C_curr_est.size(0)
    return reg_loss


class GatedFFNBlock(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        activation="relu",
        bias=True,
    ):
        """Gated Feedforward Layer"""
        super(GatedFFNBlock, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation = get_activation(activation)

        self.layers = nn.ModuleDict()
        self.layers["fc_1"] = nn.Linear(input_size, hidden_size, bias=bias)
        self.layers["fc_2"] = nn.Linear(input_size, hidden_size, bias=bias)
        self.layers["fc_3"] = nn.Linear(hidden_size, output_size, bias=bias)


    def forward(self, x):
        return self.layers["fc_3"](self.activation(self.layers["fc_2"](x)) * self.layers["fc_1"](x))
