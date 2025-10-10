import torch
import torch.nn as nn
from .layers import get_activation, get_normalization


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

        self.layers = nn.ModuleDict()
        for i in range(len(hidden_sizes)):
            if use_gated_ffn and i > 0:
                self.layers[f"gated_ffn_block_{i}"] = GatedFFNBlock(
                    hidden_sizes[i - 1],
                    hidden_sizes[i],
                    hidden_sizes[i],
                    activation=gated_ffn_activation,
                    bias=bias,
                )
            else:
                self.layers[f"fc_{i}"] = nn.Linear(hidden_sizes[i - 1], hidden_sizes[i], bias=bias)

            if norm_after_activation:
                self.layers[f"act_{i}"] = get_activation(activation)
                if normalization:
                    self.layers[f"norm_{i}"] = get_normalization(
                        normalization, hidden_sizes[i], affine=normalization_affine
                    )
            else:
                if normalization:
                    self.layers[f"norm_{i}"] = get_normalization(
                        normalization, hidden_sizes[i], affine=normalization_affine
                    )
                self.layers[f"act_{i}"] = get_activation(activation)

            if dropout_p > 0:
                self.layers[f"drop_{i}"] = nn.Dropout(dropout_p)

        self.layers["out"] = nn.Linear(hidden_sizes[-1], output_size, bias=bias)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        for _, layer in self.layers.items():
            x = layer(x)

        return x


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
