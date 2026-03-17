import torch
import torch.nn as nn
from typing import Callable, Sequence, Optional


class MLP(nn.Module):
    def __init__(
        self,
        layer_sizes: Sequence[int],
        activation: Callable = nn.ReLU,
        output_activation: Optional[nn.Module] = None,
        output_bias_init: float = 0.0,
    ):
        super().__init__()

        self.output_activation = output_activation

        layers = []

        # Hidden layers
        for in_size, out_size in zip(layer_sizes[:-2], layer_sizes[1:-1]):
            layers.append(nn.Linear(in_size, out_size))
            layers.append(activation())

        # Output layer
        self.output_layer = nn.Linear(layer_sizes[-2], layer_sizes[-1])

        # Custom bias initialization (critical for hover initialization)
        # nn.init.constant_(self.output_layer.bias, output_bias_init)

        self.hidden = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.hidden(x)
        x = self.output_layer(x)

        if self.output_activation is not None:
            x = self.output_activation(x)

        return x


class CNN(nn.Module):
    """Placeholder for a Convolutional Neural Network."""
    pass


class RNN(nn.Module):
    """Placeholder for a Recurrent Neural Network."""
    pass