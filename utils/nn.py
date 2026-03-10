import torch
import torch.nn as nn
from typing import Callable, Sequence, Optional


class MLP(nn.Module):
    """
    Multi-Layer Perceptron with optional output activation.

    Usage:
        # Standard MLP with ReLU hidden activations, linear output
        model = MLP([obs_dim, 64, 64, action_dim])

        # SRT controller: sigmoid output so actions are in [0, 1]
        # output_bias_init=0.0 → sigmoid(0) = 0.5 → mid-range motor speed ≈ hover
        model = MLP(
            layer_sizes=[obs_dim, 64, 64, action_dim],
            activation=nn.ReLU,
            output_activation=nn.Sigmoid(),
            output_bias_init=0.0,
        )

    Args:
        layer_sizes       : List of ints [input_dim, hidden..., output_dim]
        activation        : Hidden layer activation class (default: nn.ReLU)
        output_activation : Output layer activation (default: None = linear)
        output_bias_init  : Initial value for output layer bias.
                            Setting to 0.0 with sigmoid → outputs start at 0.5
                            which maps to mid-range motor speed (≈ hover).
    """

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
        nn.init.constant_(self.output_layer.bias, output_bias_init)

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