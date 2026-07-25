"""Model implementations for the Fashion-MNIST benchmark."""

from torch import nn


class MultiLayerPerceptron(nn.Module):
    """MLP baseline: 784 -> 128 -> 10, as in TensorFlow 2024 tutorial."""

    def __init__(self, input_shape: int = 784, hidden_units: int = 128, output_shape: int = 10):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_shape),
        )

    def forward(self, x):
        return self.layer_stack(x)
