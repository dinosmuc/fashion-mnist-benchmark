"""Model implementations for the Fashion-MNIST benchmark."""

import torch
from torch import nn


class MultiLayerPerceptron(nn.Module):
    """MLP baseline: 784 -> 128 -> 10, as in TensorFlow 2024 tutorial."""

    def __init__(self, input_shape: int = 784, hidden_units: int = 128, output_shape: int = 10) -> None:
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_shape),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the class logits for a batch of images."""
        return self.layer_stack(x)


class ConvNet(nn.Module):
    """CNN baseline: simple MNIST convnet (Chollet, 2021)"""

    def __init__(self, input_shape: int = 1, output_shape: int = 10) -> None:
        super().__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(p=0.5), nn.Linear(in_features=5 * 5 * 64, out_features=output_shape)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the class logits for a batch of images."""
        x = self.conv_blocks(x)
        x = self.classifier(x)
        return x
