"""Fashion-MNIST loading: raw tensors, flat arrays, and DataLoaders."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"

CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

SEED = 42

VAL_SIZE = 6000  # 10% of the official training set, held out for early stopping


def get_dataset(root: Path = DATA_ROOT) -> tuple[Dataset, Dataset]:
    """Returns the train and test sets as tensors, downloading them on first use."""

    # Creating an instance of ToTensor class
    transform = transforms.ToTensor()

    # Downloading the dataset and storing it as tensors in data folder
    train_dataset = datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)

    return train_dataset, test_dataset


def get_numpy_data(root: Path = DATA_ROOT) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (X_train, y_train, X_test, y_test) as flat NumPy arrays - for the random forest."""

    # Tuple unpacking to get the dataset as tensors
    train_dataset, test_dataset = get_dataset(root)

    # Transformation of tensors to NumPy arrays and flattening
    X_train = train_dataset.data.numpy().reshape(len(train_dataset), -1)
    y_train = train_dataset.targets.numpy()

    X_test = test_dataset.data.numpy().reshape(len(test_dataset), -1)
    y_test = test_dataset.targets.numpy()

    return X_train, y_train, X_test, y_test


def get_dataloaders(
    batch_size: int = 128,
    val_size: int = VAL_SIZE,
    root: Path = DATA_ROOT,
    seed: int = SEED,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Returns train, validation and test loaders for the PyTorch models.

    The validation split is drawn from the official training set with a seeded generator,
    so the same images are held out on every run.
    """

    train_dataset, test_dataset = get_dataset(root)

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(
        train_dataset, [len(train_dataset) - val_size, val_size], generator=generator
    )

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader
