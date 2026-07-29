"""Reusable training loop code"""

import torch

from .early_stopping import early_stopping

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
early_stopping = early_stopping

def train_step(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    early_stopping: EarlyStopping = early_stopping,
    device: torch.device = device,
):
    train_loss = 0.0
    model.train()
    model.to(device)

    # Training process
    for X, y in data_loader:
        X, y = X.to(device), y.to(device)

        y_train_pred = model(X)

        loss = loss_fn(y_train_pred, y)

        train_loss += loss.item()

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    # Calculating loss per epoch and printing what is happening
    train_loss /= len(data_loader)
    print(f"Train Loss: {train_loss:.5f}")


def test_step(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    early_stopping: EarlyStopping = None,
    device: torch.device = device,
):

    test_loss = 0
    model.eval()
    model.to(device)

    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)

            y_test_pred = model(X)

            loss = loss_fn(y_test_pred, y)

            test_loss += loss.item()

        # Avarage test loss per epoch
        test_loss /= len(data_loader)

        # Check early stopping condition if provided
        if early_stopping is not None:
            early_stopping.check_early_stop(test_loss)

        print(f"Test Loss: {test_loss:.5f}")
