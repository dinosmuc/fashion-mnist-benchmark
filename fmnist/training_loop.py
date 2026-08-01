"""Reusable training loop code"""

import torch

from .early_stopping import EarlyStopping

PATIENCE = 3
DELTA = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_step(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
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
    return train_loss


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
        return test_loss

def train_model(model: torch.nn.Module,
                train_loader: torch.utils.data.DataLoader,
                test_loader: torch.utils.data.DataLoader,
                loss_fn: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                epochs: int,
                early_stopping: EarlyStopping=None,
                device: torch.device=device):

    if early_stopping is None:
        early_stopping = EarlyStopping(patience=PATIENCE, delta=DELTA, verbose=True)

    history = {"train_loss": [], "test_loss": []}

    for epoch in range(epochs):
        print(f"Epoch: {epoch + 1} / {epochs}")

        train_loss = train_step(model, train_loader, loss_fn, optimizer, device=device)
        test_loss = test_step(model, test_loader, loss_fn, early_stopping=early_stopping, device=device)

        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)

        if early_stopping.stop_training:
            print(f"Early stopping triggered after epoch {epoch + 1}")
            break

    return model, history
