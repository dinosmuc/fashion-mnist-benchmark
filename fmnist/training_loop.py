"""Reusable training loop code"""

import time

import numpy as np
import torch

from .data import SEED
from .early_stopping import EarlyStopping

PATIENCE = 3
DELTA = 0.001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = SEED) -> None:
    """
    Fixes the seeds so that results are reproducible.
    cuDNN picks convolution algorithms non-deterministically by default, which
    is enough to change CNN results between runs, so that is disabled too.
    """

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_step(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device = DEVICE,
) -> float:
    """One pass over data_loader that backpropagates and updates the weights.

    Returns the average batch loss.
    """

    train_loss = 0.0
    model.train()

    # Training process
    for X, y in data_loader:
        X, y = X.to(device), y.to(device)

        y_train_pred = model(X)

        loss = loss_fn(y_train_pred, y)

        train_loss += loss.item()

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    return train_loss / len(data_loader)


def eval_step(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device = DEVICE,
) -> float:
    """One pass over data_loader with gradients disabled, leaving the weights untouched.

    Returns the average batch loss.
    """

    eval_loss = 0.0
    model.eval()

    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            eval_loss += loss_fn(model(X), y).item()

    return eval_loss / len(data_loader)


def train_model(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    early_stopping: EarlyStopping | None = None,
    device: torch.device = DEVICE,
) -> tuple[torch.nn.Module, dict[str, list[float]], float]:
    """Trains with early stopping on the validation split.

    Returns (model, history, train_time). The returned model carries the weights
    of the best validation epoch.
    """

    if early_stopping is None:
        early_stopping = EarlyStopping(patience=PATIENCE, delta=DELTA, verbose=True)

    model.to(device)
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        train_loss = train_step(model, train_loader, loss_fn, optimizer, device=device)
        val_loss = eval_step(model, val_loader, loss_fn, device=device)

        print(f"Epoch {epoch}/{epochs} | train loss {train_loss:.5f} | val loss {val_loss:.5f}")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        early_stopping.check_early_stop(val_loss, model=model, epoch=epoch)

        if early_stopping.stop_training:
            print(f"Early stopping triggered after epoch {epoch}")
            break

    if device.type == "cuda":
        torch.cuda.synchronize()
    train_time = time.perf_counter() - start

    early_stopping.restore_best(model)

    return model, history, train_time


def predict(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device = DEVICE,
) -> tuple[np.ndarray, np.ndarray]:
    """Runs model over data_loader.

    Returns (y_true, y_pred) as NumPy arrays in the form evaluate_split expects.
    """

    model.eval()
    model.to(device)

    y_true, y_pred = [], []

    with torch.inference_mode():
        for X, y in data_loader:
            logits = model(X.to(device))
            y_pred.append(logits.argmax(dim=1).cpu())
            y_true.append(y)

    return torch.cat(y_true).numpy(), torch.cat(y_pred).numpy()
