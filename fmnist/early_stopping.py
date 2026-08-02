"""Early stopping class which is used in the training loop
source: https://medium.com/biased-algorithms/a-practical-guide-to-implementing-early-stopping-in-pytorch-for-model-training-99a7cbd46e9d"""

import copy

import torch


class EarlyStopping:
    """Stops training once the validation loss has not improved for `patience` epochs."""

    def __init__(self, patience: int = 5, delta: float = 0.0, verbose: bool = False) -> None:
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.best_loss: float | None = None
        self.best_epoch: int | None = None
        self.best_state: dict | None = None
        self.no_improvement_count = 0
        self.stop_training = False

    def check_early_stop(
        self,
        val_loss: float,
        model: torch.nn.Module | None = None,
        epoch: int | None = None,
    ) -> None:
        """Records this epoch's validation loss and snapshots the weights if it improved."""

        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.no_improvement_count = 0

            if model is not None:
                self.best_state = copy.deepcopy(model.state_dict())

        else:
            self.no_improvement_count += 1
            if self.no_improvement_count >= self.patience:
                self.stop_training = True
                if self.verbose:
                    print("Stopping early as no improvement has been observed.")

    def restore_best(self, model: torch.nn.Module) -> torch.nn.Module:
        """Loads the best snapshot back into model."""

        if self.best_state is not None:
            model.load_state_dict(self.best_state)
            if self.verbose:
                print(f"Restored best weights from epoch {self.best_epoch} | val loss: {self.best_loss:.5f}")

        return model
