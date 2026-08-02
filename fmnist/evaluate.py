from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

from .data import CLASS_NAMES

OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "outputs"


def _cm_path(model_name: str, split_name: str, output_dir: Path = OUTPUT_ROOT) -> Path:
    """Path of the confusion matrix figure for one model on one split."""
    return output_dir / "figures" / f"cm_{model_name}_{split_name}.png"


def _metrics_path(model_name: str, split_name: str, output_dir: Path = OUTPUT_ROOT) -> Path:
    """Path of the per-class metrics table for one model on one split."""
    return output_dir / "tables" / f"metrics_{model_name}_{split_name}.csv"


def evaluate_split(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    split_name: str,
    output_dir: Path = OUTPUT_ROOT,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Writes the confusion matrix (PNG) and per-class precision/recall (CSV) for one
    model on one split. Returns (cm, metrics_df)."""

    fig_dir = output_dir / "figures"
    tab_dir = output_dir / "tables"

    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=range(10))

    fig, ax = plt.subplots(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=False, xticks_rotation=45)

    ax.set_title(f"{model_name} - {split_name} set")
    fig.savefig(_cm_path(model_name, split_name, output_dir), dpi=150, bbox_inches="tight")
    plt.close(fig)

    report = classification_report(
        y_true, y_pred, labels=range(10), target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )

    metrics_df = pd.DataFrame(report).T
    metrics_df.to_csv(_metrics_path(model_name, split_name, output_dir))

    return cm, metrics_df


def evaluation_artifacts(
    model_name: str,
    splits: tuple[str, ...] = ("train", "test"),
    output_dir: Path = OUTPUT_ROOT,
) -> list[Path]:
    """Paths of the files evaluate_split wrote for one model."""
    return [p(model_name, s, output_dir) for s in splits for p in (_cm_path, _metrics_path)]
