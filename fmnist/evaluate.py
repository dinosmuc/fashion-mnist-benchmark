from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

from fmnist.data import CLASS_NAMES

OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "outputs"


def evaluate_split(y_true, y_pred, model_name, split_name, output_dir=OUTPUT_ROOT):
    """Confusion matrix (PNG) + per-class precision/recall (CSV) for one
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
    fig.savefig(fig_dir / f"cm_{model_name}_{split_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    report = classification_report(
        y_true, y_pred, labels=range(10), target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )

    metrics_df = pd.DataFrame(report).T
    metrics_df.to_csv(tab_dir / f"metrics_{model_name}_{split_name}.csv")

    return cm, metrics_df
