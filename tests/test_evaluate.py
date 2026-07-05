import numpy as np

from fmnist.data import CLASS_NAMES
from fmnist.evaluate import evaluate_split


def generate_data():
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 10, size=50)
    y_pred = rng.integers(0, 10, size=50)

    return y_true, y_pred


def test_evaluate_split(tmp_path):
    y_true, y_pred = generate_data()
    cm, mdf = evaluate_split(y_true, y_pred, "rf", "train", output_dir=tmp_path)

    assert (tmp_path / "figures" / "cm_rf_train.png").exists()
    assert (tmp_path / "tables" / "metrics_rf_train.csv").exists()

    assert cm.shape == (10, 10)

    assert all(name in mdf.index for name in CLASS_NAMES)
