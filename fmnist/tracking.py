"""MLflow tracking helpers.

MLflow 3 removed the filesystem tracking backend, so runs go to a local
SQLite database. Both the database and the artifact root are anchored to
the project directory rather than the CWD, so results land in the same
place whether a run starts from the repo root or from notebooks/.
"""

import os
import platform
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import mlflow

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
ARTIFACT_ROOT = PROJECT_ROOT / "mlartifacts"

EXPERIMENT_NAME = "fashion-mnist"


def _experiment_id(name: str) -> str:
    """Experiment id created with a pinned artifact root if missing."""

    mlflow.set_tracking_uri(TRACKING_URI)

    experiment = mlflow.get_experiment_by_name(name)
    if experiment is not None:
        return experiment.experiment_id

    return mlflow.create_experiment(name, artifact_location=ARTIFACT_ROOT.as_uri())


def log_run(
    run_name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    artifacts: Iterable[Path] | None = None,
    experiment: str = EXPERIMENT_NAME,
) -> str:
    """Logs one completed experiment to MLflow and returns its run id."""

    experiment_id = _experiment_id(experiment)

    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
        mlflow.set_tags({"cpu_count": os.cpu_count(), "platform": platform.platform()})
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        for path in artifacts or []:
            mlflow.log_artifact(str(path))

    return run.info.run_id
