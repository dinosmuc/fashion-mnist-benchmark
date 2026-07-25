"""MLflow tracking helpers"""

import mlflow

EXPERIMENT_NAME = "fashion-mnist"


def log_run(run_name: str, params: dict, metrics: dict, artifacts=None, experiment=EXPERIMENT_NAME):
    """Log one completed experiment to MLflow"""

    mlflow.set_experiment(experiment)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        for path in artifacts or []:
            mlflow.log_artifact(str(path))

    return run.info.run_id
