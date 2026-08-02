"""Random forest baseline, reproducing Xiao et al. (2017)."""

import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier

XIAO_BEST_PARAMS = {"n_estimators": 100, "criterion": "entropy", "max_depth": 100}

N_RUNS = 5


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 100,
    criterion: str = "entropy",
    max_depth: int = 100,
    random_state: int = 42,
    n_jobs: int = -1,
) -> tuple[RandomForestClassifier, float]:
    """Trains a random forest classifier as per Xiao et al. (2017).

    Returns the fitted model and the train time.
    """

    model = RandomForestClassifier(
        n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, random_state=random_state, n_jobs=n_jobs
    )

    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start

    return model, train_time


def repeated_test_accuracy(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_runs: int = N_RUNS,
    n_jobs: int = -1,
) -> tuple[np.ndarray, np.ndarray]:
    """Fits the random forest once per seed to collect test accuracy.

    Xiao et al. (2017) repeat the algorithm 5 times and report the mean test accuracy.
    XIAO_BEST_PARAMS is passed explicitly so this reproduction cannot drift from the headline run.
    """

    accuracies, train_times = [], []

    for seed in range(n_runs):
        model, train_time = train_random_forest(X_train, y_train, **XIAO_BEST_PARAMS, random_state=seed, n_jobs=n_jobs)
        accuracies.append((model.predict(X_test) == y_test).mean())
        train_times.append(train_time)

    return np.array(accuracies), np.array(train_times)
