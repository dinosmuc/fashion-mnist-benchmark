import time

from sklearn.ensemble import RandomForestClassifier


def train_random_forest(
    X_train, y_train, n_estimators=100, criterion="entropy", max_depth=100, random_state=42, n_jobs=-1
):
    """Trains a random forest classifier on the training data and returns the fitted model and train time"""

    model = RandomForestClassifier(
        n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, random_state=random_state, n_jobs=n_jobs
    )

    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start

    return model, train_time
