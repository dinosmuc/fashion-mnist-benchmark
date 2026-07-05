from sklearn.ensemble import RandomForestClassifier


def train_random_forest(
    X_train,
    y_train,
    n_estimators=100,
    max_depth=None,
    max_features="sqrt",
    min_samples_leaf=1,
    min_samples_split=2,
    random_state=42,
):
    """Trains a random forest classifier on the training data and returnes the fitted model"""

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


if __name__ == "__main__":
    pass
