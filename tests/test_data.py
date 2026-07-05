from fmnist.data import get_numpy_data


def test_get_numpy_data_shapes(tmp_path):
    """Testing are all shapes correct, flatttering and is there any change in the API"""

    X_train, y_train, X_test, y_test = get_numpy_data()

    # Testing shapes
    assert X_train.shape == (60000, 784)
    assert X_test.shape == (10000, 784)
    assert y_train.shape == (60000,)
    assert y_test.shape == (10000,)

    # Testing flattering
    assert X_train.shape[1] == 28 * 28
