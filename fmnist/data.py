from pathlib import Path

from torchvision import datasets, transforms

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"


def get_dataset(root=DATA_ROOT):
    """Retruns train and test data set as tensors - for CNN"""

    # Creating an instance of ToTensor class
    transform = transforms.ToTensor()

    # Downloading the dataset and storing it as tensors in data folder
    train_dataset = datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)

    return train_dataset, test_dataset


def get_numpy_data(root=DATA_ROOT):
    """Returns (X_train, y_train, X_test, y_test) as flat Numpy arrays - for random forest"""

    # Tuple unpacking to get the dataset as tensors
    train_dataset, test_dataset = get_dataset(root)

    # Tranformation of tensors to Numpy arrays and spliting
    X_train = train_dataset.data.numpy().reshape(len(train_dataset), -1)
    y_train = train_dataset.targets.numpy()

    X_test = test_dataset.data.numpy().reshape(len(test_dataset), -1)
    y_test = test_dataset.targets.numpy()

    return X_train, y_train, X_test, y_test
