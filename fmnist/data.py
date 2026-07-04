from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms


def get_dataset(root="../data"):
    """Retruns train and test data set as tensors - for CNN"""

    # Creating an instance of ToTensor class
    transform = transforms.ToTensor()

    # Downloading the dataset and storing it as tensors in data folder
    train_dataset = datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)

    return train_dataset, test_dataset


def get_numpy_data(root="../data", val_size=0.2, random_state=42):
    """Returns (X_train, y_train, X_val, y_val, X_test, y_test) as flat Numpy arrays - for random forest"""

    # Tuple unpacking to get the dataset as tensors
    train_dataset, test_dataset = get_dataset(root)

    # Tranformation of tensors to Numpy arrays and spliting
    X = train_dataset.data.numpy().reshape(len(train_dataset), -1)
    y = train_dataset.targets.numpy()

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size, random_state=random_state)

    X_test = test_dataset.data.numpy().reshape(len(test_dataset), -1)
    y_test = test_dataset.targets.numpy()

    return X_train, y_train, X_val, y_val, X_test, y_test


if __name__ == "__main__":
    pass
