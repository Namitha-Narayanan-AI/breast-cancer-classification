import pandas as pd
from sklearn.datasets import load_breast_cancer


def load_dataset():
    """
    Load the Breast Cancer Wisconsin Diagnostic dataset.

    Returns:
        X: Feature dataframe
        y: Target labels
        target_names: Class names
    """
    dataset = load_breast_cancer()

    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    y = pd.Series(dataset.target, name="target")

    return X, y, dataset.target_names