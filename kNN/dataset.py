from pandas.core.frame import DataFrame
from sklearn import datasets


def get_dataset() -> tuple[DataFrame, DataFrame]:
    iris_data = datasets.load_iris()
    X = iris_data["data"]
    y = iris_data["target"]

    return X, y
