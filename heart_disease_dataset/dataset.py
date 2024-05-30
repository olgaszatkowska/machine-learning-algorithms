from pandas.core.frame import DataFrame


def get_dataset() -> tuple[DataFrame, DataFrame]:
    from ucimlrepo import fetch_ucirepo

    heart_disease = fetch_ucirepo(id=45)

    X = heart_disease.data.features
    y = heart_disease.data.targets

    return X, y
