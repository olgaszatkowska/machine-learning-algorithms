import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from numpy.typing import NDArray
from imblearn.over_sampling import SMOTENC

from heart_disease_dataset.dataset import get_dataset


def get_transformed_data(sample_data: bool = False) -> tuple[NDArray, NDArray]:
    X, y = get_dataset()
    categorical_features = _get_categorical_features()
    categorical_features_missing_values = _get_categorical_features_missing_values()
    all_categorical = categorical_features + categorical_features_missing_values
    numerical_features = get_numerical_features()

    categorical_pipeline = Pipeline([("passthrough", "passthrough")])
    categorical_missing_values_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="most_frequent"))]
    )
    numerical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("num", numerical_pipeline, numerical_features),
            ("cat", categorical_pipeline, categorical_features),
            (
                "cat_missing",
                categorical_missing_values_pipeline,
                categorical_features_missing_values,
            ),
        ]
    )

    X_transformed = preprocessor.fit_transform(X)
    
    if not sample_data:
        return X_transformed, y.to_numpy()
    
    X_df = pd.DataFrame(
        X_transformed, columns=numerical_features + all_categorical
    )
    
    smote_nc = SMOTENC(
        categorical_features=all_categorical,
        random_state=42,
    )
    X_resampled, y_resampled = smote_nc.fit_resample(X_df, y)


    return X_resampled.to_numpy(), y_resampled.to_numpy()


def _get_categorical_features() -> list[str]:
    return ["sex", "cp", "fbs", "restecg", "exang", "slope"]


def _get_categorical_features_missing_values() -> list[str]:
    return ["thal"]


def get_categorical_features() -> list[str]:
    return _get_categorical_features() + _get_categorical_features_missing_values()


def get_numerical_features() -> list[str]:
    return ["age", "ca", "chol", "oldpeak", "thalach", "trestbps"]
