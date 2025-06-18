import pandas as pd
import numpy as np
import model.logistic_regression.config as cfg

from typing import Union, List
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    OrdinalEncoder,
    OneHotEncoder,
)
from sklearn.base import BaseEstimator, TransformerMixin


class DropFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, features: List[str]):
        self.features = features

    def fit(self):
        return self

    def transform(self, X: pd.DataFrame):
        X = X.drop(columns=self.features)
        return X


class BinaryEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, mapping={"Yes": 1, "No": 0}):
        self.mapping = mapping

    def fit(self):
        return self

    def transform(self, X: Union[pd.Series, np.ndarray]):
        if isinstance(X, pd.Series):
            X = X.to_frame()
        elif isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        return X.applymap(lambda val: self.mapping.get(val, np.nan))


preprocessor = ColumnTransformer(
    transformers=[
        ("dropper", DropFeatures(cfg.DROP_FEATURES)),
        ("scaler", StandardScaler(), cfg.SCALE_FEATURES),
        ("normalizer", MinMaxScaler(feature_range=(0, 1)), cfg.NORMALIZE_FEATURES),
        ("ordinal_encoder", OrdinalEncoder(), cfg.ORDINAL_ENCODE_FEATURES),
        (
            "onehot_encoder",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=int),
            cfg.ONEHOT_ENCODE_FEATURES,
        ),
        ("binary_encoder", BinaryEncoder(), cfg.BINARY_ENCODE_FEATURES),
    ],
    remainder="passthrough",
)
