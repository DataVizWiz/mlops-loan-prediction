import model.logistic_regression.config as cfg

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    OrdinalEncoder,
    OneHotEncoder,
)
from sklearn.base import BaseEstimator, TransformerMixin


class BinaryEncoder(BaseEstimator, TransformerMixin):
    pass


preprocessor = ColumnTransformer(
    transformers=[
        ("scaler", StandardScaler(), cfg.SCALE_FEATURES),
        ("normalizer", MinMaxScaler(feature_range=(0, 1)), cfg.NORMALIZE_FEATURES),
        ("ordinal_encoder", OrdinalEncoder(), cfg.ORDINAL_ENCODE_FEATURES),
        (
            "onehot_encoder",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=int),
            cfg.ONEHOT_ENCODE_FEATURES,
        ),
    ],
    remainder="passthrough",
)

print(preprocessor)
