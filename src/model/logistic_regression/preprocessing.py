"""
if __name__ == "__main__":
    preprocessor = PreprocessLoans(cfg.RAW_PATH)
    preprocessor.run()
"""

import polars as pl
import model.logistic_regression.config as cfg

from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
)
from sklearn.linear_model import LogisticRegression


class PreprocessLoans:
    """Preprocess loans data for LR."""

    def __init__(self, path: str):
        """Initialize"""
        df = pl.read_csv(path)
        for col in cfg.DROP_FEATURES:
            df.drop_in_place(col)
        self.df_X = df.drop(cfg.TARGET)
        self.df_y = df[cfg.TARGET].to_frame().cast(pl.Int8)
        self.model = LogisticRegression()

    def apply_scaling(self):
        """Scale numeric features."""
        df = self.df_X[cfg.SCALE_FEATURES]
        arr = df.to_numpy()

        scaler = StandardScaler()
        scaled_arr = scaler.fit_transform(arr)
        df_scaled = pl.DataFrame(scaled_arr, cfg.SCALE_FEATURES).cast(pl.Float32)
        self.df_X[df_scaled.columns] = df_scaled

    def apply_normalization(self):
        """Normalize numeric features."""
        df = self.df_X[cfg.NORMALIZE_FEATURES]
        arr = df.to_numpy()

        normalizer = MinMaxScaler(feature_range=(0, 1))
        normed_arr = normalizer.fit_transform(arr)
        df_norm = pl.DataFrame(normed_arr, cfg.NORMALIZE_FEATURES).cast(pl.Float32)
        self.df_X[df_norm.columns] = df_norm

    def apply_ordinal_encoding(self):
        """Ordinal encode categorical features."""
        feats = cfg.ORDINAL_ENCODE_FEATURES
        df = self.df_X[feats]

        encoder = OrdinalEncoder()
        encoded_arr = encoder.fit_transform(df)
        self.df_X[feats] = encoded_arr
        self.df_X[feats] = self.df_X[feats].cast(pl.Int8)

    def apply_one_hot_encoding(self):
        """One-hot encode categorical features."""
        feats = cfg.ONEHOT_ENCODE_FEATURES
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=int)
        encoder.set_output(transform="polars")
        df = encoder.fit_transform(self.df_X[feats]).cast(pl.Int8)
        self.df_X[df.columns] = df
        self.df_X = self.df_X.drop(feats)

    def apply_binary_encoding(self):
        """Encode binary features as integers"""
        bin_map = {"Yes": 1, "No": 0}

        for feat in cfg.BINARY_ENCODE_FEATURES:
            self.df_X = self.df_X.with_columns(
                pl.col(feat).replace(bin_map).alias(feat).cast(pl.Int8)
            )

    def transform(self):
        """Orchestrate preprocessing."""
        self.apply_scaling()
        self.apply_normalization()
        self.apply_ordinal_encoding()
        self.apply_one_hot_encoding()
        self.apply_binary_encoding()
