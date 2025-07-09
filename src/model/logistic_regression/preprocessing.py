"""
if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.fit_transform(df)
"""

import numpy as np
import joblib
import polars as pl
import model.logistic_regression.config as cfg

from model.transformers import Scaler, Encoder
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    OneHotEncoder,
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


class ModelTrainer:
    def __init__(self):
        """Model trainer."""
        self.bundle = {}
        self.df = None
        self.df_X = None
        self.df_y = None

        self.X_train = None
        self.y_train = None

        self.standard_scaler = None
        self.minmax_scaler = None
        self.onehot_encoder = None
        self.model = LogisticRegression()
        self.pkl = "models/logistic_regression.pkl"

    def fit_transform(self, df: pl.DataFrame):
        """Fit and transform training data."""
        self.df = df
        self._set_X_and_y_frames()
        self._train_standard_scaler()
        self._train_minmax_scaler()
        self._train_onehot_encoder()
        self._apply_binary_encoder()
        self._train_test_split()
        self._train_model()
        self._save_model()

    def _save_model(self):
        """Save model to pkl file."""
        with open(self.pkl, "wb") as f:
            joblib.dump(self.bundle, f)

    def _train_test_split(self):
        """Train and test split."""
        X = self.df_X.to_numpy()
        y = self.df_y.to_numpy().ravel()

        self.X_train, X_test, self.y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        path = "data/test/test_data.npz"
        np.savez(path, X_test=X_test, y_test=y_test)
        print(f"Test data saved to {path}.")

    def _set_X_and_y_frames(self):
        """Set X and y dfs."""
        self.df_y = self.df["Default"].to_frame()
        self.df_X = self.df

        for col in ["LoanID", "Default"]:
            self.df_X.drop_in_place(col)

    def _train_standard_scaler(self):
        """Fit and transform standard scaler."""
        features = [col for col in cfg.NUMERIC_FEATURES if col != "DTIRatio"]
        self.standard_scaler = Scaler(features, StandardScaler())
        self.df_X = self.standard_scaler.fit_transform(self.df_X)
        self.bundle["standard_scaler"] = self.standard_scaler

    def _train_minmax_scaler(self):
        """Fit and transform mix max scaler."""
        self.minmax_scaler = Scaler(["DTIRatio"], MinMaxScaler(feature_range=(0, 1)))
        self.df_X = self.minmax_scaler.fit_transform(self.df_X)
        self.bundle["minmax_scaler"] = self.minmax_scaler

    def _train_onehot_encoder(self):
        """Fit and transform onehot encoder."""
        features = cfg.CATEGORICAL_FEATURES[1:5]
        encoder = OneHotEncoder(sparse_output=False, dtype=int)
        encoder.set_output(transform="polars")
        self.onehot_encoder = Encoder(features, encoder)
        self.df_X = self.onehot_encoder.fit_transform(self.df_X)
        self.bundle["onehot_encoder"] = self.onehot_encoder

    def _apply_binary_encoder(self):
        """Encode binary features as integers"""
        bin_map = {"Yes": 1, "No": 0}

        for feature in cfg.CATEGORICAL_FEATURES[-3:]:
            self.df_X = self.df_X.with_columns(
                pl.col(feature).replace(bin_map).alias(feature)
            )

    def _train_model(self):
        """Train LR model."""
        self.model.fit(self.X_train, self.y_train)
        self.bundle["model"] = self.model


class ModelPredictor:
    """Model predictor."""

    def __init__(self):
        """Initialize"""
        self.df = None

        with open("models/logistic_regression.pkl", "rb") as f:
            self.bundle = joblib.load(f)

    def transform(self, payload: dict):
        """Transform new data."""
        self.df = pl.DataFrame(payload)
        self._apply_transformer(self.bundle["standard_scaler"])
        self._apply_transformer(self.bundle["minmax_scaler"])
        self._apply_transformer(self.bundle["onehot_encoder"])
        self._apply_binary_encoder()

    def _apply_transformer(self, transformer: object):
        """Apply transformer on new data."""
        self.df = transformer.transform(self.df)

    def _apply_binary_encoder(self):
        """Encode binary features as integers"""
        bin_map = {"Yes": 1, "No": 0}

        for feature in cfg.CATEGORICAL_FEATURES[-3:]:
            self.df = self.df.with_columns(
                pl.col(feature).replace(bin_map).alias(feature)
            )
