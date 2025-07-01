"""
if __name__ == "__main__":
    Scaler = PreprocessLoans(cfg.RAW_PATH)
    Scaler.transform()
"""

import numpy as np
import joblib
import polars as pl
import model.logistic_regression.config as cfg

from model.transformers import Scaler, Encoder
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    OrdinalEncoder,
    OneHotEncoder,
)


class ModelTrainer:
    def __init__(self):
        """Model trainer."""
        self.bundle = {}
        self.df_X = None
        self.df_y = None

        self.standard_scaler = None
        self.minmax_scaler = None
        self.onehot_encoder = None

    def fit_transform(self, df_X: pl.DataFrame):
        """Fit and transform training data."""
        self.df_X = df_X
        self._set_X_and_y_frames()
        self._train_standard_scaler()
        self._train_minmax_scaler()
        self._train_onehot_encoder()
        self.save_model()

    def save_model(self):
        """Save model to pkl file."""
        with open("models/logistic_regression.pkl", "wb") as f:
            joblib.dump(self.bundle, f)

    def _set_X_and_y_frames(self):
        """Set X and y dfs."""
        self.df_y = self.df_X["Default"].to_frame()

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
        self.onehot_encoder.fit_transform(self.df_X)
        self.bundle["onehot_encoder"] = self.onehot_encoder


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


if __name__ == "__main__":
    path = "data/raw/loans.csv"
    df = pl.read_csv(path)

    trainer = ModelTrainer()
    trainer.fit_transform(df)

    loan_data = {
        "Age": 56,
        "Income": 85994,
        "LoanAmount": 50587,
        "CreditScore": 520,
        "MonthsEmployed": 80,
        "NumCreditLines": 4,
        "InterestRate": 15.23,
        "LoanTerm": 36,
        "DTIRatio": 0.98773,
        "Education": "Bachelor's",
        "EmploymentType": "Full-time",
        "MaritalStatus": "Divorced",
        "HasMortgage": "Yes",
        "HasDependents": "Yes",
        "LoanPurpose": "Other",
        "HasCoSigner": "Yes",
        "Default": 0,
    }

    predictor = ModelPredictor()
    predictor.transform(loan_data)
    print(predictor.df)
