"""
if __name__ == "__main__":
    preprocessor = PreprocessLoans(cfg.RAW_PATH)
    preprocessor.transform()
"""

import numpy as np
import joblib
import polars as pl
import model.logistic_regression.config as cfg

from model.transformers import Preprocessor
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    OrdinalEncoder,
    OneHotEncoder,
)


class ModelTrainer:
    """Preprocess loans data for LR."""

    def __init__(self):
        """Initialize"""
        self.bundle = {}
        self.df_X = None
        self.df_y = None

        self.standard_scaler = None
        self.mixmax_scaler = None
        self.ordinal_encoder = None
        self.onehot_encoder = None

        self.payload = None

    def _set_training_frames(self):
        self.df_y = self.df_X["Default"].to_frame()

        for col in ["LoanID", "Default"]:
            self.df_X.drop_in_place(col)

    def _train_standard_scaler(self):
        """Fit and transform standard scaler."""
        features = [col for col in cfg.NUMERIC_FEATURES if col != "DTIRatio"]
        self.standard_scaler = Preprocessor(features, StandardScaler())
        self.df_X = self.standard_scaler.fit_transform(self.df_X)

    def _train_mixmax_scaler(self):
        """Fit and transform mix max scaler."""
        self.mixmax_scaler = Preprocessor(
            ["DTIRatio"], MinMaxScaler(feature_range=(0, 1))
        )
        self.df_X = self.mixmax_scaler.fit_transform(self.df_X)

    def _train_ordinal_encoder(self):
        """Fit and transform ordinal encoder."""
        self.ordinal_encoder = Preprocessor(["LoanTerm"], OrdinalEncoder())
        self.df_X = self.ordinal_encoder.fit_transform(self.df_X)

    def fit_transform(self, df_X: pl.DataFrame):
        """Fit and transform training data."""
        self.df_X = df_X
        self._set_training_frames()
        self._train_standard_scaler()
        self._train_mixmax_scaler()
        self._train_ordinal_encoder()

    def _apply_standard_scaler(self):
        self.payload = self.standard_scaler.transform(self.payload)

    def _apply_minmax_scaler(self):
        self.payload = self.mixmax_scaler.transform(self.payload)

    def transform(self, payload: dict):
        self.payload = payload
        self._apply_standard_scaler()
        self._apply_minmax_scaler()

    def fit_transform_onehot_encoder(self):
        """One-hot encode categorical features."""
        feats = cfg.ONEHOT_ENCODE_FEATURES
        self.onehot_encoder = OneHotEncoder(
            handle_unknown="ignore", sparse_output=False, dtype=int
        )
        self.onehot_encoder.set_output(transform="polars")
        df = self.onehot_encoder.fit_transform(self.df_X[feats]).cast(pl.Int8)
        self.df_X = self.df_X.with_columns(df)
        self.df_X = self.df_X.drop(feats)

    def transform_binary_encoder(self):
        """Encode binary features as integers"""
        bin_map = {"Yes": 1, "No": 0}

        for feat in cfg.BINARY_ENCODE_FEATURES:
            self.df_X = self.df_X.with_columns(
                pl.col(feat).replace(bin_map).alias(feat).cast(pl.Int8)
            )


if __name__ == "__main__":
    path = "data/raw/loans.csv"
    df = pl.read_csv(path)

    trainer = ModelTrainer()
    trainer.fit_transform(df)
    print(trainer.df_X.select("LoanTerm"))

    # loan_data = {
    #     "Age": 56,
    #     "Income": 85994,
    #     "LoanAmount": 50587,
    #     "CreditScore": 520,
    #     "MonthsEmployed": 80,
    #     "NumCreditLines": 4,
    #     "InterestRate": 15.23,
    #     "LoanTerm": 36,
    #     "DTIRatio": 0.98773,
    #     "Education": "Bachelor's",
    #     "EmploymentType": "Full-time",
    #     "MaritalStatus": "Divorced",
    #     "HasMortgage": "Yes",
    #     "HasDependents": "Yes",
    #     "LoanPurpose": "Other",
    #     "HasCoSigner": "Yes",
    #     "Default": 0,
    # }

    # trainer.transform(loan_data)
    # print(trainer.payload)
