"""
if __name__ == "__main__":
    preprocessor = PreprocessLoans(cfg.RAW_PATH)
    preprocessor.transform()
"""

import numpy as np
import joblib
import polars as pl
import model.logistic_regression.config as cfg

from typing import Union
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
)
from model.transformers import NumericScaler, NumericNormalizer


class ModelTrainer:
    """Preprocess loans data for LR."""

    def __init__(self, data_input: Union[str, dict]):
        """Initialize"""
        self.data_input = data_input
        self.bundle = {}
        self.df_X = None
        self.df_y = None
        self.df_X_new = None
        self.scaler = None
        self.normalizer = None
        self.ordinal_encoder = None
        self.onehot_encoder = None

        if isinstance(data_input, str) and data_input.endswith(".csv"):
            self.set_training_frames()
        else:
            self.df_X = pl.DataFrame([self.data_input])

        # with open("models/logistic_regression.pkl", "rb") as f:
        #     self.bundle = joblib.load(f)

    def set_training_frames(self):
        df = pl.read_csv(self.data_input)

        for col in cfg.DROP_FEATURES:
            df.drop_in_place(col)

        self.df_X = df.drop(cfg.TARGET)
        self.df_y = df[cfg.TARGET].to_frame().cast(pl.Int8)

    def _train_scaler(self):
        """Fit and transform numeric scaler."""
        self.scaler = NumericScaler(cfg.SCALE_FEATURES)
        self.df_X = self.scaler.fit_transform(self.df_X)

    def _train_normalizer(self):
        """Fit and transform numeric normalizer."""
        self.normalizer = NumericNormalizer(cfg.NORMALIZE_FEATURES)
        self.df_X = self.normalizer.fit_transform(self.df_X)

    def fit_transform(self):
        """Fit and transform training data."""
        self._train_scaler()
        self._train_normalizer()

    def _apply_scaler(self):
        self.df_X_new = self.scaler.transform(self.df_X_new)

    def _apply_normalizer(self):
        self.df_X_new = self.normalizer.transform(self.df_X_new)

    def transform(self, df_X: pl.DataFrame):
        """Transform input payload."""
        self.df_X_new = df_X
        self._apply_scaler()
        self._apply_normalizer()

    def fit_transform_ordinal_encoder(self):
        """Ordinal encode categorical features."""
        feats = cfg.ORDINAL_ENCODE_FEATURES
        df = self.df_X.select(feats)
        self.ordinal_encoder = OrdinalEncoder()
        arr_encoded = self.ordinal_encoder.fit_transform(df)
        self.df_X[feats] = arr_encoded
        self.df_X[feats] = self.df_X[feats].cast(pl.Int8)

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
    preprocessor = ModelTrainer(cfg.RAW_PATH)
    preprocessor.fit_transform()
    print(preprocessor.df_X.select("DTIRatio"))

    # with open("models/logistic_regression.pkl", "wb") as f:
    #     joblib.dump(preprocessor, f)

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
    df_new = pl.DataFrame(loan_data)

    # # to be handled in an orchestator function
    # preprocessor = joblib.load("models/logistic_regression.pkl")
    # print(preprocessor)
    preprocessor.transform(df_new)
    print(preprocessor.df_X_new.select("DTIRatio"))
