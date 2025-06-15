import os
import joblib
import polars as pl
import backend.control.config as cfg

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
)
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE


class TrainLogisticRegression:
    """Train a LogisticRegression model"""

    def __init__(self):
        """Initialize"""
        df = pl.read_csv(cfg.RAW_PATH)
        df.drop_in_place(cfg.LOAN_ID_COL)
        self.df_X = df.drop(cfg.TARGET_COL)
        self.df_y = df[cfg.TARGET_COL].to_frame().cast(pl.Int8)
        self.model = LogisticRegression()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def apply_scaling(self):
        """Scale numeric features"""
        scale_feats = [x for x in cfg.NUM_FEATS if x != "DTIRatio"]

        df_num = self.df_X[scale_feats]
        num_arr = df_num.to_numpy()

        scaler = StandardScaler()
        scaled_arr = scaler.fit_transform(num_arr)
        df_scaled = pl.DataFrame(scaled_arr, scale_feats).cast(pl.Float32)
        self.df_X[df_scaled.columns] = df_scaled

    def apply_normalization(self):
        """Normalize DTIRatio"""
        norm_feats = [x for x in cfg.NUM_FEATS if x == "DTIRatio"]

        df_dtir = self.df_X[norm_feats]
        dtir_arr = df_dtir.to_numpy()

        normalizer = MinMaxScaler(feature_range=(0, 1))
        normed_arr = normalizer.fit_transform(dtir_arr)
        df_norm = pl.DataFrame(normed_arr, norm_feats).cast(pl.Float32)
        self.df_X[df_norm.columns] = df_norm

    def apply_ordinal_encoding(self):
        """Ordinal encode LoanTerm"""
        encoder = OrdinalEncoder()
        encoded_arr = encoder.fit_transform(self.df_X[["LoanTerm"]])
        self.df_X[["LoanTerm"]] = encoded_arr
        self.df_X[["LoanTerm"]] = self.df_X[["LoanTerm"]].cast(pl.Int8)

    def apply_one_hot_encoding(self):
        """One-hot encode categorical features"""
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=int)
        encoder.set_output(transform="polars")
        df_cat_enc = encoder.fit_transform(self.df_X[cfg.CAT_FEATS]).cast(pl.Int8)
        self.df_X[df_cat_enc.columns] = df_cat_enc
        self.df_X = self.df_X.drop(cfg.CAT_FEATS)

    def apply_binary_encoding(self):
        """Encode binary features as integers"""
        bin_feats = ["HasMortgage", "HasDependents", "HasCoSigner"]
        bin_map = {"Yes": 1, "No": 0}

        for feat in bin_feats:
            self.df_X = self.df_X.with_columns(
                pl.col(feat).replace(bin_map).alias(feat).cast(pl.Int8)
            )

    def upload_preprocessed_features(self):
        """Upload engineering dataset to S3 bucket"""
        # [TODO] Upload to a S3
        t_dir = cfg.TRANSFORMED_DIR
        if not os.path.exists(t_dir):
            os.mkdir(t_dir)

        df_eng = pl.concat([self.df_X, self.df_y], how="horizontal")
        df_eng.write_csv(f"{t_dir}/feats_engineered.csv")

    def train_model(self):
        """Balance training data using SMOTE"""
        X = self.df_X.to_numpy()
        y = self.df_y.to_numpy()

        X_train, self.X_test, y_train, self.y_test = train_test_split(
            X, y, train_size=0.8, random_state=42
        )
        smote = SMOTE(random_state=42)
        self.X_train, self.y_train = smote.fit_resample(X_train, y_train)
        self.model.fit(self.X_train, self.y_train)

    def save_test_data(self):
        """Save test data to CSV"""
        if not os.path.exists(cfg.TEST_DATA_DIR):
            os.mkdir(cfg.TEST_DATA_DIR)

        df_X = pl.DataFrame(self.X_test)
        df_y = pl.DataFrame(self.y_test)
        df_X.write_csv(f"{cfg.TEST_DATA_DIR}/X_test.csv")
        df_y.write_csv(f"{cfg.TEST_DATA_DIR}/y_test.csv")
        print(f"X_test, y_test saved to {cfg.TEST_DATA_DIR}")

    def save_model(self):
        """Train and save LR model"""
        pkl_path = f"{cfg.REGISTRY}/logistic_regression.pkl"
        joblib.dump(self.model, pkl_path)
        print(f"Model has been saved to {pkl_path}")


if __name__ == "__main__":
    train = TrainLogisticRegression()
    train.apply_scaling()
    train.apply_normalization()
    train.apply_ordinal_encoding()
    train.apply_one_hot_encoding()
    train.apply_binary_encoding()
    train.upload_preprocessed_features()
    train.train_model()
    train.save_test_data()
    train.save_model()
