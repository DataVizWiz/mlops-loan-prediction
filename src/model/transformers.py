import numpy as np
import polars as pl

from typing import List
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class NumericScaler:
    """Numeric scaler."""

    def __init__(self, features: List[str]):
        self.features = features
        self.scaler = StandardScaler()

    def _select_features_as_array(self, df_X: pl.DataFrame) -> np.ndarray:
        """Covert df to numpy array."""
        return df_X.select(self.features).to_numpy()

    def _replace_features_with_scaled(
        self, df: pl.DataFrame, scaled: np.ndarray
    ) -> pl.DataFrame:
        """Update df_X with scaled array values."""
        scaled_df = pl.DataFrame(scaled, self.features).cast(pl.Float32)
        return df.with_columns(scaled_df)

    def fit_transform(self, df_X: pl.DataFrame) -> pl.DataFrame:
        """Fit scaler and transform df."""
        arr = self._select_features_as_array(df_X)
        scaled_arr = self.scaler.fit_transform(arr)
        return self._replace_features_with_scaled(df_X, scaled_arr)

    def transform(self, df_X: pl.DataFrame) -> pl.DataFrame:
        """Scale new df_X values."""
        arr = self._select_features_as_array(df_X)
        scaled_arr = self.scaler.transform(arr)
        return self._replace_features_with_scaled(df_X, scaled_arr)


class NumericNormalizer:
    """Numeric scaler."""

    def __init__(self, features: List[str]):
        self.features = features
        self.normalizer = MinMaxScaler(feature_range=(0, 1))

    def _select_features_as_array(self, df_X: pl.DataFrame) -> np.ndarray:
        """Covert df to numpy array."""
        return df_X.select(self.features).to_numpy()

    def _replace_features_with_normalized(
        self, df: pl.DataFrame, normalized: np.ndarray
    ) -> pl.DataFrame:
        """Update df_X with scaled array values."""
        normalized_df = pl.DataFrame(normalized, self.features).cast(pl.Float32)
        return df.with_columns(normalized_df)

    def fit_transform(self, df_X: pl.DataFrame) -> pl.DataFrame:
        """Fit scaler and transform df."""
        arr = self._select_features_as_array(df_X)
        normalized_arr = self.normalizer.fit_transform(arr)
        return self._replace_features_with_normalized(df_X, normalized_arr)

    def transform(self, df_X: pl.DataFrame) -> pl.DataFrame:
        """Scale new df_X values."""
        arr = self._select_features_as_array(df_X)
        scaled_arr = self.normalizer.fit_transform(arr)
        return self._replace_features_with_normalized(df_X, scaled_arr)
