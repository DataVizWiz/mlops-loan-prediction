import numpy as np
import polars as pl

from typing import List


class Scaler:
    """Apply scaling on numeric features."""

    def __init__(self, features: List[str], scaler: object):
        self.features = features
        self.scaler = scaler

    def fit_transform(self, df_X: pl.DataFrame) -> pl.DataFrame:
        """Fit scaler and transform df."""
        numeric_df = df_X.select(self.features)
        scaled_arr = self.scaler.fit_transform(numeric_df)
        scaled_df = pl.DataFrame(scaled_arr, self.features)
        return df_X.with_columns(scaled_df)

    def transform(self, payload: dict) -> dict:
        """Scale new inputs."""
        arr = np.array([[payload[name] for name in self.features]])
        scaled_arr = self.scaler.transform(arr)
        scaled_dict = dict(zip(self.features, scaled_arr[0]))
        return {**payload, **scaled_dict}
