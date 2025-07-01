import polars as pl
from typing import List


class Scaler:
    """Apply scaling on features."""

    def __init__(self, features: List[str], scaler: object):
        self.features = features
        self.scaler = scaler

    def fit_transform(self, df_X: pl.DataFrame) -> pl.DataFrame:
        """Fit scaler and transform df."""
        fil_df = df_X.select(self.features)
        scaled_arr = self.scaler.fit_transform(fil_df)
        scaled_df = pl.DataFrame(scaled_arr, self.features)
        return df_X.with_columns(scaled_df)

    def transform(self, payload: dict) -> pl.DataFrame:
        """Scale new inputs."""
        df = pl.DataFrame(payload)
        fil_df = df.select(self.features)
        scaled_arr = self.scaler.transform(fil_df)
        scaled_df = pl.DataFrame(scaled_arr, self.features)
        return df.with_columns(scaled_df)


class Encoder:
    """Apply encoding on features."""

    def __init__(self, features: List[str], encoder: object):
        self.features = features
        self.encoder = encoder

    def fit_transform(self, df_X: pl.DataFrame) -> pl.DataFrame:
        """Fit scaler and transform df."""
        fil_df = df_X.select(self.features)
        encoded_df = self.encoder.fit_transform(fil_df)
        df_X = df_X.drop(self.features)
        return pl.concat([df_X, encoded_df], how="horizontal")

    def transform(self, payload: dict) -> pl.DataFrame:
        """Scale new inputs."""
        df = pl.DataFrame(payload)
        fil_df = df.select(self.features)
        encoded_df = self.encoder.transform(fil_df)
        df = df.drop(self.features)
        return pl.concat([df, encoded_df], how="horizontal")
