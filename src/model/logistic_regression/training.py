import polars as pl
from model.logistic_regression.preprocessing import ModelTrainer


if __name__ == "__main__":
    path = "data/raw/loans.csv"
    df = pl.read_csv(path)

    trainer = ModelTrainer()
    trainer.fit_transform(df)
