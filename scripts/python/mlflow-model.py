import mlflow
import mlflow.sklearn
import pandas as pd

from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score


def preprocessing():
    URL = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    df = pd.read_csv(URL, sep=";")

    TARGET = "quality"
    X = df.drop(columns=TARGET)
    y = df[TARGET]

    return train_test_split(X, y, random_state=6, test_size=0.2)


def eval_function(actual, pred):
    rmse = root_mean_squared_error(actual, pred)
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def main(alpha, l1_ratio):
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("WineQuality-ElasticNet")
    print("Tracking URI:", mlflow.get_tracking_uri())

    X_train, X_test, y_train, y_test = preprocessing()

    with mlflow.start_run():
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)

        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=6)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse, mae, r2 = eval_function(y_test, y_pred)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2_score", r2)
        mlflow.sklearn.log_model(model, "ElasticNet-Trained")


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--alpha", "-a", type=float, default=0.2)
    args.add_argument("--l1_ratio", "-l1", type=float, default=0.3)
    parsed_args = args.parse_args()

    main(parsed_args.alpha, parsed_args.l1_ratio)
