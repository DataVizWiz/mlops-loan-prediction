import joblib
import polars as pl
import backend.control.config as cfg

if __name__ == "__main__":
    model_path = f"{cfg.REGISTRY}/logistic_regression.pkl"
    lr_model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")

    X_test_path = f"{cfg.TEST_DATA_DIR}/X_test.csv"
    y_test_path = f"{cfg.TEST_DATA_DIR}/y_test.csv"

    X_test = pl.read_csv(X_test_path).to_numpy()
    y_test = pl.read_csv(y_test_path).to_numpy()

    pred = lr_model.predict(X_test)
