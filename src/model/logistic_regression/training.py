import joblib
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from model.logistic_regression.config import RAW_PATH
from model.logistic_regression.preprocessing import PreprocessLoans


if __name__ == "__main__":
    preprocessor = PreprocessLoans(RAW_PATH)
    preprocessor.transform()

    X = preprocessor.df_X.to_numpy()
    y = preprocessor.df_y.to_numpy().ravel()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    np.savez("data/test/test_data.npz", X_test=X_test, y_test=y_test)

    model = LogisticRegression()
    model.fit(X_train, y_train)
    joblib.dump(model, "models/logistic_regression.pkl")
