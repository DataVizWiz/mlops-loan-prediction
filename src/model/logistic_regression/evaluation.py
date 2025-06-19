import joblib
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

if __name__ == "__main__":
    model_path = "models/logistic_regression.pkl"
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}.")

    data = np.load("data/test/test_data.npz")
    X_test = data["X_test"]
    y_test = data["y_test"]

    y_pred = model.predict(X_test)

    # [TODO] log these to a db
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
