from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from model.logistic_regression.archive.preprocessing import preprocessor

pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression()),
    ]
)
