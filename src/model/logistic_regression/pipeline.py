from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline(
    [
        ("scaler",),
        ("normalizer",),
        ("ordinal_encoder",),
        ("onehot_encoder",),
        ("binary_encoder",),
        ("classifier", LogisticRegression()),
    ]
)
