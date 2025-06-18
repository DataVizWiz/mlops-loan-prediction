import pandas as pd
import model.logistic_regression.config as cfg

from model.logistic_regression.preprocessing import PreprocessLoans
from model.logistic_regression.pipeline import pipeline
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    preprocessor = PreprocessLoans(cfg.RAW_PATH)
    preprocessor.transform()
    print(preprocessor.df_X)
    print(preprocessor.df_y)

# X = df.drop(cfg.TARGET, axis=1)
# y= df[cfg.TARGET]

# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

# test_data = X_test.copy()
# test_data[cfg.TARGET] = y_test
# test_data.to_csv(cfg.TEST_DATA_PATH)

# pipeline.fit(X_train, y_train)
