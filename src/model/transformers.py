import numpy as np
import polars as pl

from typing import List
from sklearn.preprocessing import OrdinalEncoder


class Preprocessor:
    """Apply preprocessing on features."""

    def __init__(self, features: List[str], preprocessor: object):
        self.features = features
        self.preprocessor = preprocessor
        self.is_int = True if isinstance(preprocessor, OrdinalEncoder) else False

    def fit_transform(self, df_X: pl.DataFrame) -> pl.DataFrame:
        """Fit scaler and transform df."""
        fil_df = df_X.select(self.features)
        pp_arr = self.preprocessor.fit_transform(fil_df)

        if self.is_int:
            return df_X.with_columns(
                [
                    pl.Series(name, pp_arr[:, i]).cast(pl.Int8)
                    for i, name in enumerate(self.features)
                ]
            )
        else:
            scaled_df = pl.DataFrame(pp_arr, self.features)
            return df_X.with_columns(scaled_df)

    def transform(self, payload: dict) -> dict:
        """Scale new inputs."""
        arr = np.array([[payload[name] for name in self.features]])
        pp_arr = self.preprocessor.transform(arr)
        scaled_dict = dict(
            zip(
                self.features,
                (pp_arr[0].astype(np.int8) if self.is_int else pp_arr[0]),
            )
        )
        return {**payload, **scaled_dict}


# path = "data/raw/loans.csv"
# df = pl.read_csv(path)

# features = ["LoanTerm"]

# pp = Preprocessor(features, OrdinalEncoder())
# i = pp.fit_transform(df)
# print(i.select("LoanTerm"))

# loan_data = {
#     "Age": 56,
#     "Income": 85994,
#     "LoanAmount": 50587,
#     "CreditScore": 520,
#     "MonthsEmployed": 80,
#     "NumCreditLines": 4,
#     "InterestRate": 15.23,
#     "LoanTerm": 36,
#     "DTIRatio": 0.98773,
#     "Education": "Bachelor's",
#     "EmploymentType": "Full-time",
#     "MaritalStatus": "Divorced",
#     "HasMortgage": "Yes",
#     "HasDependents": "Yes",
#     "LoanPurpose": "Other",
#     "HasCoSigner": "Yes",
#     "Default": 0,
# }

# j = pp.transform(loan_data)
# print(j)
