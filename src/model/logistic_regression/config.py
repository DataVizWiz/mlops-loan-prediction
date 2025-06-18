DROP_FEATURES = ["LoanID"]
TARGET = "Default"

SCALE_FEATURES = [
    "Age",
    "Income",
    "LoanAmount",
    "CreditScore",
    "MonthsEmployed",
    "NumCreditLines",
    "InterestRate",
]

NORMALIZE_FEATURES = ["DTIRatio"]
ORDINAL_ENCODE_FEATURES = ["LoanTerm"]
ONEHOT_ENCODE_FEATURES = ["Education", "EmploymentType", "MaritalStatus", "LoanPurpose"]
BINARY_ENCODE_FEATURES = ["HasMortgage", "HasDependents", "HasCoSigner"]

RAW_PATH = "data/raw/loans.csv"
TEST_DATA_PATH = "data/test/X_test.csv"
REGISTRY = "models"
