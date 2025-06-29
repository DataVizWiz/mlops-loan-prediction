DROP_FEATURES = ["LoanID"]
TARGET = "Default"

NUMERIC_FEATURES = [
    "Age",
    "Income",
    "LoanAmount",
    "CreditScore",
    "MonthsEmployed",
    "NumCreditLines",
    "InterestRate",
    "DTIRatio",
]

CATEGORICAL_FEATURES = [
    "LoanTerm",
    "Education",
    "EmploymentType",
    "MaritalStatus",
    "LoanPurpose",
    "HasMortgage",
    "HasDependents",
    "HasCoSigner",
]

RAW_PATH = "data/raw/loans.csv"
TEST_DATA_PATH = "data/test/X_test.csv"
REGISTRY = "models"
