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

LOAN_ID_COL = "LoanID"
TARGET_COL = "Default"

RAW_PATH = "data/raw/loans.csv"
TRANSFORMED_DIR = "data/transformed"
TEST_DATA_DIR = "data/test"
REGISTRY = "models"
