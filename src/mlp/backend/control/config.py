NUM_FEATS = [
    "Age",
    "Income",
    "LoanAmount",
    "CreditScore",
    "MonthsEmployed",
    "NumCreditLines",
    "InterestRate",
    "DTIRatio",
]
CAT_FEATS = ["Education", "EmploymentType", "MaritalStatus", "LoanPurpose"]
BIN_FEATS = ["HasMortgage", "HasDependents", "HasCoSigner"]

LOAN_ID_COL = "LoanID"
TARGET_COL = "Default"

RAW_PATH = "data/raw/loans.csv"
TRANSFORMED_DIR = "data/transformed"
TEST_DATA_DIR = "data/test"
REGISTRY = "models"
