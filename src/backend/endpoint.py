import joblib
import polars as pl
import model.logistic_regression.config as cfg

from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()
model_path = f"{cfg.REGISTRY}/logistic_regression.pkl"
model = joblib.load(model_path)


class LoanInputs(BaseModel):
    age: int
    income: int
    loan_amount: int
    credit_score: int
    months_employed: int
    num_credit_lines: int
    interest_rate: float
    loan_term: int
    dti_ratio: float
    education: str
    employment_type: str
    marital_status: str
    has_mortgage: str
    has_dependents: str
    loan_purpose: str
    has_co_signer: str


class LoanPrediction(BaseModel):
    default: int


@app.get("/")
async def read_root():
    return {"message": "Welcome to Loan Predictions"}


@app.post("/predict", response_model=LoanPrediction)
def predict(payload: LoanInputs):
    data = payload.model_dump()
    model_input = {
        "age": data["Age"],
        "income": data["Income"],
        "loan_amount": data["LoanAmount"],
        "credit_score": data["CreditScore"],
        "months_employed": data["EmploymentDuration"],
        "num_credit_lines": data["NumCreditLines"],
        "interest_rate": data["InterestRate"],
        "loan_term": data["LoanTerm"],
        "dti_ratio": data["DTIRatio"],
        "education": data["Education"],
        "employment_type": data["EmploymentType"],
        "marital_status": data["MaritalStatus"],
        "has_mortgage": data["HasMortgage"],
        "has_dependents": data["HasDependents"],
        "loan_purpose": data["LoanPurpose"],
        "has_co_signer": data["HasCoSigner"],
    }

    # preprocess data
    # make prediction
