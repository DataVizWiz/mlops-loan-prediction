from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model.logistic_regression.preprocessing import ModelPredictor

app = FastAPI()


class LoanData(BaseModel):
    Age: int
    Income: float
    LoanAmount: float
    CreditScore: int
    MonthsEmployed: int
    NumCreditLines: int
    InterestRate: float
    LoanTerm: int
    DTIRatio: float
    Education: str
    EmploymentType: str
    MaritalStatus: str
    HasMortgage: str
    HasDependents: str
    LoanPurpose: str
    HasCoSigner: str
    Default: int


@app.post("/predict")
def predict(loan: LoanData):
    try:
        predictor = ModelPredictor()
        predictor.transform(loan.dict())
        return {"predictions": predictor.df.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
