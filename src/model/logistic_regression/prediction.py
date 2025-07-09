from model.logistic_regression.preprocessing import ModelPredictor

if __name__ == "__main__":
    loan_data = {
        "Age": 56,
        "Income": 85994,
        "LoanAmount": 50587,
        "CreditScore": 520,
        "MonthsEmployed": 80,
        "NumCreditLines": 4,
        "InterestRate": 15.23,
        "LoanTerm": 36,
        "DTIRatio": 0.98773,
        "Education": "Bachelor's",
        "EmploymentType": "Full-time",
        "MaritalStatus": "Divorced",
        "HasMortgage": "Yes",
        "HasDependents": "Yes",
        "LoanPurpose": "Other",
        "HasCoSigner": "Yes",
        "Default": 0,
    }

    predictor = ModelPredictor()
    predictor.transform(loan_data)
    print(predictor.df)
