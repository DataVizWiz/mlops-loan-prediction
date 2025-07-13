import streamlit as st

# def handle_submit():
#     st.session_state.form_submitted = True

# if "form_submitted" not in st.session_state:
#     st.session_state.form_submitted = False

# if not st.session_state.form_submitted:
with st.form("loan_predictions"):
    st.title("Loan Default Prediction")

    st.header("Numerical Inputs")
    age = st.number_input("The Age of the borrower", step=1)
    income = st.number_input("The annual income of the borrower (USD)", step=1)
    loan_amount = st.number_input("The amount of money being borrowed", step=1)
    credit_score = st.number_input(
        "The credit score of the borrower, indicating their creditworthiness", step=1
    )
    months_employed = st.number_input(
        "The number of months the borrower has been employed", step=1
    )
    num_credit_lines = st.number_input(
        "The number of credit lines the borrower has open", step=1
    )
    interest_rate = st.number_input("The interest rate for the loan")
    loan_term = st.number_input("The term length of the loan in months", step=1)
    dti_ratio = st.number_input(
        "The Debt-to-Income ratio, indicating the borrowers debt compared to their income"
    )

    st.header("Categorical Inputs")
    education = st.radio(
        "The highest level of education attained by the borrower",
        ("PhD", "Master's", "Bachelor's", "High School"),
    )
    employment_type = st.radio(
        "The type of employment status of the borrower",
        ("Full-time", "Part-time", "Self-employed", "Unemployed"),
    )
    marital_status = st.radio(
        "The marital status of the borrower", ("Single", "Married", "Divorced")
    )
    has_mortgage = st.radio("Whether the borrower has a mortgage", ("Yes", "No"))
    has_dependents = st.radio("Whether the borrower has dependents", ("Yes", "No"))
    loan_purpose = st.radio(
        "The purpose of the loan", ("Home", "Auto", "Education", "Business", "Other")
    )
    has_cosigner = st.radio("Whether the loan has a co-signer", ("Yes", "No"))

    submit = st.form_submit_button("Predict")

    if submit:
        input_data = {
            "Age": age,
            "Income": income,
            "LoanAmount": loan_amount,
            "CreditScore": credit_score,
            "MonthsEmployed": months_employed,
            "NumCreditLines": num_credit_lines,
            "InterestRate": interest_rate,
            "LoanTerm": loan_term,
            "DTIRatio": dti_ratio,
            "Education": education,
            "EmploymentType": employment_type,
            "MaritalStatus": marital_status,
            "HasMortgage": has_mortgage,
            "HasDependents": has_dependents,
            "LoanPurpose": loan_purpose,
            "HasCoSigner": has_cosigner,
            "Default": 0,
        }

        try:
            response = requests.post("http://localhost:8000/predict", json=input_data)
            result = response.json()
            prediction = result.get("prediction")

            if prediction:
                st.success(f"Prediction: {prediction}")
            else:
                st.warning("Received an empty response from the backend.")

        except Exception as e:
            st.error(f"Failed to connect to backend: {e}")

    # submitted = st.form_submit_button("Submit", on_click=handle_submit)
# else:
#     st.success("✅ Form submitted successfully!")
