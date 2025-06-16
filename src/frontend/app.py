import streamlit as st


def handle_submit():
    st.session_state.form_submitted = True


if "form_submitted" not in st.session_state:
    st.session_state.form_submitted = False

if not st.session_state.form_submitted:
    with st.form("loan_predictions"):
        st.title("Loan Default Prediction")

        st.header("Numerical Inputs")
        st.number_input("The Age of the borrower", step=1)
        st.number_input("The annual income of the borrower (USD)", step=1)
        st.number_input("The amount of money being borrowed", step=1)
        st.number_input(
            "The credit score of the borrower, indicating their creditworthiness",
            step=1,
        )
        st.number_input("The number of months the borrower has been employed", step=1)
        st.number_input("The number of credit lines the borrower has open", step=1)
        st.number_input("The interest rate for the loan")
        st.number_input("The term length of the loan in months", step=1)
        st.number_input(
            "The Debt-to-Income ratio, indicating the borrowers debt compared to their income"
        )

        st.header("Categorical Inputs")
        st.radio(
            "The highest level of education attained by the borrower",
            ("PhD", "Master's", "Bachelor's", "High School"),
        )
        st.radio(
            "The type of employment status of the borrower",
            ("Full-time", "Part-time", "Self-employed", "Unemployed"),
        )
        st.radio(
            "The marital status of the borrower", ("Single", "Married", "Divorced")
        )
        st.radio("Whether the borrower has a mortgage", ("Yes", "No"))
        st.radio("Whether the borrower has dependents", ("Yes", "No"))
        st.radio(
            "The purpose of the loan",
            ("Home", "Auto", "Education", "Business", "Other"),
        )
        st.radio("Whether the loan has a co-signer", ("Yes", "No"))

        submitted = st.form_submit_button("Submit", on_click=handle_submit)
else:
    st.success("✅ Form submitted successfully!")
