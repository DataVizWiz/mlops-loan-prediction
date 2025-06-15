python setup.py bdist_wheel
pip uninstall -y dist/mlops_loan_default_predictions-0.1.0-py3-none-any.whl
pip install dist/mlops_loan_default_predictions-0.1.0-py3-none-any.whl