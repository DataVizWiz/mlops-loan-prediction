param (
    [string]$commMess
)

Remove-Item src/mlops_loan_default_predictions.egg-info -Recurse -Force
pip freeze > requirements.dev.txt
black .
git add .
git commit -m $commMess
git push