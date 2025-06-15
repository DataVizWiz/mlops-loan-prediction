(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -

$env:Path += ";$env:APPDATA\Python\Scripts"

poetry --version
poetry new --src .

