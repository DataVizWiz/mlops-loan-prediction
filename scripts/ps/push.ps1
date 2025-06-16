param (
    [string]$commMess
)

pipdeptree --freeze --warn silence > .\requirements.dev.txt
black src
git add .
git commit -m $commMess
git push