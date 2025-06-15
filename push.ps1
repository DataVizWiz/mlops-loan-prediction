param (
    [string]$commMess
)

black .
git add .
git commit -m $commMess
git push