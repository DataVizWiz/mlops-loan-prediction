param (
    [string]$commMess
)

black src
git add .
git commit -m $commMess
git push