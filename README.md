# multiview-seg-dtu

### Basic git
#### How to create a branch
```bash
# MAKE SURE YOU ARE IN MAIN
git status
git pull
# CREATE NEW BRANCH
git checkout -b <branch-name>
```

#### How to push new branch
```bash
# MAKE SURE YOU ARE IN THE SPECIFIC BRANCH
git status
git push --set-upstream origin <branch-name> 
```

#### How to commit new changes
```bash
# MAKE SURE YOU ARE IN THE SPECIFIC BRANCH
git status
# IF NEW FILES ADDED
git add <file-names>
# OR TO COMMIT ALL THE NEW FILES
git add .
# COMMIT AND ADD A COMMENT
git commit -m 'your message here'
# UPLOAD CHANGES
git push
```

#### How to pull changes from main to another branch

``` bash
# MAKE SURE YOUR BRANCH IS UPTODATE
git pull
# SWITCH TO MAIN
git checkout main
git pull
git checkout <branch-name>
git merge main
# EDIT CODE
# IF NEW FILES CREATED
git add .
# THEN COMMIT CHANGES AND ADD A MESSAGE
git commit -a -m “message”
git push
# CHECK EVERYTHING UPLOADED CORRECTLY
git status
# THEN FROM GITHUB CREATE A PULL REQUEST TO MAIN
```

#### Create environment

``` bash
# CREATE ENV
python3.12 -m venv <name-of-env>
# TO ACTIVATE ENV ON WINDOWS
.\<name-of-env>\Scripts\activate
# TO ACTIVATE ENV ON MAC
source <name-of-env>/bin/activate
# INSTALL LIBRARIES
pip install -r requirements.txt
# SELEC

```
