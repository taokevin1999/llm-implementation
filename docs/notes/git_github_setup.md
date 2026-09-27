# Git and GitHub Setup Guide for WSL

This guide documents the Git/GitHub setup used for the `llm-implementation` project inside WSL (Ubuntu).

It assumes:

- WSL and Ubuntu are already installed.
- You are working inside your Linux home directory, e.g. `/home/<username>`.
- GitHub account already exists.
- The project directory is:

```bash
~/projects/llm-implementation
```

---

## 1. Check that Git is installed

Run:

```bash
git --version
```

Example output:

```text
git version 2.43.0
```

If a version number appears, Git is installed.

---

## 2. Configure Git identity

Git attaches an author name and email address to every commit.

### Set author name

```bash
git config --global user.name "<username>"
```

### Set author email

Use an email associated with your GitHub account, or GitHub's `noreply` email if you prefer privacy.

```bash
git config --global user.email "YOUR_EMAIL"
```

### Verify configuration

```bash
git config --global --list
```

Expected entries include:

```text
user.name=<username>
user.email=YOUR_EMAIL
```

The Git author name does not need to exactly match the GitHub display name.

The email is more important for GitHub commit attribution.

---

## 3. Check for an existing SSH key

Run:

```bash
ls -al ~/.ssh
```

If the directory does not exist, that is normal on a new WSL installation.

Typical existing SSH key files would be:

```text
id_ed25519
id_ed25519.pub
```

---

## 4. Generate a new SSH key

Run:

```bash
ssh-keygen -t ed25519 -C "YOUR_GITHUB_EMAIL"
```

Explanation:

- `ssh-keygen` creates SSH keys.
- `-t ed25519` selects the Ed25519 key type.
- `-C` attaches a comment, usually your GitHub email.

When asked:

```text
Enter file in which to save the key (~/.ssh/id_ed25519):
```

press **Enter** to accept the default.

You may also choose a passphrase for additional protection.

The command creates:

```text
~/.ssh/id_ed25519
~/.ssh/id_ed25519.pub
```

Important:

```text
id_ed25519       = private key — NEVER share
id_ed25519.pub   = public key — safe to share with GitHub
```

---

## 5. Verify SSH key files

Run:

```bash
ls -al ~/.ssh
```

Typical permissions look like:

```text
-rw------- id_ed25519
-rw-r--r-- id_ed25519.pub
```

The private key is intentionally more restricted.

---

## 6. Start the SSH agent

Run:

```bash
eval "$(ssh-agent -s)"
```

Example output:

```text
Agent pid 4829
```

The SSH agent keeps an unlocked SSH key available in memory during the session.

---

## 7. Add the private key to the SSH agent

Run:

```bash
ssh-add ~/.ssh/id_ed25519
```

If a passphrase was set, enter it when prompted.

A successful result looks similar to:

```text
Identity added: ~/.ssh/id_ed25519
```

---

## 8. Display the public key

Run:

```bash
cat ~/.ssh/id_ed25519.pub
```

`cat` prints the contents of a file to the terminal.

The output should be one long line beginning with:

```text
ssh-ed25519 ...
```

Copy the entire line.

Do **not** display or share the private key file:

```text
~/.ssh/id_ed25519
```

---

## 9. Add the public key to GitHub

On GitHub:

```text
Profile picture
→ Settings
→ SSH and GPG keys
→ New SSH key
```

Suggested title:

```text
My WSL Laptop
```

Choose:

```text
Authentication Key
```

Paste the complete contents of:

```text
~/.ssh/id_ed25519.pub
```

and save the key.

---

## 10. Test GitHub SSH authentication

Run:

```bash
ssh -T git@github.com
```

Explanation:

- `ssh` connects using SSH.
- `-T` disables allocation of an interactive terminal.
- `git@github.com` is GitHub's SSH endpoint.

The first connection may ask whether you trust GitHub's host key.

If appropriate, type:

```text
yes
```

A successful authentication message says that authentication succeeded but GitHub does not provide shell access.

---

# Creating the Local Repository

## 11. Go to the project directory

```bash
cd ~/projects/llm-implementation
```

Verify:

```bash
pwd
```

Expected:

```text
~/projects/llm-implementation
```

---

## 12. Check whether the directory is already a Git repository

Run:

```bash
git status
```

If it says:

```text
fatal: not a git repository
```

the folder has not yet been initialized.

---

## 13. Initialize the repository

Run:

```bash
git init
```

This creates the hidden directory:

```text
.git/
```

The `.git` directory contains Git's repository metadata and history.

Conceptually:

```text
llm-implementation/
├── .git/
└── project files...
```

Do not manually edit files inside `.git`.

---

## 14. Rename the default branch to `main`

If Git initialized the repository on `master`, run:

```bash
git branch -M main
```

Verify:

```bash
git status
```

Expected:

```text
On branch main
```

---

# First Commit

## 15. Create a README

Run:

```bash
touch README.md
```

`touch` creates an empty file if it does not already exist.

Check Git status:

```bash
git status
```

The file should appear under:

```text
Untracked files:
```

An **untracked file** exists on disk, but Git is not yet including it in version history.

---

## 16. Understand the basic Git state model

The core workflow is:

```text
Working directory
      ↓ git add
Staging area
      ↓ git commit
Local Git history
      ↓ git push
GitHub remote repository
```

---

## 17. Stage the README

Run:

```bash
git add README.md
```

Then:

```bash
git status
```

The file should now appear under:

```text
Changes to be committed:
```

The staging area lets you choose exactly what will be included in the next commit.

---

## 18. Create the first commit

Run:

```bash
git commit -m "Initialize repository"
```

Explanation:

- `git commit` records the staged snapshot in local repository history.
- `-m` supplies the commit message directly.

After committing:

```bash
git status
```

should show:

```text
nothing to commit, working tree clean
```

A **commit is local**. It does not automatically upload anything to GitHub.

---

# Connecting the Local Repository to GitHub

## 19. Create an empty GitHub repository

Create a GitHub repository named:

```text
llm-implementation
```

The GitHub repository name does not technically need to match the local folder name, but using the same name reduces confusion.

Because the local repository already exists, do **not** initialize the GitHub repository with:

```text
README
.gitignore
license
```

This avoids creating a separate initial history on GitHub.

---

## 20. Add the GitHub repository as a remote

Copy the repository's **SSH URL**, which looks like:

```text
git@github.com:YOUR_USERNAME/llm-implementation.git
```

Then run:

```bash
git remote add origin git@github.com:YOUR_USERNAME/llm-implementation.git
```

Here:

```text
origin
```

is simply the conventional nickname for the primary remote repository.

Verify:

```bash
git remote -v
```

You should see `origin` listed for both fetch and push.

---

## 21. Push `main` to GitHub

Run:

```bash
git push -u origin main
```

Explanation:

- `git push` uploads local commits to a remote.
- `origin` is the remote repository nickname.
- `main` is the branch being pushed.
- `-u` sets `origin/main` as the upstream branch for local `main`.

After this initial setup:

```text
local main ↔ origin/main
```

Future pushes can usually be performed with:

```bash
git push
```

The distinction is:

```text
git commit = save snapshot locally
git push   = upload existing commits to GitHub
```

You can create several local commits before pushing them.

---

# Adding a `.gitignore`

## 22. Create `.gitignore`

Run:

```bash
touch .gitignore
```

Edit it with:

```bash
nano .gitignore
```

For the LLM implementation project, begin with:

```gitignore
# Python cache files
__pycache__/
*.pyc

# Virtual environments
.venv/
venv/

# Jupyter notebook checkpoints
.ipynb_checkpoints/

# Environment / secret files
.env

# Large local artifacts
data/
checkpoints/
outputs/
```

Save in Nano using:

```text
Ctrl + O
Enter
Ctrl + X
```

Verify the contents with:

```bash
cat .gitignore
```

---

## 23. Why `.gitignore` matters for ML projects

Git should mainly track:

```text
source code
configuration files
small documentation files
tests
lightweight experiment definitions
```

Git should generally not track:

```text
large datasets
model checkpoints
generated outputs
Python caches
virtual environments
credentials or API keys
```

For this course, large datasets, model weights, checkpoints, caches, and experiment artifacts will increasingly be stored on the 4 TB external SSD rather than inside Git.

---

## 24. Commit and push `.gitignore`

Check:

```bash
git status
```

Stage:

```bash
git add .gitignore
```

Commit:

```bash
git commit -m "Add gitignore"
```

Push:

```bash
git push
```

---

# Inspecting Commit History

## 25. View compact commit history

Run:

```bash
git log --oneline
```

Example:

```text
a1b2c3d Add gitignore
d4e5f6g Initialize repository
```

The characters at the left are abbreviated commit hashes.

Each commit represents a recorded project snapshot.

---

# Daily Git Workflow

The normal workflow for this course will usually be:

```bash
git status
```

Inspect what changed.

Then stage selected files:

```bash
git add FILE_NAME
```

or, later when appropriate:

```bash
git add .
```

Create a commit:

```bash
git commit -m "Short description of change"
```

Push committed work:

```bash
git push
```

A useful mental model is:

```text
edit
  ↓
inspect
  ↓
stage
  ↓
commit locally
  ↓
push to GitHub
```

---

# Important Commands Learned

| Command | Purpose |
|---|---|
| `git --version` | Check installed Git version |
| `git config --global ...` | Configure global Git identity |
| `ssh-keygen` | Generate SSH key pair |
| `ssh-add` | Add private key to SSH agent |
| `ssh -T git@github.com` | Test GitHub SSH authentication |
| `git init` | Turn a directory into a Git repository |
| `git status` | Inspect repository state |
| `git branch -M main` | Rename current branch to `main` |
| `git add` | Stage changes |
| `git commit` | Record staged changes locally |
| `git remote add origin ...` | Connect a remote repository |
| `git remote -v` | Show configured remotes |
| `git push` | Upload commits to GitHub |
| `git log --oneline` | Show compact commit history |
| `cat FILE` | Print a file's contents |
| `touch FILE` | Create an empty file |
| `nano FILE` | Edit a text file in the terminal |

---

# Current Project State

At the end of this setup, the repository should roughly look like:

```text
~/projects/llm-implementation/
├── .git/
├── .gitignore
└── README.md
```

and the local `main` branch should track:

```text
origin/main
```

on GitHub.

This completes the initial Git/GitHub setup for Week 1 of the LLM implementation course.
