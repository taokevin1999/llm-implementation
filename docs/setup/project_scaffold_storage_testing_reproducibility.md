# LLM Implementation Project Setup — VS Code, Project Scaffold, External SSD, Testing, and Reproducibility

This guide documents the remaining Week 1 setup for an LLM implementation project after the initial WSL/Python/PyTorch/GPU setup and Git/GitHub setup are complete.

It is sanitized: usernames, emails, device-specific names, and other personal identifiers are represented with generic placeholders.

---

## 1. Open the project in VS Code through WSL

The project directory should live inside the WSL Linux filesystem, for example:

```text
~/projects/llm-implementation
```

Check whether the VS Code command is available:

```bash
code --version
```

Open the project:

```bash
cd ~/projects/llm-implementation
code .
```

The `.` means “the current directory.”

Conceptually:

```text
Windows
└── VS Code interface
    └── WSL: Ubuntu
        ├── project files
        ├── Git
        ├── Python
        ├── PyTorch
        └── CUDA access
```

Install the Microsoft **WSL** extension in VS Code. When the project is correctly opened through WSL, the lower-left corner should indicate something similar to:

```text
WSL: Ubuntu
```

---

## 2. Use the VS Code integrated terminal

Open:

```text
Terminal → New Terminal
```

or press:

```text
Ctrl + `
```

Check the current directory:

```bash
pwd
```

Expected:

```text
/home/<username>/projects/llm-implementation
```

The VS Code terminal and a standalone WSL terminal use the same Ubuntu environment when VS Code is connected to WSL. The main difference to watch is Python environment activation.

For example:

```bash
which python
```

may point to a different interpreter if the `llm` environment is not active.

---

## 3. Select the correct Python interpreter

Install the Microsoft **Python** extension on the WSL side of VS Code.

Open the command palette:

```text
Ctrl + Shift + P
```

Run:

```text
Python: Select Interpreter
```

Select:

```text
~/miniforge3/envs/llm/bin/python
```

Avoid selecting:

```text
/usr/bin/python3
```

or the Miniforge `base` environment.

If VS Code says:

```text
CONDA_PREFIX is set for this VS Code session.
Selection saved for new terminals only.
```

close the current terminal and open a new one.

Verify:

```bash
which python
```

Expected:

```text
/home/<username>/miniforge3/envs/llm/bin/python
```

---

## 4. Project-specific VS Code settings

VS Code may create:

```text
.vscode/
```

Inspect it before committing:

```bash
ls -la .vscode
```

A generic `settings.json` such as:

```json
{
    "python-envs.defaultEnvManager": "ms-python.python:conda",
    "python-envs.defaultPackageManager": "ms-python.python:conda"
}
```

is safe to commit.

Do not commit settings containing personal absolute paths, usernames, secrets, or credentials.

---

## 5. Create an environment check script

Create:

```text
scripts/check_environment.py
```

with:

```python
import sys
import torch

print("Python executable:", sys.executable)
print("Python version:", sys.version)
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

Run:

```bash
python scripts/check_environment.py
```

Expected behavior:

```text
Python executable: /home/<username>/miniforge3/envs/llm/bin/python
PyTorch version: ...
CUDA available: True
GPU: NVIDIA GeForce RTX 4080 Laptop GPU
```

This verifies:

```text
VS Code
   ↓
WSL / Ubuntu
   ↓
llm Python environment
   ↓
PyTorch
   ↓
CUDA
   ↓
NVIDIA GPU
```

---

## 6. Create the project scaffold

At the repository root:

```bash
mkdir -p src tests configs
```

Recommended structure:

```text
llm-implementation/
├── src/
├── tests/
├── configs/
├── scripts/
├── docs/
├── .vscode/
├── .gitignore
├── environment.yml
└── README.md
```

Typical roles:

```text
src/      main implementation code
tests/    automated tests
configs/  model/training configuration files
scripts/  executable utilities
docs/     setup and project documentation
```

`src` is simply conventional shorthand for **source**.

---

## 7. Create the Python package directory

Create:

```bash
mkdir -p src/llm_impl
touch src/llm_impl/__init__.py
```

Result:

```text
src/
└── llm_impl/
    └── __init__.py
```

Later, the package may contain files such as:

```text
attention.py
embeddings.py
layers.py
model.py
training.py
```

---

## 8. External SSD storage plan

Keep code, Git history, and Python environments on the WSL Linux filesystem.

Use the external SSD for large artifacts.

Recommended division:

```text
WSL internal filesystem
├── Git repository
├── source code
├── Python environments
├── configs
└── documentation

External SSD
├── datasets
├── checkpoints
├── model weights
├── caches
├── experiment outputs
└── research artifacts
```

---

## 9. Mount the external SSD in WSL

Suppose Windows assigns the external SSD:

```text
F:
```

Check mounted Windows drives:

```bash
ls /mnt
```

If `F:` does not appear automatically, create a mount point:

```bash
sudo mkdir -p /mnt/f
```

Mount it:

```bash
sudo mount -t drvfs F: /mnt/f
```

Verify:

```bash
df -h /mnt/f
```

A nominal 4 TB drive typically appears as approximately:

```text
3.6T
```

usable capacity.

---

## 10. Create the SSD directory layout

Create:

```bash
mkdir -p /mnt/f/llm-storage/{datasets,checkpoints,model_weights,caches,outputs,research}
```

Verify:

```bash
ls /mnt/f/llm-storage
```

Expected:

```text
caches
checkpoints
datasets
model_weights
outputs
research
```

---

## 11. Verify SSD write access

Create a temporary file:

```bash
touch /mnt/f/llm-storage/test_file.txt
```

Verify:

```bash
ls -l /mnt/f/llm-storage
```

Remove it:

```bash
rm /mnt/f/llm-storage/test_file.txt
```

This confirms WSL can both read from and write to the SSD.

---

## 12. Create a symbolic link from the repository to the SSD

From the project root:

```bash
ln -s /mnt/f/llm-storage storage
```

This creates:

```text
storage -> /mnt/f/llm-storage
```

A symbolic link is effectively a Linux filesystem shortcut.

The project can now refer to:

```text
storage/datasets
storage/checkpoints
storage/model_weights
storage/caches
storage/outputs
storage/research
```

while the actual files remain on the external SSD.

---

## 13. Ignore the storage symlink in Git

Because the symlink is machine-specific, do not commit it.

Add to `.gitignore`:

```gitignore
storage
```

Then verify:

```bash
git status
```

The `storage` symlink should not appear as an untracked file.

---

## 14. Check SSD mount persistence

From Windows PowerShell:

```powershell
wsl --shutdown
```

Then reopen WSL and check:

```bash
df -h /mnt/f
```

and:

```bash
ls /mnt/f/llm-storage
```

Also verify the project symlink:

```bash
ls storage
```

If everything still works, no additional mount configuration is needed.

Two caveats:

```text
1. If the SSD is unplugged, storage/ points to a missing location.
2. If Windows assigns a different drive letter, the symlink target must be updated.
```

---

## 15. Install pytest

With the `llm` environment active:

```bash
python -m pip install pytest
```

Verify:

```bash
pytest --version
```

Observed in this setup:

```text
pytest 9.1.1
```

Using `python -m pip` makes it explicit that the package is installed into the currently active Python interpreter.

---

## 16. Create the first automated test

Create:

```text
tests/test_smoke.py
```

with:

```python
def test_python_arithmetic():
    assert 2 + 2 == 4
```

Run:

```bash
pytest
```

Expected:

```text
1 passed
```

Pytest discovers tests automatically when:

```text
file names begin with test_
function names begin with test_
```

The `assert` statement means that the condition must be true or the test fails.

Later, tests can check:

```text
tensor shapes
attention masks
forward-pass dimensions
gradient behavior
model configuration validity
```

---

## 17. Create a reproducible environment file

Create:

```text
environment.yml
```

with:

```yaml
name: llm

channels:
  - conda-forge

dependencies:
  - python=3.12
  - pip
  - pip:
      - --extra-index-url https://download.pytorch.org/whl/cu126
      - torch==2.14.0+cu126
      - pytest==9.1.1
```

Conceptually:

```text
environment.yml
      ↓
Mamba reads environment recipe
      ↓
Python 3.12
      ↓
pip
      ↓
PyTorch CUDA build
      ↓
pytest
```

---

## 18. Recreate the environment on another machine

A fresh machine could run:

```bash
mamba env create -f environment.yml
```

then:

```bash
mamba activate llm
```

Do not run the create command if an environment named `llm` already exists unless you intentionally want to recreate it.

---

## 19. Human-readable environment recipe vs. lock file

A command such as:

```bash
pip freeze
```

records many transitive dependencies that were installed automatically.

For a learning project, a small human-readable recipe is easier to understand and maintain.

Useful distinction:

```text
environment.yml
    = human-readable dependency recipe

lock file
    = exact resolved dependency versions
```

A stricter lock file can be introduced later.

---

## 20. Commit the project scaffold

Typical files worth committing include:

```text
src/llm_impl/__init__.py
tests/test_smoke.py
scripts/check_environment.py
.vscode/settings.json
environment.yml
.gitignore
docs/
README.md
```

Files and directories that should generally remain outside Git include:

```text
storage/
datasets
checkpoints
large model weights
generated outputs
caches
secrets
API keys
```

---

## 21. Normal Git workflow

The core workflow remains:

```text
edit files
   ↓
git status
   ↓
git add
   ↓
git commit
   ↓
git push
```

Example:

```bash
git status
git add environment.yml
git commit -m "Add reproducible Python environment"
git push
```

---

## 22. When GitHub has changes that local Git does not

If files are edited or deleted directly on GitHub, the local and remote histories can diverge.

Example:

```text
Local:
A → B → C

GitHub:
A → B → D
```

A push may be rejected because Git does not want to overwrite remote history.

Use:

```bash
git pull --rebase origin main
```

Conceptually:

```text
1. Download GitHub changes.
2. Temporarily set aside local commits.
3. Update local history to the remote history.
4. Replay local commits on top.
```

Result:

```text
A → B → D → C'
```

Then:

```bash
git push
```

This preserves both sets of changes.

Avoid:

```bash
git push --force
```

unless you deliberately intend to rewrite remote history and understand the consequences.

---

## 23. Final Week 1 sanity checks

### Git

```bash
git status
```

Desired:

```text
nothing to commit, working tree clean
```

### Python / PyTorch / GPU

```bash
python scripts/check_environment.py
```

Confirm:

```text
correct Python interpreter
PyTorch imports successfully
CUDA available: True
expected NVIDIA GPU detected
```

### Tests

```bash
pytest
```

Confirm:

```text
1 passed
```

---

## 24. Final project architecture

At the end of Week 1:

```text
llm-implementation/
├── .git/
├── .vscode/
│   └── settings.json
├── configs/
├── docs/
│   └── setup/
├── scripts/
│   └── check_environment.py
├── src/
│   └── llm_impl/
│       └── __init__.py
├── tests/
│   └── test_smoke.py
├── storage -> /mnt/f/llm-storage
├── .gitignore
├── environment.yml
└── README.md
```

The external SSD contains:

```text
/mnt/f/llm-storage/
├── datasets/
├── checkpoints/
├── model_weights/
├── caches/
├── outputs/
└── research/
```

The `storage` symlink provides a convenient project-relative path while keeping large files outside Git and off the internal WSL filesystem.

---

## 25. Completed Week 1 development stack

```text
Windows
└── VS Code
    └── WSL / Ubuntu
        ├── Git + GitHub
        ├── Miniforge
        │   └── llm environment
        │       ├── Python
        │       ├── PyTorch
        │       └── pytest
        ├── CUDA
        ├── NVIDIA GPU
        ├── project source code
        ├── automated tests
        └── external SSD storage
```

This completes the Week 1 project setup, storage, testing, and reproducibility layer.
