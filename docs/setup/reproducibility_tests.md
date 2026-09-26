# LLM Implementation — Week 2

## Goal

Turn the repository into a proper installable Python project and establish a reproducible development workflow using:

- Python packages and imports
- `pyproject.toml`
- editable installation
- `pytest`
- random seed control
- CPU/GPU device handling
- scripts, tests, experiments, and notebooks

---

## 1. Python Modules, Packages, and Imports

A Python file is a **module**.

A directory containing related modules is a **package**.

Example project package:

```text
src/
└── llm_impl/
    ├── __init__.py
    ├── model.py
    ├── training.py
    └── utils.py
```

Typical imports:

```python
import llm_impl
from llm_impl.utils import set_seed
```

Python searches for importable modules using locations stored in:

```python
import sys
print(sys.path)
```

Useful debugging commands:

```bash
which python
python -c "import sys; print(sys.executable)"
python -c "import torch; print(torch.__file__)"
```

These help verify which interpreter and package installation are being used.

---

## 2. `__name__` and Script Entry Points

Reusable scripts should normally place executable logic inside `main()`:

```python
def main():
    ...

if __name__ == "__main__":
    main()
```

When a file is executed directly:

```python
__name__ == "__main__"
```

When imported, its module name is used instead.

This prevents importing a module from unintentionally starting a training or evaluation run.

---

## 3. Project Packaging with `pyproject.toml`

Repository structure:

```text
llm-implementation/
├── pyproject.toml
├── environment.yml
├── src/
│   └── llm_impl/
│       └── __init__.py
└── tests/
```

Example `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "llm-implementation"
version = "0.1.0"
description = "LLM implementation study project"
requires-python = ">=3.12"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

The repository name and Python import package do not need to match exactly:

```text
Repository/distribution: llm-implementation
Python package:          llm_impl
```

---

## 4. Editable Installation

Install the current project into the active Python environment with:

```bash
python -m pip install -e .
```

Meaning:

```text
python        use the active Python interpreter
-m pip       run that interpreter's pip module
install      install a project
-e           editable mode
.            current directory
```

Editable installation makes the environment reference the working source tree, so normal source-code edits do not require reinstalling the package.

Verify:

```bash
python -c "import llm_impl; print(llm_impl.__file__)"
```

Avoid manually modifying `sys.path` inside project code.

---

## 5. Generated `.egg-info`

Running an editable install with `setuptools` may create:

```text
*.egg-info/
```

This contains generated package metadata.

Do not edit it manually and normally do not commit it.

Add to `.gitignore`:

```gitignore
*.egg-info/
```

It can be regenerated with:

```bash
python -m pip install -e .
```

---

## 6. Testing the Installed Package

Tests should import the project through its normal package path.

Example:

```python
import torch

from llm_impl.utils import set_seed


def test_same_seed_same_tensor():
    set_seed(42)
    x1 = torch.randn(5)

    set_seed(42)
    x2 = torch.randn(5)

    assert torch.equal(x1, x2)


def test_different_seed_different_tensor():
    set_seed(42)
    x1 = torch.randn(5)

    set_seed(43)
    x2 = torch.randn(5)

    assert not torch.equal(x1, x2)
```

Run all tests:

```bash
python -m pytest
```

Typical testing targets in ML code include:

- tensor shapes
- masking behavior
- numerical invariants
- deterministic helper functions
- expected error conditions

For floating-point results, prefer tolerance-based comparisons such as:

```python
torch.allclose(actual, expected)
```

rather than exact equality when rounding error is possible.

---

## 7. Reproducibility

A training run depends on more than code:

```text
code
environment
data
configuration
randomness
hardware / numerical kernels
```

A useful seed utility:

```python
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)
```

Important:

```text
same seed ≠ guaranteed bitwise-identical GPU execution
```

GPU kernels, floating-point operation order, library versions, and hardware can still affect results.

Use fixed seeds mainly for reproducibility and debugging. For scientific comparisons, run multiple seeds and measure variation.

---

## 8. Recording System Information

Example utility:

```python
def system_info() -> dict:
    info = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cuda_version"] = torch.version.cuda

    return info
```

A serious experiment should eventually record:

```text
Git commit
environment/package versions
dataset/version
configuration
seed
device
metrics
checkpoint
logs
```

---

## 9. Scripts vs. Source Code

Use the following separation:

```text
src/          reusable implementation
scripts/      executable workflows
tests/        correctness checks
experiments/  research/model comparisons
notebooks/    interactive exploration and visualization
scratch/      temporary learning/debugging work
```

Reusable model logic belongs under `src/`.

Example:

```text
src/llm_impl/
├── model.py
├── attention.py
├── training.py
└── utils.py
```

Executable tasks belong under `scripts/`.

Example:

```text
scripts/
├── train.py
├── evaluate.py
└── generate.py
```

Notebooks should consume reusable package code rather than becoming the only place where important implementation exists.

---

## 10. Reproducible Script Pattern

Example:

```python
import torch
from torch import nn

from llm_impl.utils import set_seed, system_info


def main():
    seed = 42
    input_dim = 4
    output_dim = 2
    batch_size = 3

    set_seed(seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = nn.Linear(input_dim, output_dim).to(device)

    x = torch.randn(
        batch_size,
        input_dim,
        device=device,
    )

    with torch.no_grad():
        output = model(x)

    print("seed:", seed)
    print("device:", device)
    print("system info:", system_info())
    print("output shape:", output.shape)


if __name__ == "__main__":
    main()
```

Run:

```bash
python scripts/repro_demo.py
```

---

## 11. Recommended Repository Layout

```text
llm-implementation/
├── README.md
├── environment.yml
├── pyproject.toml
├── src/
│   └── llm_impl/
│       ├── __init__.py
│       └── utils.py
├── scripts/
│   └── repro_demo.py
├── tests/
│   └── test_reproducibility.py
├── experiments/
├── notebooks/
├── scratch/
└── docs/
```

Large datasets, model weights, checkpoints, caches, and experiment outputs should remain outside Git and live on dedicated external storage.

---

## 12. Standard Development Workflow

From the repository root:

```bash
conda activate llm

python -c "import llm_impl; print(llm_impl.__file__)"

python scripts/repro_demo.py

python -m pytest

git status
```

After tests pass:

```bash
git add .
git commit -m "Set up Python package and reproducibility workflow"
git push
```

Conceptually:

```text
edit code
→ run tests
→ inspect changes
→ commit
→ push
```

---

## Week 2 Takeaways

```text
file
→ module
→ package
→ installed project
```

and:

```text
source code
→ tests
→ scripts
→ experiments
```

Key files and commands:

```text
pyproject.toml
environment.yml
python -m pip install -e .
python -m pytest
```

The project is now structured so later Transformer components can be implemented as reusable, testable, reproducible Python code.
