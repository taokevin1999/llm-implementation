# WSL 2 + Ubuntu + Python + PyTorch + NVIDIA GPU Setup Guide

This guide documents the working setup used for an LLM implementation environment on Windows with WSL 2, Ubuntu 24.04 LTS, Miniforge, Python 3.12, PyTorch, and an NVIDIA RTX 4080 Laptop GPU.

## 1. Install WSL 2

Open **PowerShell** and check/install WSL.

```powershell
wsl --install
```

If WSL is installed but no Linux distribution is present, list available distributions:

```powershell
wsl --list --online
```

or:

```powershell
wsl -l -o
```

## 2. Install Ubuntu 24.04 LTS

Install Ubuntu 24.04:

```powershell
wsl --install -d Ubuntu-24.04
```

On first launch, Ubuntu asks you to create:

- a Linux username
- a Linux password

The password is used for `sudo`.

> When entering a Linux password in the terminal, no characters, dots, or asterisks appear. This is normal.

---

## 3. Enter the Linux Home Directory

If WSL is launched while PowerShell is in:

```text
C:\Windows\System32
```

then `pwd` may initially show:

```text
/mnt/c/WINDOWS/system32
```

This is normal. WSL translated the Windows path into its Linux-mounted equivalent.

Move to the Linux home directory:

```bash
cd ~
```

Check:

```bash
pwd
```

For this setup:

```text
/home/<username>
```

Useful path mapping:

```text
Windows C:\        <-> Linux /mnt/c/
C:\Windows         <-> /mnt/c/WINDOWS
C:\Users           <-> /mnt/c/Users
Linux home         <-> /home/<username>
```

For development work, prefer the Linux filesystem under `/home/...` rather than working inside `/mnt/c/...`.

---

## 4. Verify Ubuntu Version

Run:

```bash
lsb_release -a
```

Working setup:

```text
Distributor ID: Ubuntu
Description:    Ubuntu 24.04.5 LTS
Release:        24.04
Codename:       noble
```

The message:

```text
No LSB modules are available.
```

is harmless.

---

## 5. Update Ubuntu Packages

Refresh the package catalog:

```bash
sudo apt update
```

Install available upgrades:

```bash
sudo apt upgrade -y
```

Difference:

```text
apt update       -> refreshes the list of available packages
apt upgrade      -> installs newer versions of installed packages
```

---

## 6. Check Basic Development Tools

Check Python:

```bash
python3 --version
```

Observed:

```text
Python 3.12.3
```

Check Git:

```bash
git --version
```

Observed:

```text
git version 2.43.0
```

Check GCC:

```bash
gcc --version
```

If GCC is missing, install the standard build tool bundle:

```bash
sudo apt install build-essential -y
```

Then verify:

```bash
gcc --version
g++ --version
```

Observed:

```text
gcc 13.3.0
g++ 13.3.0
```

`build-essential` installs commonly needed tools such as:

- `gcc`
- `g++`
- `make`
- C/C++ development libraries

---

## 7. Verify NVIDIA GPU Access from WSL

Run:

```bash
nvidia-smi
```

Observed setup:

```text
GPU: NVIDIA GeForce RTX 4080 Laptop GPU
Driver Version: 560.94
CUDA Version: 12.6
VRAM: approximately 12 GB
```

Important:

> The `CUDA Version` shown by `nvidia-smi` indicates the CUDA version supported by the installed NVIDIA driver. It does **not** mean the CUDA Toolkit is installed inside Ubuntu.

Under WSL 2, the NVIDIA driver is primarily provided by Windows.

Do **not** install an Ubuntu NVIDIA display driver with commands such as:

```bash
sudo apt install nvidia-driver-...
```

unless there is a specific reason and you understand the WSL driver model.

---

## 8. Check CPU Architecture

Run:

```bash
uname -m
```

Observed:

```text
x86_64
```

This determines which Miniforge installer to download.

---

## 9. Install Miniforge

Move to the Linux home directory:

```bash
cd ~
```

Download the Linux x86_64 installer:

```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
```

Check that it downloaded:

```bash
ls
```

Expected file:

```text
Miniforge3-Linux-x86_64.sh
```

Run the installer:

```bash
bash Miniforge3-Linux-x86_64.sh
```

During installation:

1. Read/scroll through the license.
2. Type `yes` when asked to accept it.
3. Accept the default install location:

```text
~/miniforge3
```

4. When asked whether to initialize the shell, answer:

```text
yes
```

The installer modifies:

```text
~/.bashrc
```

For this setup, Miniforge was installed at:

```text
~/miniforge3
```

---

## 10. Reload the Shell Configuration

After installation:

```bash
source ~/.bashrc
```

Check Mamba:

```bash
mamba --version
```

Observed:

```text
2.9.0
```

Check Conda:

```bash
conda --version
```

Observed:

```text
conda 26.7.2
```

Check which Python is active:

```bash
which python
```

In the Miniforge base environment:

```text
~/miniforge3/bin/python
```

The prompt may now begin with:

```text
(base)
```

---

## 11. Create a Dedicated LLM Python Environment

Do not install project packages directly into the Ubuntu system Python or Miniforge `base`.

Create a dedicated environment:

```bash
mamba create -n llm python=3.12 pip -y
```

Activate it:

```bash
mamba activate llm
```

The prompt should now begin with:

```text
(llm)
```

Verify:

```bash
which python
python --version
which pip
```

Expected structure:

```text
~/miniforge3/envs/llm/bin/python
Python 3.12.x
~/miniforge3/envs/llm/bin/pip
```

Conceptually:

```text
Ubuntu
|
|-- system Python
|   `-- /usr/bin/python3
|
`-- Miniforge
    |-- base environment
    |
    `-- llm environment
        |-- Python 3.12
        `-- project packages
```

---

## 12. Install GPU-Enabled PyTorch

Make sure the `llm` environment is active:

```text
(llm)
```

Install PyTorch using the CUDA 12.6 wheel index:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
```

Using:

```bash
python -m pip
```

instead of plain:

```bash
pip
```

makes it explicit that packages are being installed for the currently active Python interpreter.

---

## 13. Verify PyTorch

Check the installed PyTorch version:

```bash
python -c "import torch; print(torch.__version__)"
```

Check whether PyTorch can access CUDA:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Expected:

```text
True
```

---

## 14. Create a Tensor on the GPU

Run:

```bash
python -c "import torch; x=torch.tensor([1.0,2.0,3.0], device='cuda'); print(x); print(x.device)"
```

Expected output resembles:

```text
tensor([1., 2., 3.], device='cuda:0')
cuda:0
```

This verifies that PyTorch actually created a tensor in GPU memory.

Check the GPU name from PyTorch:

```bash
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

Expected:

```text
NVIDIA GeForce RTX 4080 Laptop GPU
```

At this point the full path is working:

```text
Python
  |
  v
PyTorch
  |
  v
CUDA runtime
  |
  v
Windows NVIDIA driver / WSL GPU interface
  |
  v
NVIDIA RTX 4080 Laptop GPU
```

---

# Daily Use

When opening a new Ubuntu terminal, activate the project environment:

```bash
mamba activate llm
```

Confirm the active Python if needed:

```bash
which python
```

For this project it should point to something like:

```text
~/miniforge3/envs/llm/bin/python
```

---

# Useful Commands

Show current directory:

```bash
pwd
```

Go to Linux home:

```bash
cd ~
```

List files:

```bash
ls
```

Show active Conda/Mamba environments:

```bash
conda env list
```

Activate the LLM environment:

```bash
mamba activate llm
```

Leave the current environment:

```bash
mamba deactivate
```

Check GPU:

```bash
nvidia-smi
```

Check CUDA availability in PyTorch:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Check PyTorch GPU name:

```bash
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

---


# Disk Usage and Safe Cleanup

A GPU-enabled PyTorch environment can take several gigabytes because it includes PyTorch plus NVIDIA CUDA-related libraries such as cuDNN, cuBLAS, and other runtime components.

For this setup, the initial disk usage was approximately:

```text
Miniforge total:          ~7.2 GB
llm environment:          ~6.6 GB
pip cache:                ~1.1 GB
Miniforge package cache:  ~0.7 GB
Ubuntu/system files:      remaining usage
```

The `llm` environment itself should generally be kept. Most of its size is real installed software required for GPU-enabled PyTorch.

## Check Disk Usage

Check total Miniforge size:

```bash
du -sh ~/miniforge3
```

Check only the `llm` environment:

```bash
du -sh ~/miniforge3/envs/llm
```

Check pip's download cache:

```bash
du -sh ~/.cache/pip 2>/dev/null
```

Check Ubuntu's APT package cache:

```bash
sudo du -sh /var/cache/apt/archives
```

Check Miniforge's package cache:

```bash
du -sh ~/miniforge3/pkgs
```

Check total Linux filesystem usage:

```bash
df -h /
```

---

## Safely Clear the pip Cache

After packages have been installed, pip may keep downloaded installation files in its cache.

To remove them safely:

```bash
python -m pip cache purge
```

This does **not** uninstall PyTorch or other installed packages. It only removes cached download files.

Verify afterward:

```bash
du -sh ~/.cache/pip
```

In this setup, the pip cache dropped from approximately:

```text
1.1 GB
```

to:

```text
4.0 KB
```

---

## Safely Clear Miniforge/Mamba Package Caches

Miniforge also keeps package archives and other cached package data.

Check the cache size:

```bash
du -sh ~/miniforge3/pkgs
```

To clean unused caches safely:

```bash
mamba clean --all -y
```

This removes items such as:

- downloaded package archives
- unused cached packages
- index caches
- temporary package files

It should **not** remove packages currently installed in the active `llm` environment.

Avoid more aggressive cleanup options such as:

```bash
mamba clean --force-pkgs-dirs
```

unless you specifically know why they are needed, because aggressive removal of package directories can interfere with environments.

---

## What Not to Delete

Do not manually delete:

```text
~/miniforge3/envs/llm
```

unless you intentionally want to remove the entire LLM Python environment.

That directory contains the actual installed Python, PyTorch, and CUDA-related libraries.

For this setup it was approximately:

```text
6.6 GB
```

which is reasonable for a GPU-enabled PyTorch environment.

---

## Important WSL Disk-Space Note

WSL 2 stores the Linux filesystem inside a dynamically growing virtual disk file (`ext4.vhdx`) on Windows.

Deleting files inside Ubuntu frees space **inside the Linux filesystem**, but Windows may not immediately show the same amount of space returned to the C: drive because the virtual disk file does not always shrink automatically.

Therefore:

```text
Deleting Linux cache files
        ↓
Linux reports more free space
        ↓
Windows C: may not immediately shrink by the same amount
```

If reclaiming Windows-side C: drive space becomes necessary, the WSL virtual disk can be compacted separately after Linux-side cleanup.

---

# Storage Plan for LLM Work

Recommended long-term layout:

```text
Internal drive / WSL Linux filesystem
|
|-- Ubuntu / WSL
|-- Miniforge
|-- Python environments
|-- Git repositories
`-- source code

4 TB external SSD
|
|-- datasets
|-- pretrained model weights
|-- checkpoints
|-- Hugging Face caches
|-- experiment outputs
`-- research artifacts
```

Keep code and environments on the fast Linux filesystem unless there is a reason to move them.

Use the external SSD for large files that would otherwise fill the internal drive.

---

# Important Rules

1. Prefer working under `/home/<username>` rather than `/mnt/c/...` for Linux development.
2. Do not install project packages into Ubuntu's system Python.
3. Do not put project packages into the Miniforge `base` environment.
4. Use the dedicated `llm` environment.
5. Prefer `python -m pip` over plain `pip`.
6. Do not install a separate Linux NVIDIA display driver inside WSL unless specifically required.
7. `nvidia-smi` showing a CUDA version does not mean the full CUDA Toolkit is installed.
8. Verify GPU access with both `nvidia-smi` and PyTorch.
9. Keep large datasets and checkpoints on the external SSD.
10. Record major environment changes in this guide so the setup is reproducible.

---

# Working Environment Snapshot

As of the initial setup:

```text
WSL:                2.7.14
Ubuntu:             24.04.5 LTS
Ubuntu codename:    noble
Architecture:       x86_64
System Python:      3.12.3
Git:                2.43.0
GCC:                13.3.0
G++:                13.3.0
Mamba:              2.9.0
Conda:              26.7.2
LLM Python:         3.12.x
GPU:                NVIDIA GeForce RTX 4080 Laptop GPU
GPU VRAM:           ~12 GB
NVIDIA driver:      560.94
Driver CUDA support: 12.6
```

