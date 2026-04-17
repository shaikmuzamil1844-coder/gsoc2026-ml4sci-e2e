# GSoC 2026 - ML4SCI E2E

End-to-end deep learning notebooks for particle collision event classification, built around the ML4SCI GSoC 2026 application tasks.

This repository contains two notebook-based pipelines:

- `modified_Task1_Electron_Photon_Classification.ipynb`
  Electron vs photon classification from `32x32x2` calorimeter images
- `Task2_sparse_neural_network.ipynb`
  Quark vs gluon jet classification from `125x125x3` CMS-style detector images

## Why This Repo Is Easier To Run Now

The notebooks were updated so they are more developer-friendly outside Google Colab:

- no hardcoded Kaggle secret values
- no required `!pip`, `!wget`, `!unzip`, or other Colab-only shell cells
- safer dependency bootstrapping from standard Python
- CPU-safe fallback execution paths for local validation
- synthetic dataset fallback when external data is unavailable
- reproducible local notebook runner in `tools/run_notebook.py`

## Repository Layout

```text
gsoc2026-ml4sci-e2e/
|-- modified_Task1_Electron_Photon_Classification.ipynb
|-- Task2_sparse_neural_network.ipynb
|-- requirements.txt
|-- tools/
|   |-- patch_notebooks.py
|   `-- run_notebook.py
`-- README.md
```

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/shaikmuzamil1844-coder/gsoc2026-ml4sci-e2e.git
cd gsoc2026-ml4sci-e2e
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Run the notebooks

In Jupyter:

```bash
jupyter notebook
```

Or execute a notebook programmatically:

```bash
python tools/run_notebook.py modified_Task1_Electron_Photon_Classification.ipynb
python tools/run_notebook.py Task2_sparse_neural_network.ipynb
```

## Task 1: Electron vs Photon Classification

### Objective

Classify electron and photon events using `32x32` calorimeter images with two channels:

- channel 1: hit energy
- channel 2: hit time

### Model

A lightweight ResNet-style binary classifier:

- input: `2 x 32 x 32`
- residual backbone with progressive channel expansion
- global average pooling
- dropout
- final linear classification head

### Dataset

Primary source:
[Kaggle - Electron vs Photons (ML4SCI)](https://www.kaggle.com/datasets/vishakkbhat/electron-vs-photons-ml4sci)

Expected files:

- `SingleElectronPt50_IMGCROPS_n249k_RHv1.hdf5`
- `SinglePhotonPt50_IMGCROPS_n249k_RHv1.hdf5`

If Kaggle credentials are not available locally, the notebook can generate a small synthetic fallback dataset so the notebook still runs end-to-end for validation.

### Reference Result

From the original project version:

- test AUC: `0.7178`
- model: ResNet-15 style architecture
- optimizer: Adam
- scheduler: CosineAnnealingLR

## Task 2: Quark vs Gluon Jet Classification

### Objective

Classify quark-initiated vs gluon-initiated jets using multi-channel jet images:

- ECAL
- HCAL
- reconstructed tracks

### Model

ResNet-15 style architecture inspired by the CMS open data end-to-end jet classification setup.

### Dataset

Primary source:
ML4SCI CERNBox download referenced in the notebook.

Behavior now:

- tries to use the real dataset when available
- validates the downloaded `.npz` file before use
- falls back to synthetic sparse jet images if download is unavailable

### Key Observation

These detector images are highly sparse, which is exactly why sparse neural network methods are compelling for this problem:

- dense convolutions waste compute on zero-valued pixels
- sparse methods can reduce unnecessary FLOPs
- this repository acts as a strong dense baseline for future sparse experiments

## Reproducibility Notes

- on GPU, notebooks keep the larger training configuration
- on CPU, notebooks automatically use smaller execution settings so `Run All` completes in a normal local environment
- fallback CPU runs are intended for execution validation, not final benchmark reporting

## Developer Notes

If you want to keep iterating on notebook usability:

- use `tools/patch_notebooks.py` to reapply the notebook source fixes
- use `tools/run_notebook.py` for local notebook execution without depending on Jupyter UI behavior
- avoid committing datasets, model checkpoints, and virtual environments

## References

- [Andrews et al. - End-to-End Jet Classification of Quarks and Gluons with the CMS Open Data](https://arxiv.org/abs/1902.08276)
- [Andrews et al. - Particle Classification with Calorimeter Images](https://arxiv.org/abs/1807.11916)
- [He et al. - Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- [ML4SCI](https://ml4sci.org/)

## Author

Shaik Muzamil  
B.Tech AI/ML - Dhanalakshmi Srinivasan University  
GitHub: [@shaikmuzamil1844-coder](https://github.com/shaikmuzamil1844-coder)
