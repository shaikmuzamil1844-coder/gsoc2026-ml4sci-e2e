# GSoC 2026 - ML4SCI End-to-End Baselines

Dense deep-learning baselines for particle-physics event classification, prepared as a stronger starting point for a GSoC proposal around sparse neural networks.

This repository focuses on two image-based tasks that align well with the ML4SCI end-to-end project direction:

- Task 1: electron vs photon classification from calorimeter images
- Task 2: quark vs gluon jet classification from sparse detector images

The repo is intentionally positioned as a dense-baseline research scaffold:

- notebooks for exploration and presentation
- reusable Python modules for cleaner engineering
- smoke tests and validation scripts for reproducibility
- documentation that separates real benchmark claims from local fallback runs

## Project Pitch

Modern detector images are often extremely sparse. Dense CNNs still work well as baselines, but they waste substantial computation on zero-valued regions. That makes this repository a useful pre-sparse starting point:

- it establishes dense ResNet-style baselines
- it highlights where sparsity matters most
- it makes the transition to sparse methods easier to justify experimentally

For a GSoC mentor, this repo should read as:

- a runnable baseline
- a reproducible engineering artifact
- a launchpad for dense-vs-sparse comparisons

## What Improved In This Version

Compared with the earlier notebook-only state, this repo now has:

- portable notebooks that no longer depend on Colab-only shell magic
- no hardcoded Kaggle secrets
- synthetic fallback data paths for local execution validation
- CPU-safe settings so `Run All` can complete on a normal machine
- reusable code in `src/ml4sci_e2e`
- a repository validation script in `scripts/validate_repo.py`
- smoke tests in `tests/test_smoke.py`
- clearer GSoC-facing documentation in `docs/results.md` and `docs/roadmap.md`

## Repository Layout

```text
gsoc2026-ml4sci-e2e/
|-- docs/
|   |-- results.md
|   `-- roadmap.md
|-- scripts/
|   `-- validate_repo.py
|-- src/
|   `-- ml4sci_e2e/
|       |-- data.py
|       |-- metrics.py
|       `-- models.py
|-- tests/
|   `-- test_smoke.py
|-- tools/
|   |-- patch_notebooks.py
|   `-- run_notebook.py
|-- modified_Task1_Electron_Photon_Classification.ipynb
|-- Task2_sparse_neural_network.ipynb
|-- pyproject.toml
|-- requirements.txt
`-- README.md
```

## Tasks

### Task 1: Electron vs Photon

Problem:

- binary classification from `32 x 32 x 2` calorimeter images
- channels represent hit energy and hit time

Model:

- lightweight ResNet-15 style architecture
- residual blocks for stable training on small images
- global average pooling and dropout before the binary head

Primary dataset:

- Kaggle: Electron vs Photons (ML4SCI)

Reference result already reported in the original notebook work:

- test AUC: `0.7178`

### Task 2: Quark vs Gluon

Problem:

- binary classification from `125 x 125 x 3` detector images
- channels represent ECAL, HCAL, and reconstructed tracks

Model:

- deeper ResNet-15 style architecture
- designed as a dense baseline for future sparse experiments

Primary dataset:

- ML4SCI CERNBox source referenced in the notebook

Paper baseline used as the target reference:

- AUC: `0.8076`
- `1/FPR @ TPR=70%`: `4.47`

### Why Task 2 Is Especially Relevant

Task 2 exposes the core motivation for a sparse-ML GSoC proposal:

- detector images are overwhelmingly sparse
- dense convolutions waste FLOPs on inactive pixels
- sparse operators could preserve accuracy while improving efficiency

## Installation

### 1. Clone

```bash
git clone https://github.com/shaikmuzamil1844-coder/gsoc2026-ml4sci-e2e.git
cd gsoc2026-ml4sci-e2e
```

### 2. Create and activate an environment

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
python -m pip install -e .
```

## How To Run

### Notebook execution

```bash
python tools/run_notebook.py modified_Task1_Electron_Photon_Classification.ipynb
python tools/run_notebook.py Task2_sparse_neural_network.ipynb
```

### Repository validation

This is the fastest way to show that the repo is wired correctly:

```bash
python scripts/validate_repo.py
python -m unittest tests.test_smoke
```

## Real Results vs Fallback Runs

This distinction matters for credibility.

### Real benchmark runs

Use these for:

- proposal claims
- mentor discussions
- scientific comparison

Characteristics:

- real datasets
- preferably GPU-backed training
- meaningful reported performance

### Fallback validation runs

Use these for:

- local reproducibility
- notebook portability
- CI-style smoke checks

Characteristics:

- synthetic fallback data when external data is unavailable
- reduced CPU-safe settings
- useful for execution validation, not final scientific claims

Read more in [docs/results.md](docs/results.md).

## Why This Repo Is More Mentor-Friendly Now

Mentors usually want three signals:

1. You can frame the research problem clearly.
2. You can engineer a clean starting point.
3. You understand the difference between a demo and a benchmark.

This repo now addresses all three more directly:

- the notebooks are easier to run
- the code is no longer trapped entirely inside notebooks
- the docs explain what is baseline engineering vs what still needs real experimentation

## Suggested Next GSoC Step

The strongest follow-up would be to build the sparse-method comparison on top of this baseline:

1. profile dense training and inference
2. introduce sparse layers or sparse representations
3. compare accuracy, memory, throughput, and compute
4. document where sparse methods win and where they do not

That roadmap is summarized in [docs/roadmap.md](docs/roadmap.md).

## References

- [Andrews et al. - End-to-End Jet Classification of Quarks and Gluons with the CMS Open Data](https://arxiv.org/abs/1902.08276)
- [Andrews et al. - Particle Classification with Calorimeter Images](https://arxiv.org/abs/1807.11916)
- [He et al. - Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- [ML4SCI](https://ml4sci.org/)

## Author

Shaik Muzamil  
B.Tech AI/ML - Dhanalakshmi Srinivasan University  
GitHub: [@shaikmuzamil1844-coder](https://github.com/shaikmuzamil1844-coder)
