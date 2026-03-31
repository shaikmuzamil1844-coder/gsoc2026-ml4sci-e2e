# 🔬 GSoC 2026 — ML4SCI E2E | Sparse Neural Network Pipeline

**Organization:** [ML4SCI](https://ml4sci.org/)  
**Project:** Sparse Neural Network Pipeline for Particle Collision Event Classification (E2E)  
**Author:** Shaik Muzamil | [@shaikmuzamil1844-coder](https://github.com/shaikmuzamil1844-coder)

---

## 📁 Repository Structure

```
gsoc2026-ml4sci-e2e/
├── modified_Task1_Electron_Photon.ipynb        # Task 1: Electron vs Photon Classification
├── Task2_sparse_neural_network.ipynb           # Task 2: Quark vs Gluon Classification
└── README.md                                   # This file
```

---

# 🎯 Task 1: Electron vs Photon Classification

## Task Overview

Binary classification of **electron** vs **photon** events from 32×32 calorimeter detector images, each with 2 channels:

| Channel | Description |
|---------|-------------|
| Channel 1 | Hit Energy |
| Channel 2 | Hit Time |

**Goal:** Maximize AUC score on the held-out test set.

---

## 🏗️ Model Architecture — ResNet-15

A lightweight custom ResNet inspired by [Andrews et al. (2020)](https://arxiv.org/abs/1807.11916), designed to handle small 32×32 particle physics images.

```
Input (2, 32, 32)
     │
  Stem Conv (32 filters)
     │
  Layer 1 — ResBlock: 32 → 64
  Layer 2 — ResBlock: 64 → 128  [stride 2]
  Layer 3 — ResBlock: 128 → 256 [stride 2]
  Layer 4 — ResBlock: 256 → 256 [stride 2]
     │
  Global Avg Pool → Dropout(0.3) → FC(1)
     │
  BCEWithLogitsLoss
```

- **Total Parameters:** ~1.2M
- **Input channels:** 2 (Energy + Time)
- **Output:** Binary (Electron / Photon)

---

## 📊 Results

| Metric | Value |
|--------|-------|
| **Test AUC** | **0.7178** |
| Test Accuracy | ~66% |
| Model | ResNet-15 |
| Epochs | 25 |
| Optimizer | Adam (lr=5e-4, wd=1e-4) |
| Scheduler | CosineAnnealingLR |
| Train Samples | 24,000 |
| Test Samples | 6,000 |

---

## 📁 Dataset

**Source:** [Kaggle — Electron vs Photons (ML4SCI)](https://www.kaggle.com/datasets/vishakkbhat/electron-vs-photons-ml4sci)

| File | Samples |
|------|---------|
| `SingleElectronPt50_IMGCROPS_n249k_RHv1.hdf5` | 249,000 |
| `SinglePhotonPt50_IMGCROPS_n249k_RHv1.hdf5` | 249,000 |

> This task uses 15,000 samples per class (30,000 total) for training efficiency.

---

## ⚙️ Setup & Usage

### 1. Clone the repo
```bash
git clone https://github.com/shaikmuzamil1844-coder/gsoc2026-ml4sci-e2e.git
cd gsoc2026-ml4sci-e2e
```

### 2. Install dependencies
```bash
pip install numpy h5py torch torchvision scikit-learn matplotlib seaborn
```

### 3. Download dataset
Set your Kaggle credentials as environment variables (do NOT hardcode them):
```bash
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
```
Then run the download cell in the notebook, or:
```bash
kaggle datasets download -d vishakkbhat/electron-vs-photons-ml4sci
unzip electron-vs-photons-ml4sci.zip
```

### 4. Run the notebook
Open `modified_Task1_Electron_Photon.ipynb` in Jupyter or Google Colab and run all cells.

> **Google Colab:** Recommended for GPU access (T4 or better).

---

## 🔑 Key Design Choices

- **2-channel input** — Energy and Time channels used together; no channel dropped
- **Residual connections** — Help with gradient flow on small 32×32 images
- **CosineAnnealingLR** — Smooth LR decay over 25 epochs, avoids sharp drops
- **BCEWithLogitsLoss** — Numerically stable binary cross-entropy
- **Dropout(0.3)** — Applied before final FC layer to reduce overfitting

---

---

# 🎯 Task 2: Quark vs Gluon Jet Classification

## Task Description

End-to-End deep learning pipeline to classify **quark-initiated** vs **gluon-initiated** jets using 125×125 multi-channel CMS detector images.

**Input Channels:**

| Channel | Description |
|---------|-------------|
| Channel 1 | ECAL (Electromagnetic Calorimeter) |
| Channel 2 | HCAL (Hadronic Calorimeter) |
| Channel 3 | Reconstructed Tracks |

**Reference:** Andrews et al., *End-to-End Jet Classification of Quarks and Gluons with the CMS Open Data* ([arXiv:1902.08276](https://arxiv.org/abs/1902.08276))  
**Goal:** Maximize AUC score on the test set using the ResNet-15 architecture from the paper.

---

## 🏗️ Model Architecture — ResNet-15

```
Input (3 × 125 × 125)
    ↓
Stem: Conv2d(3→64, 3×3) + BN + ReLU
    ↓
Layer 1: ResBlock(64→64) × 2
    ↓
Layer 2: ResBlock(64→128, stride=2) + ResBlock(128→128)
    ↓
Layer 3: ResBlock(128→256, stride=2) + ResBlock(256→256)
    ↓
Layer 4: ResBlock(256→512, stride=2) + ResBlock(512→512)
    ↓
Global Average Pooling → Dropout(0.3) → Linear(512→1)
    ↓
Output: Binary classification (Quark=1 / Gluon=0)
```

- **Total Parameters:** ~11 million
- **Input channels:** 3 (ECAL + HCAL + Tracks)
- **Output:** Binary (Quark / Gluon)

---

## 📊 Results

| Metric | Andrews et al. Baseline | This Model |
|--------|------------------------|------------|
| **Test AUC** | 0.8076 | *see notebook output* |
| **1/FPR @ TPR=70%** | 4.47 | *see notebook output* |

---

## 📁 Dataset

**Source:** ML4SCI CERNBox — auto-downloaded in notebook.  
If download fails, the notebook **automatically generates synthetic jet images** so training can still proceed.

---

## ⚙️ Training Configuration

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 5e-4 |
| LR Schedule | StepLR (halved every 10 epochs) |
| Weight Decay | 1e-4 |
| Epochs | 30 |
| Batch Size | 128 |
| Loss Function | BCEWithLogitsLoss |
| Gradient Clipping | max norm = 1.0 |

---

## 🔧 Data Preprocessing

1. **log(1+x) transform** — compresses large dynamic range of energy deposits
2. **Transpose** — converts (N, H, W, C) → PyTorch format (N, C, H, W)
3. **Per-channel z-score normalization** — zero mean, unit variance per channel
4. **80/10/10 stratified split** — train / validation / test

---

## 💡 Key Finding — Sparsity

The CMS detector images are **~90%+ sparse (zero pixels)**. This directly motivates the GSoC 2026 proposal:

- Dense CNN wastes ~90% of compute on empty pixels
- Sparse convolutions skip zero pixels entirely
- Expected speedup: **2–5× fewer FLOPs** with same AUC

---

## 🚀 How to Run (Task 2)

> **Important:** Always run cells **top to bottom in order**.

1. Open `Task2_sparse_neural_network.ipynb` in Google Colab
2. Go to **Runtime → Change runtime type → T4 GPU**
3. Click **Runtime → Run All**

```bash
pip install torch torchvision numpy h5py scikit-learn matplotlib seaborn
```

---

---

## 🔗 References

- Andrews et al. (2020) — [arXiv:1902.08276](https://arxiv.org/abs/1902.08276)
- Andrews et al. (2020) — [arXiv:1807.11916](https://arxiv.org/abs/1807.11916)
- He et al., "Deep Residual Learning" (2015) — [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
- ML4SCI E2E Project — [https://ml4sci.org](https://ml4sci.org)
- GSoC 2026 — ML4SCI Organization

---

## 👤 Author

**Shaik Muzamil**  
B.Tech AI/ML — Dhanalakshmi Srinivasan University (2023–2027)  
GitHub: [@shaikmuzamil1844-coder](https://github.com/shaikmuzamil1844-coder)

---

*Submitted as part of GSoC 2026 application to ML4SCI.*
