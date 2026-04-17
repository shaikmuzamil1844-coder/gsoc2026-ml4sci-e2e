from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml4sci_e2e import (  # noqa: E402
    Task1ResNet15,
    Task2ResNet15,
    binary_classification_metrics,
    generate_task1_synthetic_dataset,
    generate_task2_synthetic_dataset,
    normalize_channels,
    stratified_split,
    to_nchw,
)


def run_task1_smoke() -> None:
    X, y = generate_task1_synthetic_dataset(samples_per_class=32)
    X = normalize_channels(to_nchw(X), log1p=False)
    split = stratified_split(X, y)

    model = Task1ResNet15(in_channels=X.shape[1])
    batch = torch.tensor(split.X_train[:8])
    logits = model(batch)
    metrics = binary_classification_metrics(
        np.array([0, 0, 1, 1], dtype=np.int64),
        np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float32),
    )

    print("[task1] train/val/test:", split.X_train.shape[0], split.X_val.shape[0], split.X_test.shape[0])
    print("[task1] logits shape:", tuple(logits.shape))
    print("[task1] sample metrics:", {k: v for k, v in metrics.items() if k != "confusion_matrix"})


def run_task2_smoke() -> None:
    X, y = generate_task2_synthetic_dataset(samples_per_class=24)
    raw_sparsity = float((X == 0).mean() * 100)
    X = normalize_channels(to_nchw(np.log1p(X)), log1p=False)
    split = stratified_split(X, y)

    model = Task2ResNet15(in_channels=X.shape[1])
    batch = torch.tensor(split.X_train[:4])
    logits = model(batch)
    metrics = binary_classification_metrics(
        np.array([0, 0, 1, 1], dtype=np.int64),
        np.array([0.1, 0.3, 0.7, 0.95], dtype=np.float32),
    )

    print("[task2] train/val/test:", split.X_train.shape[0], split.X_val.shape[0], split.X_test.shape[0])
    print("[task2] logits shape:", tuple(logits.shape))
    print("[task2] synthetic sparsity: %.2f%%" % raw_sparsity)
    print("[task2] sample metrics:", {k: v for k, v in metrics.items() if k != "confusion_matrix"})


def main() -> int:
    torch.manual_seed(42)
    np.random.seed(42)

    print("Running repository smoke validation...")
    run_task1_smoke()
    run_task2_smoke()
    print("Validation complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
