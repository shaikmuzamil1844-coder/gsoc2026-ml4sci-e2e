from __future__ import annotations

import sys
import unittest
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
    inverse_fpr_at_target_tpr,
    normalize_channels,
    stratified_split,
    to_nchw,
)


class SmokeTests(unittest.TestCase):
    def test_task1_pipeline_shapes(self) -> None:
        X, y = generate_task1_synthetic_dataset(samples_per_class=16)
        X = normalize_channels(to_nchw(X))
        split = stratified_split(X, y)

        model = Task1ResNet15(in_channels=X.shape[1])
        logits = model(torch.tensor(split.X_train[:4]))

        self.assertEqual(tuple(logits.shape), (4,))
        self.assertEqual(split.X_train.shape[1:], (2, 32, 32))

    def test_task2_pipeline_shapes(self) -> None:
        X, y = generate_task2_synthetic_dataset(samples_per_class=12)
        X = normalize_channels(to_nchw(np.log1p(X)))
        split = stratified_split(X, y)

        model = Task2ResNet15(in_channels=X.shape[1])
        logits = model(torch.tensor(split.X_train[:2]))

        self.assertEqual(tuple(logits.shape), (2,))
        self.assertEqual(split.X_train.shape[1:], (3, 125, 125))

    def test_metrics_helpers(self) -> None:
        y_true = np.array([0, 0, 1, 1], dtype=np.int64)
        y_score = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float32)

        metrics = binary_classification_metrics(y_true, y_score)
        inv_fpr = inverse_fpr_at_target_tpr(y_true, y_score, target_tpr=0.5)

        self.assertGreaterEqual(metrics["auc"], 0.99)
        self.assertAlmostEqual(metrics["accuracy"], 1.0)
        self.assertTrue(np.isfinite(inv_fpr) or inv_fpr == float("inf"))


if __name__ == "__main__":
    unittest.main()
