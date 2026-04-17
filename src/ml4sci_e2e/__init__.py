"""Utilities for the ML4SCI end-to-end notebook baselines."""

from .data import (
    generate_task1_synthetic_dataset,
    generate_task2_synthetic_dataset,
    normalize_channels,
    stratified_split,
    to_nchw,
)
from .metrics import binary_classification_metrics, inverse_fpr_at_target_tpr
from .models import Task1ResNet15, Task2ResNet15

__all__ = [
    "Task1ResNet15",
    "Task2ResNet15",
    "binary_classification_metrics",
    "generate_task1_synthetic_dataset",
    "generate_task2_synthetic_dataset",
    "inverse_fpr_at_target_tpr",
    "normalize_channels",
    "stratified_split",
    "to_nchw",
]
