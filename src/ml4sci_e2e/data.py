from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class SplitData:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray


def generate_task1_synthetic_dataset(
    samples_per_class: int = 256,
    image_size: int = 32,
    channels: int = 2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a small synthetic electron/photon dataset for smoke tests."""
    rng = np.random.default_rng(seed)

    def _make_class(label: int) -> np.ndarray:
        images = np.zeros((samples_per_class, image_size, image_size, channels), dtype=np.float32)
        center = image_size // 2
        spread = 3.2 if label == 1 else 5.5
        energy_scale = 1.2 if label == 1 else 0.9
        time_mean = 0.4 if label == 1 else 0.7
        hits_mean = 28 if label == 1 else 20

        for i in range(samples_per_class):
            hits = rng.poisson(hits_mean)
            eta = np.clip(rng.normal(center, spread, hits).astype(int), 0, image_size - 1)
            phi = np.clip(rng.normal(center, spread, hits).astype(int), 0, image_size - 1)
            energy = rng.exponential(energy_scale, hits).astype(np.float32)
            timing = np.abs(rng.normal(time_mean, 0.12, hits)).astype(np.float32)
            np.add.at(images[i, :, :, 0], (eta, phi), energy)
            np.add.at(images[i, :, :, 1], (eta, phi), timing)
        return images

    X_photon = _make_class(label=0)
    X_electron = _make_class(label=1)
    y = np.concatenate(
        [
            np.zeros(samples_per_class, dtype=np.int64),
            np.ones(samples_per_class, dtype=np.int64),
        ]
    )
    X = np.concatenate([X_photon, X_electron], axis=0)
    return X, y


def generate_task2_synthetic_dataset(
    samples_per_class: int = 128,
    image_size: int = 125,
    channels: int = 3,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a sparse quark/gluon-style synthetic jet dataset."""
    rng = np.random.default_rng(seed)
    center = image_size // 2

    def _make_class(sigma: float, particles_mean: int) -> np.ndarray:
        images = np.zeros((samples_per_class, image_size, image_size, channels), dtype=np.float32)
        for i in range(samples_per_class):
            hits = rng.poisson(particles_mean)
            eta = np.clip(rng.normal(center, sigma, hits).astype(int), 0, image_size - 1)
            phi = np.clip(rng.normal(center, sigma, hits).astype(int), 0, image_size - 1)
            energy = np.abs(rng.exponential(1.0, hits)).astype(np.float32)
            np.add.at(images[i, :, :, 0], (eta, phi), energy)
            np.add.at(images[i, :, :, 1], (eta, phi), energy * 0.3)
            np.add.at(images[i, :, :, 2], (eta, phi), 1.0)
        return images

    X_gluon = _make_class(sigma=8.0, particles_mean=30)
    X_quark = _make_class(sigma=5.0, particles_mean=20)
    y = np.concatenate(
        [
            np.zeros(samples_per_class, dtype=np.int64),
            np.ones(samples_per_class, dtype=np.int64),
        ]
    )
    X = np.concatenate([X_gluon, X_quark], axis=0)
    return X, y


def to_nchw(X: np.ndarray) -> np.ndarray:
    """Convert an NHWC image tensor to NCHW."""
    if X.ndim != 4:
        raise ValueError(f"Expected a 4D tensor in NHWC format, got shape {X.shape}.")
    return np.transpose(X, (0, 3, 1, 2)).astype(np.float32)


def normalize_channels(X: np.ndarray, log1p: bool = False) -> np.ndarray:
    """Apply optional log transform and per-channel z-score normalization."""
    if X.ndim != 4:
        raise ValueError(f"Expected a 4D tensor, got shape {X.shape}.")

    X_out = np.log1p(X) if log1p else X.copy()
    for channel_idx in range(X_out.shape[1]):
        channel = X_out[:, channel_idx, :, :]
        mean = float(channel.mean())
        std = float(channel.std()) + 1e-8
        X_out[:, channel_idx, :, :] = (channel - mean) / std
    return X_out.astype(np.float32)


def stratified_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    val_size_within_temp: float = 0.5,
    random_state: int = 42,
) -> SplitData:
    """Create train/val/test splits with the same strategy used in the notebooks."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=val_size_within_temp,
        random_state=random_state,
        stratify=y_temp,
    )
    return SplitData(X_train, X_val, X_test, y_train, y_val, y_test)
