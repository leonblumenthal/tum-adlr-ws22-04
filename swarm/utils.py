"""Utility functions."""

import numpy as np


def normalize(vectors: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Normalize row vectors if their norm is >0."""

    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    norms += norms < eps
    return vectors / norms


def limit(vectors: np.ndarray, max_norm: float) -> np.ndarray:
    """Rescale row vectors if their norm is greater than `max_norm`."""

    clipped_norms = np.linalg.norm(vectors, axis=-1, keepdims=True).clip(max_norm)
    return vectors * max_norm / clipped_norms
