"""Shared utility functions for Blood Mahjong."""

import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax. Supports batched input (last axis)."""
    x_max = x.max(axis=-1, keepdims=True)
    e = np.exp(x - x_max)
    return e / e.sum(axis=-1, keepdims=True)
