"""Shared utility functions for Blood Mahjong."""

import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = x.max()
    e = np.exp(x - x_max)
    return e / e.sum()
