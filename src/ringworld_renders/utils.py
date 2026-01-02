"""
Batch processing utilities for the Ringworld renderer.

This module contains general-purpose utility functions used throughout
the rendering pipeline, particularly for batch processing operations.
"""

import numpy as np


def ensure_batch(array, ndim=2):
    """
    Ensure array has at least the specified number of dimensions.

    Args:
        array: Input array
        ndim: Minimum number of dimensions required

    Returns:
        Tuple of (array, was_single) where was_single indicates if input was 1D
    """
    was_single = array.ndim == (ndim - 1)
    if was_single:
        array = array[None, :]
    return array, was_single


def restore_single_if_needed(array, was_single):
    """
    Restore single-element result if input was single.

    Args:
        array: Result array
        was_single: Whether input was originally single

    Returns:
        Single element if was_single, otherwise full array
    """
    if was_single:
        return array[0] if array.size > 0 else array
    return array


def vectorized_min(a, b):
    """
    Element-wise minimum of two arrays, handling inf values properly.

    Args:
        a, b: Arrays to compare

    Returns:
        Array of element-wise minimums
    """
    return np.where(np.isinf(a), b, np.where(np.isinf(b), a, np.minimum(a, b)))
