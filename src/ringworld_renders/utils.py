"""
Utility functions for Ringworld Renderer.

This module provides common utilities for batch processing, array manipulation,
and other shared functionality.
"""

import numpy as np


def ensure_batch(arrays, expand_dims=None):
    """
    Ensure arrays are in batch format.

    This function handles batch conversion by detecting single inputs:
    - Single vectors (ndim=1, small size) get expanded to batch format
    - Arrays already in batch format are left unchanged

    Args:
        arrays: Single array or list/tuple of arrays to batch
        expand_dims: List of axis positions to expand for single arrays.
                    If None, expands dimension 0.

    Returns:
        Batched arrays (same type as input - single array or tuple/list)
    """
    def _should_batch(arr):
        """Check if an array represents single (non-batch) data that needs batching."""
        if not hasattr(arr, 'ndim'):
            return False
        # Only batch true single vectors, not small batches
        # Single vector: ndim=1 and shape indicates a single 3D vector
        return arr.ndim == 1 and arr.shape[0] == 3

    if isinstance(arrays, (list, tuple)):
        # Multiple arrays - batch each that needs it
        result = []
        for arr in arrays:
            if _should_batch(arr):
                if expand_dims is not None:
                    for axis in sorted(expand_dims, reverse=True):
                        arr = np.expand_dims(arr, axis=axis)
                else:
                    arr = arr[None, :]
            result.append(arr)
        return type(arrays)(result)
    else:
        # Single array
        arr = arrays
        if _should_batch(arr):
            if expand_dims is not None:
                for axis in sorted(expand_dims, reverse=True):
                    arr = np.expand_dims(arr, axis=axis)
            else:
                arr = arr[None, :]
        return arr


def unbatch_if_needed(arrays, was_single):
    """
    Convert batched arrays back to single arrays if they were originally single.

    Args:
        arrays: Single array or list/tuple of arrays to potentially unbatch
        was_single: Boolean indicating if original input was single (not batched)

    Returns:
        Arrays in original format (single or batched)

    Examples:
        # Single array that was originally single
        direction = unbatch_if_needed(directions_batch, was_single=True)  # -> shape (3,)

        # Multiple arrays
        origins, directions = unbatch_if_needed([origins_batch, directions_batch], was_single)
    """
    if not was_single:
        return arrays

    if isinstance(arrays, (list, tuple)):
        # Multiple arrays - unbatch each
        result = []
        for arr in arrays:
            if arr.shape[0] == 1:
                # Remove batch dimension
                arr = arr[0]
            result.append(arr)
        return type(arrays)(result)
    else:
        # Single array
        arr = arrays
        if arr.shape[0] == 1:
            # Remove batch dimension
            arr = arr[0]
        return arr


def detect_single_input(*arrays):
    """
    Detect if input arrays represent single (non-batched) data.

    Args:
        *arrays: Variable number of arrays to check

    Returns:
        bool: True if any array has ndim == 1 (indicating single input)
    """
    return any(arr.ndim == 1 for arr in arrays if hasattr(arr, 'ndim'))


def batch_compatible(func):
    """
    Decorator to make functions batch-compatible.

    Automatically handles conversion between single and batch formats
    at function boundaries.

    Usage:
        @batch_compatible
        def some_function(single_or_batch_input):
            # Function always receives batch format
            return batch_result
        # Function automatically returns single format if input was single
    """
    def wrapper(*args, **kwargs):
        # Detect if input was single
        was_single = detect_single_input(*args)

        # Ensure all array inputs are batched
        batched_args = []
        for arg in args:
            if hasattr(arg, 'ndim'):
                batched_args.append(ensure_batch(arg))
            else:
                batched_args.append(arg)

        # Call function with batched inputs
        result = func(*batched_args, **kwargs)

        # Convert back to single format if needed
        return unbatch_if_needed(result, was_single)

    return wrapper
