"""
Pytest fixtures and configuration for Ringworld Renderer tests.

This module provides shared fixtures and utilities to reduce test code duplication
and improve test organization.
"""

import numpy as np
import pytest
from ringworld_renders.core import Renderer


@pytest.fixture
def renderer():
    """Create a standard renderer instance for tests."""
    return Renderer()


@pytest.fixture
def origin():
    """Standard ray origin at observer position."""
    return np.array([0.0, 0.0, 0.0])


@pytest.fixture
def standard_rays():
    """Common ray directions used in multiple tests."""
    return {
        'ground': np.array([0.0, -1.0, 0.0]),  # Straight down to ground
        'zenith': np.array([0.0, 1.0, 0.0]),   # Straight up to zenith/arch
        'horizon': np.array([1.0, 0.0, 0.0]),  # Horizontal (spinward)
        'east': np.array([0.0, 0.0, 1.0]),      # Axial east
        'west': np.array([0.0, 0.0, -1.0]),     # Axial west
    }


@pytest.fixture
def batch_rays():
    """Batch of common ray directions for performance testing."""
    return np.array([
        [0.0, -1.0, 0.0],  # Ground
        [0.0, 1.0, 0.0],   # Zenith
        [1.0, 0.0, 0.0],   # Horizon
        [0.0, 0.0, 1.0],   # East
        [0.0, 0.0, -1.0],  # West
    ])


@pytest.fixture
def time_values():
    """Common time values for testing different times of day."""
    solar_day = 24.0 * 3600.0
    return {
        'noon': 0.0,
        'sunset': 6.0 * 3600.0,
        'midnight': 12.0 * 3600.0,
        'sunrise': 18.0 * 3600.0,
        'quarter_day': solar_day / 4.0,
    }


@pytest.fixture
def expected_colors():
    """Expected color values for common test cases."""
    return {
        'sun': np.array([1.0, 1.0, 0.8]),
        'ring_noon': np.array([0.2, 0.5, 0.2]) * 1.02,  # With ring-shine
        'ring_midnight': np.array([0.2, 0.5, 0.2]) * (1.0 + 0.1),  # With ring-shine
        'wall': np.array([0.1, 0.1, 0.15]),
        'shadow_square_debug': None,  # Depends on position
        'sky': np.array([0.0, 0.0, 0.0]),  # No surface color
    }


def assert_color_close(actual, expected, rtol=1e-6, atol=1e-6, err_msg=""):
    """Assert that two colors are close, with helpful error messages."""
    np.testing.assert_allclose(
        actual, expected, rtol=rtol, atol=atol,
        err_msg=f"Color mismatch: {err_msg}"
    )


def assert_color_in_range(color, min_val=0.0, max_val=1.0, err_msg=""):
    """Assert that all color components are within valid range."""
    assert np.all(color >= min_val), f"Color below minimum {min_val}: {color} - {err_msg}"
    assert np.all(color <= max_val), f"Color above maximum {max_val}: {color} - {err_msg}"


def create_test_image(renderer, width=32, height=32, **render_kwargs):
    """Create a small test image with specified parameters."""
    return renderer.render(width=width, height=height, **render_kwargs)
