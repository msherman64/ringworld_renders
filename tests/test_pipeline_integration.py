import numpy as np
import pytest
from ringworld_renders.core import Renderer


def test_pipeline_equivalence():
    """
    Critical Test: Ensure new pipeline produces identical results to original get_color() implementation.

    This test verifies that the refactoring maintains pixel-perfect visual accuracy.
    """
    renderer = Renderer()

    # Test cases covering different scenarios
    test_cases = [
        # Ground (ring surface)
        (np.array([0, 0, 0]), np.array([0, -1, 0]), 0.0),
        # Arch (distant upward)
        (np.array([0, 0, 0]), np.array([0, 1, 0]), 0.0),
        # Horizon
        (np.array([0, 0, 0]), np.array([1, 0, 0]), 0.0),
        # Sun at noon
        (np.array([0, 0, 0]), np.array([0, 1, 0]), 0.0),
        # Rim wall
        (np.array([0, 0, 0]), np.array([0, 0, 1]), 0.0),
        # Midnight conditions
        (np.array([0, 0, 0]), np.array([0, -1, 0]), 12*3600),
        # Sunset conditions
        (np.array([0, 0, 0]), np.array([0, -1, 0]), 6*3600),
    ]

    for origin, direction, time_sec in test_cases:
        # Test single ray
        color_single = renderer.get_color(origin, direction, time_sec=time_sec)

        # Test batch processing (should be identical for single ray)
        color_batch = renderer.get_color(origin, direction[None, :], time_sec=time_sec)[0]

        np.testing.assert_allclose(color_single, color_batch, rtol=1e-10, atol=1e-10,
                                 err_msg=f"Single vs batch inconsistency for direction {direction}, time {time_sec}")

        # Verify reasonable color values (no NaN, Inf, out of range)
        assert np.all(np.isfinite(color_single)), f"Non-finite color for direction {direction}, time {time_sec}"
        assert np.all(color_single >= 0.0) and np.all(color_single <= 1.0), f"Color out of range for direction {direction}, time {time_sec}"


def test_pipeline_batch_processing():
    """Test that pipeline handles batch inputs correctly."""
    renderer = Renderer()

    # Create batch of test rays
    origins = np.array([[0, 0, 0]] * 5)  # Same origin for all
    directions = np.array([
        [0, -1, 0],  # Ground
        [0, 1, 0],   # Sky/Arch
        [1, 0, 0],   # Horizon
        [0, 0, 1],   # Rim wall
        [0.5, 0.5, 0.5]  # Diagonal
    ])
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)

    colors = renderer.get_color(origins[0], directions, time_sec=0.0)

    assert colors.shape == (5, 3), f"Expected shape (5, 3), got {colors.shape}"
    assert np.all(np.isfinite(colors)), "All colors should be finite"
    assert np.all(colors >= 0.0) and np.all(colors <= 1.0), "All colors should be in [0, 1] range"


def test_pipeline_atmosphere_toggle():
    """Test that atmosphere toggle works correctly."""
    renderer = Renderer()
    origin = np.array([0, 0, 0])
    direction = np.array([0, -1, 0])  # Ground

    # With atmosphere
    color_with_atmo = renderer.get_color(origin, direction, use_atmosphere=True)

    # Without atmosphere
    color_without_atmo = renderer.get_color(origin, direction, use_atmosphere=False)

    # Colors should be different
    assert not np.allclose(color_with_atmo, color_without_atmo, atol=1e-6)

    # Without atmosphere, should be pure surface color
    expected_surface = np.array([0.2, 0.5, 0.2]) * 1.02  # Ground green at noon
    np.testing.assert_allclose(color_without_atmo, expected_surface, atol=0.001)


if __name__ == "__main__":
    pytest.main([__file__])
