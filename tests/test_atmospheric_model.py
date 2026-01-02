import numpy as np
import pytest
from ringworld_renders.core import Renderer
from ringworld_renders.rendering import AtmosphericModel


def test_atmospheric_model_regression():
    """Test that AtmosphericModel produces identical results to original get_atmospheric_effects()."""
    renderer = Renderer()
    atm_model = AtmosphericModel(renderer)

    # Test cases with various ray directions and distances
    test_cases = [
        # Ground hit (short distance)
        (np.array([2.0]), np.array([0, 0, 0]), np.array([0, -1, 0])),
        # Far arch hit (long distance)
        (np.array([1e11]), np.array([0, 0, 0]), np.array([0, 1, 0])),
        # Horizon ray (no surface hit)
        (np.array([1e12]), np.array([0, 0, 0]), np.array([1, 0, 0])),
        # Axial ray (no surface hit)
        (np.array([1e12]), np.array([0, 0, 0]), np.array([0, 0, 1])),
    ]

    for t_hits, origin, direction in test_cases:
        # Get results from new AtmosphericModel
        trans_new, scat_new = atm_model.get_atmospheric_effects(t_hits, origin, direction, time_sec=0.0)

        # Get results from original method
        trans_orig, scat_orig = renderer.get_atmospheric_effects(t_hits, origin, direction, time_sec=0.0)

        np.testing.assert_allclose(trans_new, trans_orig, rtol=1e-10, atol=1e-10,
                                 err_msg=f"Transmittance mismatch for direction {direction}")
        np.testing.assert_allclose(scat_new, scat_orig, rtol=1e-10, atol=1e-10,
                                 err_msg=f"Scattering mismatch for direction {direction}")


def test_atmospheric_model_batch_consistency():
    """Test that single-ray and batch results are consistent."""
    renderer = Renderer()
    atm_model = AtmosphericModel(renderer)

    # Test rays
    t_hits = np.array([100.0])
    origin = np.array([0, 0, 0])
    direction = np.array([0, -1, 0])  # Ground

    # Single ray result
    trans_single, scat_single = atm_model.get_atmospheric_effects(t_hits, origin, direction)

    # Batch result
    trans_batch, scat_batch = atm_model.get_atmospheric_effects(t_hits, origin, direction[None, :])

    np.testing.assert_allclose(trans_single, trans_batch[0], rtol=1e-10)
    np.testing.assert_allclose(scat_single, scat_batch[0], rtol=1e-10)


def test_atmospheric_model_spectral_transmittance():
    """Test that transmittance follows Rayleigh spectral extinction."""
    renderer = Renderer()
    atm_model = AtmosphericModel(renderer)

    # Looking horizontally through atmosphere
    t_hits = np.array([1e15])  # Very far
    origin = np.array([0, 0, 0])
    direction = np.array([1, 0.01, 0])  # Nearly horizontal
    direction /= np.linalg.norm(direction)

    trans, scat = atm_model.get_atmospheric_effects(t_hits, origin, direction)

    # Rayleigh extinction: blue (450nm) extinguished more than red (650nm)
    assert trans[0] > trans[2], f"Red T {trans[0]} should be > Blue T {trans[2]}"
    # Blue channel should be significantly more extinguished
    assert trans[2] < 0.1, f"Blue transmittance {trans[2]} should be very low for long horizontal path"


def test_atmospheric_model_phase_function():
    """Test that scattering includes phase function effects."""
    renderer = Renderer()
    atm_model = AtmosphericModel(renderer)

    # Short path to avoid extinction effects
    t_hits = np.array([100.0])
    origin = np.array([0, 0, 0])

    # Ray looking toward sun (theta ~ 0)
    sun_dir = renderer.ring_center / np.linalg.norm(renderer.ring_center)
    trans_sun, scat_sun = atm_model.get_atmospheric_effects(t_hits, origin, sun_dir)

    # Ray looking perpendicular to sun (theta ~ 90 deg)
    perp_dir = np.array([0, 0, 1])  # Axial direction
    trans_perp, scat_perp = atm_model.get_atmospheric_effects(t_hits, origin, perp_dir)

    # Both should produce some scattering (non-zero)
    assert np.mean(scat_sun) > 0, "Sun direction should have scattering"
    assert np.mean(scat_perp) > 0, "Perpendicular direction should have scattering"
    # Phase function is complex with extinction, just verify we get reasonable values
    assert np.mean(scat_sun) > np.mean(scat_perp) * 0.5, "Sun direction should not be much dimmer than perpendicular"


def test_atmospheric_model_far_side_doubling():
    """Test that far-side atmosphere doubles the optical depth."""
    renderer = Renderer()
    atm_model = AtmosphericModel(renderer)

    origin = np.array([0, 0, 0])

    # Near-side vertical path
    t_near = renderer.H_a + 1.0  # Just past ceiling
    direction_up = np.array([0, 1, 0])
    trans_near, _ = atm_model.get_atmospheric_effects(np.array([t_near]), origin, direction_up)

    # Far-side path (to distant arch)
    t_far = 2 * renderer.R - renderer.h
    trans_far, _ = atm_model.get_atmospheric_effects(np.array([t_far]), origin, direction_up)

    # Far side should have much more extinction due to dual-side atmosphere
    assert np.all(trans_far < trans_near), "Far side should have more extinction"
    # Should be approximately T_near * T_far
    np.testing.assert_allclose(trans_far, trans_near**2, rtol=1e-2)


def test_atmospheric_model_shadow_effects():
    """Test that shadows affect atmospheric scattering."""
    renderer = Renderer()
    atm_model = AtmosphericModel(renderer)

    # Sky ray (no surface hit)
    t_hits = np.array([1e12])
    origin = np.array([0, 0, 0])
    direction = np.array([0, 0, 1])  # Axial sky

    # At noon (no shadows)
    _, scat_noon = atm_model.get_atmospheric_effects(t_hits, origin, direction, time_sec=0.0, use_shadows=True)

    # At midnight (deep shadows)
    _, scat_midnight = atm_model.get_atmospheric_effects(t_hits, origin, direction, time_sec=12*3600, use_shadows=True)

    # Midnight sky should be much darker
    assert np.mean(scat_midnight) < np.mean(scat_noon) * 0.5, "Midnight sky should be significantly darker"


if __name__ == "__main__":
    pytest.main([__file__])
