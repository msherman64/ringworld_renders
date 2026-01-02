import numpy as np
import pytest
from ringworld_renders.core import Renderer
from ringworld_renders.rendering import MaterialSystem, HitResult, HitType


def test_material_system_sun():
    """Test sun material returns correct color."""
    renderer = Renderer()
    material_system = MaterialSystem(renderer)

    # Create hit result for sun
    hits = HitResult(
        distance=np.array([1.0]),
        hit_type=np.array([HitType.SUN.value]),
        hit_point=np.array([[0.0, 1.0, 0.0]])
    )

    colors = material_system.get_surface_color(hits, time_sec=0.0)
    expected = np.array([1.0, 1.0, 0.8])
    np.testing.assert_allclose(colors[0], expected)


def test_material_system_shadow_square():
    """Test shadow square material returns correct color."""
    renderer = Renderer()
    material_system = MaterialSystem(renderer)

    # Create hit result for shadow square
    hits = HitResult(
        distance=np.array([1.0]),
        hit_type=np.array([HitType.SHADOW_SQUARE.value]),
        hit_point=np.array([[0.0, 1.0, 0.0]])
    )

    # Normal mode
    colors = material_system.get_surface_color(hits, time_sec=0.0, debug_shadow_squares=False)
    expected = np.array([0.0, 0.0, 0.0])
    np.testing.assert_allclose(colors[0], expected)

    # Debug mode - should return debug colors
    colors_debug = material_system.get_surface_color(hits, time_sec=0.0, debug_shadow_squares=True)
    assert colors_debug[0] is not None  # Should be some debug color


def test_material_system_rim_wall():
    """Test rim wall material returns correct color."""
    renderer = Renderer()
    material_system = MaterialSystem(renderer)

    # Create hit result for rim wall
    hits = HitResult(
        distance=np.array([1.0]),
        hit_type=np.array([HitType.RIM_WALL.value]),
        hit_point=np.array([[0.0, 1.0, 1.0]])
    )

    colors = material_system.get_surface_color(hits, time_sec=0.0)
    expected = np.array([0.1, 0.1, 0.15])
    np.testing.assert_allclose(colors[0], expected)


def test_material_system_ring():
    """Test ring material with shadow factor and ring-shine."""
    renderer = Renderer()
    material_system = MaterialSystem(renderer)

    # Create hit result for ring surface
    hit_point = np.array([[0.0, renderer.center_y - renderer.h, 0.0]])  # Ground level
    hits = HitResult(
        distance=np.array([renderer.h]),
        hit_type=np.array([HitType.RING.value]),
        hit_point=hit_point
    )

    # Test at noon (no ring-shine)
    colors_noon = material_system.get_surface_color(hits, time_sec=0.0, use_ring_shine=True)
    expected_noon = np.array([0.2, 0.5, 0.2]) * 1.02  # s_factor=1, ambient=0.02
    np.testing.assert_allclose(colors_noon[0], expected_noon, atol=0.001)

    # Test at midnight (maximum ring-shine, but fully shadowed)
    colors_midnight = material_system.get_surface_color(hits, time_sec=12*3600, use_ring_shine=True)
    expected_midnight = np.array([0.2, 0.5, 0.2]) * 0.10  # s_factor=0, ambient=0.10
    np.testing.assert_allclose(colors_midnight[0], expected_midnight, atol=0.001)

    # Test with shadows disabled
    colors_no_shadows = material_system.get_surface_color(hits, time_sec=0.0, use_shadows=False)
    expected_no_shadows = np.array([0.2, 0.5, 0.2]) * 1.02
    np.testing.assert_allclose(colors_no_shadows[0], expected_no_shadows, atol=0.001)


def test_material_system_regression():
    """Test that MaterialSystem produces identical results to original get_color() logic."""
    renderer = Renderer()
    material_system = MaterialSystem(renderer)

    # Test cases matching original get_color() logic
    test_cases = [
        # Sun
        (HitType.SUN, np.array([[0.0, 1.0, 0.0]]), 0.0, [1.0, 1.0, 0.8]),
        # Shadow square (normal)
        (HitType.SHADOW_SQUARE, np.array([[0.0, 1.0, 0.0]]), 0.0, [0.0, 0.0, 0.0]),
        # Rim wall
        (HitType.RIM_WALL, np.array([[0.0, 1.0, 1.0]]), 0.0, [0.1, 0.1, 0.15]),
        # Ring at noon
        (HitType.RING, np.array([[0.0, renderer.center_y - renderer.h, 0.0]]), 0.0, [0.2*1.02, 0.5*1.02, 0.2*1.02]),
        # Ring at midnight (fully shadowed but with ring-shine)
        (HitType.RING, np.array([[0.0, renderer.center_y - renderer.h, 0.0]]), 12*3600, [0.2*0.10, 0.5*0.10, 0.2*0.10]),
    ]

    for hit_type, hit_point, time_sec, expected_color in test_cases:
        hits = HitResult(
            distance=np.array([1.0]),
            hit_type=np.array([hit_type.value]),
            hit_point=hit_point
        )

        colors = material_system.get_surface_color(hits, time_sec=time_sec)
        np.testing.assert_allclose(colors[0], expected_color, atol=0.001,
                                 err_msg=f"Failed for {hit_type.value}")


def test_material_system_batch_processing():
    """Test that MaterialSystem handles batch inputs correctly."""
    renderer = Renderer()
    material_system = MaterialSystem(renderer)

    # Create batch of different hit types
    hits = HitResult(
        distance=np.array([1.0, 2.0, 3.0]),
        hit_type=np.array([HitType.SUN.value, HitType.RING.value, HitType.RIM_WALL.value]),
        hit_point=np.array([
            [0.0, 1.0, 0.0],  # Sun
            [0.0, renderer.center_y - renderer.h, 0.0],  # Ring
            [0.0, 1.0, 1.0]   # Wall
        ])
    )

    colors = material_system.get_surface_color(hits, time_sec=0.0)

    # Check each result
    assert colors.shape == (3, 3)
    np.testing.assert_allclose(colors[0], [1.0, 1.0, 0.8])  # Sun
    np.testing.assert_allclose(colors[1], [0.2*1.02, 0.5*1.02, 0.2*1.02], atol=0.001)  # Ring
    np.testing.assert_allclose(colors[2], [0.1, 0.1, 0.15])  # Wall


if __name__ == "__main__":
    pytest.main([__file__])
