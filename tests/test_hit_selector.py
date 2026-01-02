import numpy as np
import pytest
from ringworld_renders.core import Renderer
from ringworld_renders.rendering import HitSelector, HitType


def test_hit_selector_regression():
    """
    Test that HitSelector produces identical results to the original get_color() intersection logic.
    """
    renderer = Renderer()
    selector = HitSelector(renderer)

    # Test cases covering different hit types
    test_cases = [
        # Ground (ring surface)
        (np.array([0, 0, 0]), np.array([0, -1, 0])),
        # Arch (upward)
        (np.array([0, 0, 0]), np.array([0, 1, 0])),
        # Horizon
        (np.array([0, 0, 0]), np.array([1, 0, 0])),
        # Sun at noon
        (np.array([0, 0, 0]), np.array([0, 1, 0])),
        # Rim wall
        (np.array([0, 0, 0]), np.array([0, 0, 1])),
    ]

    for origin, direction in test_cases:
        # Get results from new HitSelector
        hit_result = selector.select_primary(origin, direction, time_sec=0.0)

        # Get results from original intersection logic
        t_sun = renderer.intersect_sun(origin, direction)
        t_ring = renderer.intersect_ring(origin, direction)
        t_wall = renderer.intersect_rim_walls(origin, direction)
        t_ss = renderer.intersect_shadow_squares(origin, direction, 0.0)

        # Find original primary hit
        t_min = min(t for t in [t_sun, t_ring, t_ss, t_wall] if np.isfinite(t))

        # Determine original hit type
        if t_sun == t_min:
            expected_type = HitType.SUN.value
        elif t_ring == t_min:
            expected_type = HitType.RING.value
        elif t_wall == t_min:
            expected_type = HitType.RIM_WALL.value
        elif t_ss == t_min:
            expected_type = HitType.SHADOW_SQUARE.value
        else:
            expected_type = HitType.SKY.value

        # Verify results match
        assert hit_result.distance[0] == pytest.approx(t_min, rel=1e-10)
        assert hit_result.hit_type[0] == expected_type

        if np.isfinite(t_min):
            expected_hit_point = origin + t_min * direction
            np.testing.assert_allclose(hit_result.hit_point[0], expected_hit_point, rtol=1e-10)


def test_hit_selector_batch_consistency():
    """
    Test that single-ray and batch results are consistent.
    """
    renderer = Renderer()
    selector = HitSelector(renderer)

    # Test ray
    origin = np.array([0, 0, 0])
    direction = np.array([0, -1, 0])  # Ground

    # Single ray result
    single_result = selector.select_primary(origin, direction, time_sec=0.0)

    # Batch result with same ray
    batch_result = selector.select_primary(origin, direction[None, :], time_sec=0.0)

    assert single_result.distance[0] == pytest.approx(batch_result.distance[0])
    assert single_result.hit_type[0] == batch_result.hit_type[0]
    np.testing.assert_allclose(single_result.hit_point[0], batch_result.hit_point[0])


def test_hit_selector_priority_ordering():
    """
    Test that hit priority ordering is maintained: sun > ring > shadow_square > rim_wall.
    """
    renderer = Renderer()
    selector = HitSelector(renderer)

    # At midnight, sun should be behind shadow squares
    origin = np.array([0, 0, 0])
    direction = np.array([0, 1, 0])  # Up toward arch

    # At noon - should hit sun first
    noon_result = selector.select_primary(origin, direction, time_sec=0.0)
    assert noon_result.hit_type[0] == HitType.SUN.value

    # At midnight - sun should be occluded by shadow squares
    midnight_result = selector.select_primary(origin, direction, time_sec=12*3600)
    # Should hit shadow square or ring, not sun
    assert midnight_result.hit_type[0] in [HitType.SHADOW_SQUARE.value, HitType.RING.value]
    assert midnight_result.hit_type[0] != HitType.SUN.value


def test_hit_selector_shape_validation():
    """
    Test that HitResult shapes are validated correctly.
    """
    renderer = Renderer()
    selector = HitSelector(renderer)

    origin = np.array([0, 0, 0])
    directions = np.array([[0, -1, 0], [0, 1, 0]])  # Two rays

    result = selector.select_primary(origin, directions, time_sec=0.0)

    assert result.distance.shape == (2,)
    assert result.hit_type.shape == (2,)
    assert result.hit_point.shape == (2, 3)


if __name__ == "__main__":
    pytest.main([__file__])
