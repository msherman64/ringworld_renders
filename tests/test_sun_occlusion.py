import numpy as np
import pytest
from ringworld_renders.core import Renderer

def test_sun_occlusion_at_midnight():
    """
    Verify that the sun is occluded by shadow squares at midnight.
    """
    renderer = Renderer()
    # At midnight (12h), the sun is directly 'above' but blocked by a shadow square.
    time_midnight = 12.0 * 3600.0
    ray_origin = np.array([0.0, 0.0, 0.0])
    ray_up = np.array([0.0, 1.0, 0.0])
    
    # Check intersection distances
    t_ss = renderer.intersect_shadow_squares(ray_origin, ray_up, time_midnight)
    t_sun = renderer.intersect_sun(ray_origin, ray_up)
    
    assert t_ss < t_sun, f"Shadow square hit ({t_ss}) should be closer than sun ({t_sun}) at midnight."
    assert t_ss < np.inf, "Should hit a shadow square at midnight."
    
    # Check final color
    color = renderer.get_color(ray_origin, ray_up, time_sec=time_midnight, use_atmosphere=False)
    np.testing.assert_allclose(color, [0, 0, 0], atol=1e-5, err_msg="Sun should be completely black (occluded) at midnight.")

def test_sun_visible_between_squares():
    """
    Verify that the sun is visible when looking between shadow squares.
    """
    renderer = Renderer()
    # The assembly has 8 squares. Gaps are between squares.
    # ss_center_theta = (i + 0.5) * (2*pi/8)
    # Centers at: 22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5 degrees
    # Gaps at: 0, 45, 90, 135, 180, 225, 270, 315 degrees (approx, depending on width)
    
    # At T=0, Noon, theta of hit point (sun) is arctan2(0, -(-R)) = arctan2(0, R) = 0.
    # Center of square 0: (0.5) * (45) = 22.5 deg.
    # So at T=0, theta=0 is between square 7 (337.5) and square 0 (22.5).
    # Solar diameter is 0.53 deg, while gaps are much larger (approx 45 - 22 deg).
    
    time_noon = 0.0
    ray_origin = np.array([0.0, 0.0, 0.0])
    ray_up = np.array([0.0, 1.0, 0.0])
    
    t_ss = renderer.intersect_shadow_squares(ray_origin, ray_up, time_noon)
    t_sun = renderer.intersect_sun(ray_origin, ray_up)
    
    assert t_ss == np.inf, "Should NOT hit a shadow square at noon."
    assert t_sun < np.inf, "Should hit the sun at noon."
    
    # Check final color
    color = renderer.get_color(ray_origin, ray_up, time_sec=time_noon, use_atmosphere=False)
    np.testing.assert_allclose(color, [1.0, 1.0, 0.8], atol=1e-5, err_msg="Sun should be visible at noon.")

if __name__ == "__main__":
    pytest.main([__file__])
