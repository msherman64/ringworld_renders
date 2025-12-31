import numpy as np
import pytest
from ringworld_renders.core import Renderer

def test_rim_wall_geometry_hit():
    """
    Verify that an axial ray hits the Rim Wall.
    """
    renderer = Renderer()
    origin = np.array([0,0,0])
    ray_axial = np.array([0.0, 0.0, 1.0]) # Hits Positive Wall at z=W/2
    
    # Disable atmosphere to check surface color directly
    color = renderer.get_color(origin, ray_axial, use_atmosphere=False, use_shadows=False)
    
    # Wall color is [0.1, 0.1, 0.15]
    expected = np.array([0.1, 0.1, 0.15])
    np.testing.assert_allclose(color, expected, atol=0.01)

def test_rim_wall_extinction_from_center():
    """
    Verify that from the center (z=0), the Rim Wall (500k miles away) 
    is largely extinguished/blue-out.
    """
    renderer = Renderer()
    origin = np.array([0,0,0])
    ray_axial = np.array([0.0, 0.0, 1.0])
    
    # With atmosphere
    color = renderer.get_color(origin, ray_axial, use_atmosphere=True)
    
    # Should be mostly blue sky color [0.5, 0.7, 1.0] due to massive airmass
    # The wall color [0.1, 0.1, 0.15] should be extinguished.
    # Transmittance for 500,000 miles is effectively 0.
    
    assert color[2] > 0.4, "Center view should be blue (sky)"
    # Red channel of wall [0.1] should be gone.
    # But sky has some red [0.5].
    # We expect the result to conform to sky color, not wall color.
    # Sky color is [0.5, 0.7, 1.0]
    # Check that it's NOT the dark grey wall
    assert np.mean(color) > 0.3, "Should be bright sky, not dark wall"

def test_rim_wall_visibility_from_edge():
    """
    Verify that when close to the wall, we can see it.
    """
    renderer = Renderer()
    # Position: 10 miles from the positive wall
    # Wall is at z = W/2 = 500,000 miles
    dist_miles = 500000.0 - 10.0
    z_pos = dist_miles * 1609.34
    
    origin = np.array([0.0, 0.0, z_pos])
    ray_axial = np.array([0.0, 0.0, 1.0]) # Look at the wall 10 miles away
    
    color = renderer.get_color(origin, ray_axial, use_atmosphere=True)
    
    # 10 miles of air is very clear.
    # We should see the wall color [0.1, 0.1, 0.15] + slight blue tint.
    
    # Red channel check: Wall has 0.1. Sky has 0.5 but is scaled by opacity ~ 0.
    # Transmittance for 10 miles is near 1.0.
    assert color[0] < 0.2, "Should see dark wall (red channel low)"
    assert color[2] < 0.3, "Should see dark wall (blue channel low)"
    
if __name__ == "__main__":
    pytest.main([__file__])
