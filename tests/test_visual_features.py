import numpy as np
import pytest
from ringworld_renders.core import Renderer

def test_geometry_precision_deltar():
    """
    Unit Test: Verify the Delta-R solver prevents catastrophic cancellation.
    At 10ft eye height (h), the ground distance at nadir should be h.
    """
    renderer = Renderer()
    h_meters = renderer.h
    
    # Ray pointing straight down (nadir)
    ray_origin = np.array([0.0, 0.0, 0.0])
    ray_direction = np.array([0.0, -1.0, 0.0])
    
    t = renderer.intersect_ring(ray_origin, ray_direction)
    # The expected intersection is at distance h (the distance to the floor).
    # We use a slightly looser tolerance to account for the MILES_TO_METERS constant used.
    assert t == pytest.approx(h_meters, rel=1e-5), f"Expected distance {h_meters}, got {t}"


def test_upward_horizon_validation():
    """
    Diagnostic Test: Ensure the ground rises with distance (Upward Horizon).
    As the ray looks further 'forward' (slightly above horizontal), 
    the intersection distance should increase dramatically.
    """
    renderer = Renderer()
    
    # Angles in degrees (elevation)
    elevations = np.array([0.0, 0.01, 0.05, 0.1, 1.0, 10.0])
    # Convert to directions
    theta = np.deg2rad(elevations)
    directions = np.zeros((len(elevations), 3))
    directions[:, 0] = np.cos(theta) # Forward
    directions[:, 1] = np.sin(theta) # Up
    
    t = renderer.intersect_ring(np.array([0,0,0]), directions)
    
    # Print distances for manual verification in test output
    print("\nUpward Horizon Distances (miles):")
    for angle, dist in zip(elevations, t):
        print(f"Angle {angle}°: {dist/1609.34:.2f} miles")
        
    # Validation: Distances should be strictly increasing with elevation angle (within the first 90 deg)
    assert np.all(np.diff(t) > 0), "Distances should increase as we look higher towards the arch."
    # At 10 degrees, the distance should be enormous (thousands of miles)
    assert t[-1] > 1000 * 1609.34, "Elevation of 10 degrees should hit the distant arch thousands of miles away."

def test_atmospheric_blueout_validation():
    """
    Diagnostic Test: Ensure atmospheric effects match theoretical grounding.
    Distant arch should be dominated by scattered sky color.
    """
    renderer = Renderer()
    
    # Look at the far-side arch, but away from the sun (e.g. look axial)
    ray_axial = np.array([0.0, 0.1, 1.0])
    ray_axial /= np.linalg.norm(ray_axial)
    color = renderer.get_color(np.array([0,0,0]), ray_axial, use_atmosphere=True)
    
    # Base blue of sky is [0.5, 0.7, 1.0]
    # At this angle, distance is large, so scattering should dominate.
    assert color[2] > 0.05, "Blue channel should be visible due to sky scattering."
    
    # Compare with atmosphere OFF
    color_no_atmo = renderer.get_color(np.array([0,0,0]), ray_axial, use_atmosphere=False)
    # Without atmosphere, looking into space (no hit) should be black
    assert color_no_atmo[2] < 0.01, "Without atmosphere, space should be black."


def test_volumetric_shadow_validation():
    """
    Diagnostic Test: Verify air darkens correctly in shadow (Volumetric Shadowing).
    Compare the sky in-scattering contribution at noon vs midnight.
    """
    renderer = Renderer()
    ray_side = np.array([0.0, 0.0, 1.0]) # Pure sky, no surface hit
    
    color_noon = renderer.get_color(np.array([0,0,0]), ray_side, time_sec=0.0)
    color_midnight = renderer.get_color(np.array([0,0,0]), ray_side, time_sec=12.0 * 3600)
    
    # Noon sky is bright [0.5, 0.7, 1.0] scaled by (1.1)
    # Midnight sky is dark [0.5, 0.7, 1.0] scaled by (0.1) -> ambient glow
    assert np.mean(color_noon) > 5 * np.mean(color_midnight), "Noon sky should be much brighter than midnight sky."
    assert np.mean(color_midnight) > 0, "Midnight sky should have some ambient glow."

def test_ring_shine_validation():
    """
    Diagnostic Test: Verify ground illumination on the night side (Ring-shine).
    """
    renderer = Renderer()
    ray_down = np.array([0.0, -1.0, 0.0])
    
    # Midnight ground (all shadow, just ring-shine)
    color_midnight = renderer.get_color(np.array([0,0,0]), ray_down, time_sec=12.0 * 3600)
    
    expected_midnight = np.array([0.2, 0.5, 0.2]) * 0.10 # s_factor=0, ambient=0.10
    np.testing.assert_allclose(color_midnight, expected_midnight, atol=0.001)


    
    # Noon ground (no shadow, direct sun + ring-shine)
    color_noon = renderer.get_color(np.array([0,0,0]), ray_down, time_sec=0.0)
    expected_noon = np.array([0.2, 0.5, 0.2]) * 1.02 # s_factor=1, ambient=0.02
    np.testing.assert_allclose(color_noon, expected_noon, atol=0.001)

def test_perspective_fov_scaling():
    """
    Diagnostic Test: Verify changes of perspective (FOV) function correctly.
    Lower FOV should 'zoom in', changing the pixel content accordingly.
    """
    renderer = Renderer()
    # Looking at the horizon
    look_at = np.array([1.0, 0.0, 0.0])
    
    # Wide FOV image
    img_wide = renderer.render(width=32, height=32, fov=110.0, look_at=look_at)
    # Narrow FOV image
    img_narrow = renderer.render(width=32, height=32, fov=20.0, look_at=look_at)
    
    # The narrow FOV image of the horizon should be 'bluer' on average because it's looking 
    # through more concentrated airmass at the horizon, whereas 110 FOV includes 
    # local green ground.
    assert np.mean(img_narrow[:,:,2]) > np.mean(img_wide[:,:,2]), "Narrow FOV on horizon should show more haze/blue."

def test_far_side_atmo_doubling():
    """
    Diagnostic Test: Verify that looking at the distant zenith arch doubles the atmosphere.
    The path to the far side zenith should be roughly 2 * 100 miles = 200 miles of air.
    Compared to just hitting the ceiling (100 miles).
    """
    renderer = Renderer()
    # 1. Near ceiling distance (looking UP until ceiling exit)
    ray_up = np.array([0.0, 1.0, 0.0])
    # Distance just past the ceiling (H_a is 100 miles)
    t_just_past_ceiling = (renderer.H_a + 1.0) 
    trans_vertical, _ = renderer.get_atmospheric_effects(np.array([t_just_past_ceiling]), ray_up)
    
    # 2. Far side arch distance (looking UP until far surface)
    t_far_surface = 2 * renderer.R - renderer.h
    trans_far, _ = renderer.get_atmospheric_effects(np.array([t_far_surface]), ray_up)
    
    # Validation: far side should have much more extinction (lower transmittance)
    # Check blue channel (most extinguished)
    assert trans_far[2] < trans_vertical[2] * 0.95, "Far side arch should have significantly more extinction due to dual-side atmosphere."
    # With tau_zenith = 0.06, trans_vertical is ~0.94. trans_far should be ~0.94 * 0.94 = 0.88.
    assert np.allclose(trans_far, trans_vertical**2, rtol=1e-2), "Analytical path doubling should match T_near * T_far"



def test_axial_sky_clarity():
    """
    Diagnostic Test: Verify that looking out the "open side" of the ring results in no haze.
    With a 1 million mile width (W), looking axial (dz=1.0) should hit the boundary quickly.
    """
    renderer = Renderer()
    # Look pure axial (+Z)
    ray_axial = np.array([0.0, 0.0, 1.0])
    trans, scat = renderer.get_atmospheric_effects(np.array([1e12]), ray_axial)
    
    # Distance to side-wall is W/2 = 500,000 miles. 
    # Ceiling is only 100 miles. Wait. 
    # If I'm on the ground, the 'ceiling' is 100 miles UP.
    # But the 'wall' is 500,000 miles to the RIGHT.
    # So looking Axial should still be 100 miles of air? 
    # NO. If you look parallel to the ground (dy=0), you never hit the ceiling.
    # In an infinite tube, you'd have infinite air. 
    # With width clipping, you should hit the wall at 500k miles.
    
    # Actually, even 500k miles is a lot of air. 
    # But the Ringworld has RIM WALLS (not yet implemented as geometry, but implied).
    # The atmosphere only exists within the width W.
    
    # If we look axial, we should see the "edge" of the atmosphere.
    # A ray looking slightly 'up' and 'axial' will exit the ceiling or the wall.
    # A ray looking pure axial (dy=0) will exit the wall.
    
    # Let's check a ray looking diagonally out (dy=0.1, dz=1.0)
    ray_diag = np.array([0.0, 0.1, 1.0])
    ray_diag /= np.linalg.norm(ray_diag)
    trans_diag, scat_diag = renderer.get_atmospheric_effects(np.array([1e12]), ray_diag)
    
    # If the bug exists (infinite tube), this ray will trace millions of miles.
    # If fixed, it should be bounded by something reasonable (the width).
    # Actually, looking at 90 degrees (axial) through 500k miles of air
    # should result in an opaque blue fog with ambient glow (1.0 + 0.1 = 1.1)
    assert np.all(scat <= 1.11), f"Scattering {scat} should be stable and bounded by ambient glow."

def test_sky_phase_function():
    """
    TDD: Verify that sky scattering follows the Rayleigh Phase Function.
    The sky should be brighter looking near the sun than at 90 degrees to it.
    """
    renderer = Renderer()
    # At noon, sun is at theta=pi/2 (roughly 'up' in local frame)
    # 1. Looking pure UP (near sun)
    _, scat_up = renderer.get_atmospheric_effects(np.array([100.0]), np.array([0.0, 1.0, 0.0]), time_sec=0.0)
    # 2. Looking at horizon (90 deg to sun)
    _, scat_horiz = renderer.get_atmospheric_effects(np.array([100.0]), np.array([1.0, 0.0, 0.0]), time_sec=0.0)
    
    # Rayleigh P(theta) = 3/4(1 + cos^2(theta))
    # Near sun (theta ~ 0) -> P ~ 1.5
    # Horizon (theta ~ 90) -> P ~ 0.75
    # So scat_up should be ~2x brighter than scat_horiz.
    assert np.mean(scat_up) > np.mean(scat_horiz) * 1.5, "Sky should follow Rayleigh phase function (brighter near sun)"

def test_dynamic_ring_shine_midnight_noon():
    """
    TDD: Verify that 'ambient' ring-shine is brighter at midnight (full arch illumination)
    than at noon (backlit/hidden arch).
    """
    renderer = Renderer()
    origin = np.array([0.0, 0.0, 0.0])
    ray_down = np.array([0.0, -1.0, 0.0])
    
    # 1. Midnight (approx 12hr = 43200s)
    color_midnight = renderer.get_color(origin, ray_down, time_sec=43200.0, use_atmosphere=False, use_shadows=False)
    # 2. Noon (0s)
    color_noon = renderer.get_color(origin, ray_down, time_sec=0.0, use_atmosphere=False, use_shadows=False)
    
    # At midnight, the arch is overhead and fully lit. Ground should be brighter than noon ground.
    assert np.mean(color_midnight) > np.mean(color_noon), "Midnight ground should be brighter due to dynamic ring-shine"



def test_spectral_reddening():
    """
    TDD: Verify that the atmosphere implements spectral extinction (Rayleigh reddening).
    Blue light should be extinguished faster than red light over long paths.
    """
    renderer = Renderer()
    # Looking at horizon (dy=0.01)
    # Airmass should be ~100
    ray_horiz = np.array([1.0, 0.01, 0.0])
    ray_horiz /= np.linalg.norm(ray_horiz)
    
    # Hit distance is very large (misses ground, exits ceiling)
    t_hits = np.array([1e15]) 
    trans, scat = renderer.get_atmospheric_effects(t_hits, ray_horiz)
    
    print(f"\nSpectral Extinction Debug:")
    print(f"Ray dy: {ray_horiz[1]}")
    print(f"Transmittance: {trans}")
    print(f"Scattering: {scat}")
    
    # trans should be a 3-vector [T_r, T_g, T_b]
    assert hasattr(trans, "__len__") and len(trans) == 3, f"Transmittance should be spectral 3-vector, got {trans}"
    # Check that blue is extinguished more than red
    assert trans[0] > trans[2], f"Red T {trans[0]} should be > Blue T {trans[2]}"
    # For airmass 100, tau_b = 6.0, T_b = 0.002. tau_r = 1.38, T_r = 0.25.
    assert trans[0] > trans[2] * 5.0, "Red transmittance should be much higher than blue (Rayleigh reddening)"









if __name__ == "__main__":
    pytest.main([__file__])
