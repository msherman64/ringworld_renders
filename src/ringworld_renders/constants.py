"""
Physical constants and configuration for the Ringworld Renderer.
"""

# Units
MILES_TO_METERS = 1609.34

# System Physical Parameters (Niven Scale)
DEFAULT_RADIUS_MILES = 92955807.0
DEFAULT_WIDTH_MILES = 1000000.0
DEFAULT_EYE_HEIGHT_MILES = 0.001242742 # ~6.5 ft approx? Or specific elev.

# Sun
SUN_RADIUS_METERS = 432474.0 * MILES_TO_METERS

# Shadow Squares
NUM_SHADOW_SQUARES = 8
SS_RADIUS_METERS = 36571994.0 * MILES_TO_METERS
SS_LENGTH_METERS = 14360000.0 * MILES_TO_METERS
SS_HEIGHT_METERS = 1200000.0 * MILES_TO_METERS

# Time
SOLAR_DAY_SECONDS = 24.0 * 3600.0

# Visualization Colors
# Distinct colors for shadow squares (consistent across views)
SS_COLORS = [
    [1.0, 0.2, 0.2], # Red
    [0.2, 1.0, 0.2], # Green
    [0.2, 0.2, 1.0], # Blue
    [1.0, 1.0, 0.2], # Yellow
    [1.0, 0.2, 1.0], # Magenta
    [0.2, 1.0, 1.0], # Cyan
    [1.0, 0.6, 0.0], # Orange
    [0.6, 0.0, 1.0]  # Purple
]
