"""
Shadow calculation module for Ringworld Renderer.

Handles shadow square positioning, angular calculations, and shadow factor evaluation.
"""

import numpy as np
from ringworld_renders import constants


def omega_ss(N_ss):
    """Angular velocity of Shadow Squares (rad/s)."""
    T_assembly = N_ss * constants.SOLAR_DAY_SECONDS
    return -2.0 * np.pi / T_assembly


def angular_width(L_ss, R_ss):
    """Angular width of a single Shadow Square (radians)."""
    return L_ss / R_ss


class ShadowModel:
    """
    Handles shadow calculations for the Ringworld renderer.

    Provides methods for calculating shadow factors based on shadow square positions
    and angular relationships.
    """

    def __init__(self, N_ss=None, R_ss=None, L_ss=None):
        """
        Initialize the shadow model.

        Args:
            N_ss: Number of shadow squares (default: constants.NUM_SHADOW_SQUARES)
            R_ss: Shadow square orbital radius in meters (default: constants.SS_RADIUS_METERS)
            L_ss: Shadow square length in meters (default: constants.SS_LENGTH_METERS)
        """
        self.N_ss = N_ss if N_ss is not None else constants.NUM_SHADOW_SQUARES
        self.R_ss = R_ss if R_ss is not None else constants.SS_RADIUS_METERS
        self.L_ss = L_ss if L_ss is not None else constants.SS_LENGTH_METERS

        # Pre-compute derived values
        self.omega_ss_val = omega_ss(self.N_ss)
        self.angular_width_val = angular_width(self.L_ss, self.R_ss)

    def get_shadow_factor(self, positions, time_sec):
        """
        Vectorized shadow factor calculation for given positions and time.

        Args:
            positions: Nx3 array of (x,y,z) positions on the ring
            time_sec: Current time in seconds

        Returns:
            Array of shadow factors [0,1] where 1.0 = full sunlight, 0.0 = full shadow
        """
        # Handle single position
        positions = np.atleast_2d(positions)
        if positions.shape[0] == 0:
            return np.array([])

        # Calculate angular position of each point on the ring
        # theta = atan2(x, -y) maps to the ring's angular coordinate
        theta_points = np.arctan2(positions[:, 0], -positions[:, 1])

        # Calculate shadow factors
        shadow_factors = np.ones(len(positions), dtype=float)

        # Solar angular diameter for penumbra calculation (approximate)
        sigma_rad = np.deg2rad(0.53)  # Sun's angular diameter
        penumbra_width = sigma_rad

        # Start of full umbra and penumbra boundaries
        half_width = self.angular_width_val / 2.0
        hu = half_width - penumbra_width / 2.0  # Start of full umbra
        hw = half_width + penumbra_width / 2.0  # End of penumbra

        # Check each shadow square
        for i in range(self.N_ss):
            # Angular position of shadow square center
            theta_center = (i + 0.5) * (2.0 * np.pi / self.N_ss) + self.omega_ss_val * time_sec

            # Angular distance from point to shadow square center (handle wraparound)
            delta_theta = theta_points - theta_center
            delta_theta = (delta_theta + np.pi) % (2 * np.pi) - np.pi
            abs_delta_theta = np.abs(delta_theta)

            # Full shadow (umbra)
            umbra_mask = abs_delta_theta < hu
            shadow_factors[umbra_mask] = 0.0

            # Penumbra (gradual transition)
            penumbra_mask = (abs_delta_theta >= hu) & (abs_delta_theta < hw)
            if np.any(penumbra_mask):
                # Linear interpolation in penumbra
                factor = (abs_delta_theta[penumbra_mask] - hu) / (hw - hu)
                shadow_factors[penumbra_mask] = np.minimum(shadow_factors[penumbra_mask], factor)

        return shadow_factors
