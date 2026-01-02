"""
Shadow square physics and positioning for the Ringworld renderer.

This module handles all calculations related to shadow squares: their motion,
intersection detection, and shadow factor computation with penumbra effects.
"""
import numpy as np
from ringworld_renders import constants
from ringworld_renders.intersections import solve_quadratic_vectorized


class ShadowModel:
    """
    Handles shadow square physics and positioning.
    """

    def __init__(self, N_ss=None, R_ss=None, L_ss=None, H_ss=None, sun_angular_diameter=None):
        """
        Initialize shadow square parameters.

        Args:
            N_ss: Number of shadow squares
            R_ss: Shadow square orbital radius (meters)
            L_ss: Shadow square length (meters)
            H_ss: Shadow square height (meters)
            sun_angular_diameter: Angular diameter of sun (degrees)
        """
        self.N_ss = N_ss or constants.NUM_SHADOW_SQUARES
        self.R_ss = R_ss or constants.SS_RADIUS_METERS
        self.L_ss = L_ss or constants.SS_LENGTH_METERS
        self.H_ss = H_ss or constants.SS_HEIGHT_METERS
        self.sun_angular_diameter = sun_angular_diameter or 0.53  # degrees

    @property
    def omega_ss(self):
        """Angular velocity of Shadow Squares (rad/s)."""
        T_assembly = self.N_ss * constants.SOLAR_DAY_SECONDS
        return -2.0 * np.pi / T_assembly

    @property
    def angular_width(self):
        """Angular width of a single Shadow Square (radians)."""
        return self.L_ss / self.R_ss

    def intersect_shadow_squares(self, ray_origin, ray_directions, time_sec, center_y):
        """
        Vectorized intersection of ray and the shadow square cylindrical shell.

        Args:
            ray_origin: (3,) origin point
            ray_directions: (N, 3) array of ray directions
            time_sec: Current time in seconds
            center_y: Y-coordinate of ring center

        Returns:
            Array of intersection distances (inf where no intersection)
        """
        is_single = ray_directions.ndim == 1
        if is_single:
            ray_directions = ray_directions[None, :]

        dx = ray_directions[:, 0]
        dy = ray_directions[:, 1]
        ox, oy, oz = ray_origin

        # Intersection with cylinder (r = R_ss)
        # (x - cx)^2 + (y - cy)^2 = R^2
        # cx = 0, cy = center_y

        a = dx**2 + dy**2
        b = 2.0 * (ox*dx + (oy - center_y)*dy)
        c = ox**2 + (oy - center_y)**2 - self.R_ss**2

        t1, t2, valid_mask = solve_quadratic_vectorized(a, b, c)

        t = np.full(ray_directions.shape[0], np.inf)

        if np.any(valid_mask):
            # Select smallest positive t
            best_t = np.full(t1.shape, np.inf)
            m1 = t1 > 1e-6
            best_t[m1] = t1[m1]
            m2 = (t2 > 1e-6) & (t2 < best_t)
            best_t[m2] = t2[m2]

            # Now check which hits actually intersect a square (vs the gaps)
            hit_mask = best_t < np.inf
            if np.any(hit_mask):
                valid_hits = best_t[hit_mask]
                dirs = ray_directions[hit_mask]
                hit_p = ray_origin + valid_hits[:, None] * dirs

                # Check height (axial width)
                # Shadow squares are centered at z=0, width H_ss
                z_check = np.abs(hit_p[:, 2]) <= self.H_ss / 2.0

                # Check angular position
                theta = np.arctan2(hit_p[:, 0], -(hit_p[:, 1] - center_y))

                half_width = self.angular_width / 2.0
                omega = self.omega_ss

                in_any_square = np.zeros(hit_p.shape[0], dtype=bool)
                for i in range(self.N_ss):
                    ss_center_theta = (i + 0.5) * (2.0 * np.pi / self.N_ss) + omega * time_sec
                    d_theta = (theta - ss_center_theta + np.pi) % (2.0 * np.pi) - np.pi
                    in_any_square |= (np.abs(d_theta) <= half_width)

                final_hit_mask = z_check & in_any_square

                # Update t[valid_mask]
                results = np.full(best_t.shape, np.inf)
                results[hit_mask] = np.where(final_hit_mask, valid_hits, np.inf)
                t = results

        return t[0] if is_single else t

    def get_shadow_factor(self, hit_points, time_sec, center_y):
        """
        Vectorized shadow factor calculation with penumbra effects.

        Args:
            hit_points: (N, 3) array of hit points
            time_sec: Current time in seconds
            center_y: Y-coordinate of ring center

        Returns:
            Array of shadow factors (0.0 = full shadow, 1.0 = full sun)
        """
        is_single = hit_points.ndim == 1
        if is_single:
            hit_points = hit_points[None, :]

        # Angular position theta
        theta = np.arctan2(hit_points[:, 0], -(hit_points[:, 1] - center_y))

        angular_width = self.angular_width
        omega = self.omega_ss

        sigma_rad = np.deg2rad(self.sun_angular_diameter)
        penumbra_width = sigma_rad

        # Start of full umbra
        hu = (angular_width - penumbra_width) / 2.0
        # Start of penumbra
        hw = (angular_width + penumbra_width) / 2.0

        shadow_factor = np.ones(hit_points.shape[0])

        for i in range(self.N_ss):
            ss_center_theta = (i + 0.5) * (2.0 * np.pi / self.N_ss) + omega * time_sec

            d_theta = (theta - ss_center_theta + np.pi) % (2.0 * np.pi) - np.pi
            abs_d_theta = np.abs(d_theta)

            # Full shadow
            shadow_factor[abs_d_theta < hu] = 0.0

            # Penumbra
            p_mask = (abs_d_theta >= hu) & (abs_d_theta < hw)
            if np.any(p_mask):
                factor = (abs_d_theta[p_mask] - hu) / (hw - hu)
                shadow_factor[p_mask] = np.minimum(shadow_factor[p_mask], factor)

        return shadow_factor[0] if is_single else shadow_factor
