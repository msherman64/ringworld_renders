"""
Intersection calculation module for Ringworld Renderer.

Handles ray intersections with geometric primitives: rings, suns, shadow squares, and rim walls.
"""

import numpy as np
from ringworld_renders import constants


def solve_quadratic_vectorized(a, b, c):
    """
    Solve at^2 + bt + c = 0 for vectorized arrays.
    Returns: t1, t2, valid_mask
    """
    discriminant = b**2 - 4.0 * a * c
    valid_mask = (a > 1e-12) & (discriminant >= 0)

    t1 = np.full(a.shape, np.inf)
    t2 = np.full(a.shape, np.inf)

    if np.any(valid_mask):
        sqrt_disc = np.sqrt(discriminant[valid_mask])
        inv_2a = 0.5 / a[valid_mask]
        t1[valid_mask] = (-b[valid_mask] - sqrt_disc) * inv_2a
        t2[valid_mask] = (-b[valid_mask] + sqrt_disc) * inv_2a

    return t1, t2, valid_mask


class Intersector:
    """
    Handles ray intersections with Ringworld geometric primitives.

    Provides methods for intersecting rays with the ring cylinder, sun sphere,
    shadow squares, and rim walls.
    """

    def __init__(self, R, h, W, R_sun, R_ss, L_ss, H_ss, N_ss, center_y):
        """
        Initialize the intersector with physical parameters.

        Args:
            R: Ring radius (meters)
            h: Observer height (meters)
            W: Ring width (meters)
            R_sun: Sun radius (meters)
            R_ss: Shadow square orbital radius (meters)
            L_ss: Shadow square length (meters)
            H_ss: Shadow square height (meters)
            N_ss: Number of shadow squares
            center_y: Ring center Y coordinate in observer frame (meters)
        """
        self.R = R
        self.h = h
        self.W = W
        self.R_sun = R_sun
        self.R_ss = R_ss
        self.L_ss = L_ss
        self.H_ss = H_ss
        self.N_ss = N_ss
        self.center_y = center_y

        # Derived constants
        self.r_inner_limit = R - constants.RIM_WALL_HEIGHT_METERS

    def intersect_ring(self, ray_origin, ray_directions):
        """
        Vectorized stable intersection solver for ray and Ring (cylinder).
        """
        # Handle single vs batch
        is_single = ray_directions.ndim == 1
        if is_single:
            ray_directions = ray_directions[None, :]

        dx = ray_directions[:, 0]
        dy = ray_directions[:, 1]
        ox, oy, oz = ray_origin

        R = self.R
        R_minus_h = R - self.h
        h = self.h

        # a, b, c for quadratic equation at^2 + bt + c = 0
        a = dx**2 + dy**2
        b = 2.0 * ox * dx + 2.0 * (oy - R_minus_h) * dy
        c = ox**2 + (oy - R_minus_h)**2 - R**2

        # Delta-R solver: (R-h)^2 - R^2 = -2Rh + h^2 = -h(2R - h)
        # This prevents catastrophic cancellation for h << R
        if h > 1e-6:
            c = -h * (2.0 * R - h)

        t1, t2, valid_mask = solve_quadratic_vectorized(a, b, c)

        t = np.full(ray_directions.shape[0], np.inf)

        if np.any(valid_mask):
            # Select smallest positive t
            best_t = np.full(t1.shape, np.inf)
            m1 = t1 > 1e-6
            best_t[m1] = t1[m1]
            m2 = (t2 > 1e-6) & (t2 < best_t)
            best_t[m2] = t2[m2]

            # Check z bounds (ring width)
            hit_mask = best_t < np.inf
            if np.any(hit_mask):
                valid_hits = best_t[hit_mask]
                dirs = ray_directions[hit_mask]
                hit_z = oz + valid_hits * dirs[:, 2]

                # Ring width bounds
                z_valid = np.abs(hit_z) <= self.W / 2.0
                t[hit_mask] = np.where(z_valid, valid_hits, np.inf)

        return t[0] if is_single else t

    def intersect_sun(self, ray_origin, ray_directions):
        """
        Vectorized intersection with the Sun sphere.
        """
        # Handle single vs batch
        is_single = ray_directions.ndim == 1
        if is_single:
            ray_directions = ray_directions[None, :]

        dx = ray_directions[:, 0]
        dy = ray_directions[:, 1]
        dz = ray_directions[:, 2]
        ox, oy, oz = ray_origin

        # Intersection with sphere at (0, center_y, 0) with radius R_sun
        a = dx**2 + dy**2 + dz**2
        b = 2.0 * (ox * dx + (oy - self.center_y) * dy + oz * dz)
        c = ox**2 + (oy - self.center_y)**2 + oz**2 - self.R_sun**2

        t1, t2, valid_mask = solve_quadratic_vectorized(a, b, c)

        t = np.full(ray_directions.shape[0], np.inf)

        if np.any(valid_mask):
            # Select smallest positive t
            best_t = np.full(t1.shape, np.inf)
            m1 = t1 > 1e-6
            best_t[m1] = t1[m1]
            m2 = (t2 > 1e-6) & (t2 < best_t)
            best_t[m2] = t2[m2]

            t = np.where(best_t < np.inf, best_t, np.inf)

        return t[0] if is_single else t

    def intersect_shadow_squares(self, ray_origin, ray_directions, time_sec, shadows):
        """
        Vectorized intersection with Shadow Squares (cylinders at R_ss).
        """
        # Handle single vs batch
        is_single = ray_directions.ndim == 1
        if is_single:
            ray_directions = ray_directions[None, :]

        dx = ray_directions[:, 0]
        dy = ray_directions[:, 1]
        ox, oy, oz = ray_origin

        R_ss = self.R_ss
        center_y = self.center_y

        # Intersection with cylinder (r = R_ss)
        a = dx**2 + dy**2
        b = 2.0 * (ox*dx + (oy - center_y)*dy)
        c = ox**2 + (oy - center_y)**2 - R_ss**2

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
                z_check = np.abs(hit_p[:, 2]) <= self.H_ss / 2.0

                # Check angular position
                theta = np.arctan2(hit_p[:, 0], -(hit_p[:, 1] - center_y))

                half_width = shadows.angular_width_val / 2.0
                omega = shadows.omega_ss_val

                in_any_square = np.zeros(hit_p.shape[0], dtype=bool)
                for i in range(self.N_ss):
                    theta_center = (i + 0.5) * (2.0 * np.pi / self.N_ss) + omega * time_sec
                    d_theta = (theta - theta_center + np.pi) % (2.0 * np.pi) - np.pi
                    in_square = np.abs(d_theta) <= half_width
                    in_any_square |= in_square

                # Combine checks
                valid_squares = z_check & in_any_square
                t[hit_mask] = np.where(valid_squares, valid_hits, np.inf)

        return t[0] if is_single else t

    def intersect_rim_walls(self, ray_origin, ray_directions):
        """
        Vectorized intersection with Rim Walls (planes at Z = +/- W/2).
        Walls exist between R - H_w and R.
        """
        is_single = ray_directions.ndim == 1
        if is_single:
            ray_directions = ray_directions[None, :]

        dx, dy, dz = ray_directions[:, 0], ray_directions[:, 1], ray_directions[:, 2]
        ox, oy, oz = ray_origin

        # Two planes: z = W/2 (Positive) and z = -W/2 (Negative)
        t = np.full(ray_directions.shape[0], np.inf)

        # We need to handle dz near 0
        valid_dz = np.abs(dz) > 1e-12

        if np.any(valid_dz):
            # Calculate t for both planes
            with np.errstate(divide='ignore'):
                t_pos = (self.W / 2.0 - oz) / dz
                t_neg = (-self.W / 2.0 - oz) / dz

            # Candidate t
            t_cand = np.full_like(t, np.inf)

            # Positive wall candidates (dz > 0 hits Pos wall)
            mask_pos = valid_dz & (dz > 0)
            if np.any(mask_pos):
                tp = t_pos[mask_pos]
                # Check radius
                t_sub = tp
                dx_s, dy_s = dx[mask_pos], dy[mask_pos]
                hx = ox + t_sub * dx_s
                hy = oy + t_sub * dy_s

                r_sq = hx**2 + (hy - self.center_y)**2
                valid_r = (r_sq <= self.R**2) & (r_sq >= self.r_inner_limit**2)

                subset_t = t_cand[mask_pos]
                subset_t[valid_r] = tp[valid_r]
                t_cand[mask_pos] = subset_t

            # Negative wall candidates (dz < 0 hits Neg wall)
            mask_neg = valid_dz & (dz < 0)
            if np.any(mask_neg):
                tn = t_neg[mask_neg]
                t_sub = tn
                dx_s, dy_s = dx[mask_neg], dy[mask_neg]
                hx = ox + t_sub * dx_s
                hy = oy + t_sub * dy_s

                r_sq = hx**2 + (hy - self.center_y)**2
                valid_r = (r_sq <= self.R**2) & (r_sq >= self.r_inner_limit**2)

                subset_t = t_cand[mask_neg]
                subset_t[valid_r] = tn[valid_r]
                t_cand[mask_neg] = subset_t

            # Select smallest positive t
            valid_candidates = t_cand < np.inf
            if np.any(valid_candidates):
                t = np.where(valid_candidates, t_cand, np.inf)

        return t[0] if is_single else t
