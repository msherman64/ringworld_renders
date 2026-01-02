"""
Ray-geometry intersection calculations for the Ringworld renderer.

This module contains all intersection solvers for rays against geometric primitives
used in the Ringworld scene: cylinder (ring), sphere (sun), shadow squares, rim walls.
"""
import numpy as np
from ringworld_renders import constants


def solve_quadratic_vectorized(a, b, c):
    """
    Solve at^2 + bt + c = 0 for vectorized arrays.

    Args:
        a, b, c: Arrays of quadratic coefficients

    Returns:
        tuple: (t1, t2, valid_mask) where t1, t2 are roots and valid_mask indicates valid solutions
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


def intersect_ring(ray_origin, ray_directions, R, h, W):
    """
    Vectorized stable intersection solver for ray and Ring (cylinder).

    Args:
        ray_origin: (3,) origin point
        ray_directions: (N, 3) array of ray directions
        R: Ring radius (meters)
        h: Observer eye height (meters)
        W: Ring width (meters)

    Returns:
        Array of intersection distances (inf where no intersection)
    """
    # Handle single vs batch
    is_single = ray_directions.ndim == 1
    if is_single:
        ray_directions = ray_directions[None, :]

    dx = ray_directions[:, 0]
    dy = ray_directions[:, 1]
    ox, oy, oz = ray_origin

    R_minus_h = R - h

    # a, b, c for quadratic equation at^2 + bt + c = 0
    a = dx**2 + dy**2
    b = 2.0 * (ox*dx + oy*dy - dy * R_minus_h)
    c = ox**2 + oy**2 - 2.0 * oy * R_minus_h - h * (2.0 * R - h)

    t1, t2, valid_mask = solve_quadratic_vectorized(a, b, c)

    t = np.full(ray_directions.shape[0], np.inf)

    if np.any(valid_mask):
        # Select smallest positive t
        best_t = np.full(t1.shape, np.inf)
        m1 = t1 > 1e-6
        best_t[m1] = t1[m1]
        m2 = (t2 > 1e-6) & (t2 < best_t)
        best_t[m2] = t2[m2]

        # Check width
        hit_z = ray_origin[2] + best_t * ray_directions[:, 2]
        width_mask = np.abs(hit_z) <= W / 2.0

        final_t = np.full(t1.shape, np.inf)
        final_t[width_mask] = best_t[width_mask]

        t = final_t

    return t[0] if is_single else t


def intersect_sun(ray_origin, ray_directions, sun_center, R_sun):
    """
    Vectorized intersection solver for ray and Sun (sphere).

    Args:
        ray_origin: (3,) origin point
        ray_directions: (N, 3) array of ray directions
        sun_center: (3,) center of sun sphere
        R_sun: Sun radius (meters)

    Returns:
        Array of intersection distances (inf where no intersection)
    """
    # Handle single vs batch
    is_single = ray_directions.ndim == 1
    if is_single:
        ray_directions = ray_directions[None, :]

    oc = ray_origin - sun_center

    # Quadratic equation coefficients
    a = np.sum(ray_directions**2, axis=1)
    b = 2.0 * np.sum(oc * ray_directions, axis=1)
    c = np.sum(oc**2) - R_sun**2

    t1, _, valid_mask = solve_quadratic_vectorized(a, b, c)

    t = np.full(ray_directions.shape[0], np.inf)

    if np.any(valid_mask):
        m = t1 > 1e-6
        t_valid = np.full(t1.shape, np.inf)
        t_valid[m] = t1[m]
        t = t_valid

    return t[0] if is_single else t


def intersect_rim_walls(ray_origin, ray_directions, R, h, W, H_wall):
    """
    Vectorized intersection solver for ray and Rim Walls.

    Args:
        ray_origin: (3,) origin point
        ray_directions: (N, 3) array of ray directions
        R: Ring radius (meters)
        h: Observer eye height (meters)
        W: Ring width (meters)
        H_wall: Rim wall height (meters)

    Returns:
        Array of intersection distances (inf where no intersection)
    """
    # Handle single vs batch
    is_single = ray_directions.ndim == 1
    if is_single:
        ray_directions = ray_directions[None, :]

    dx = ray_directions[:, 0]
    dy = ray_directions[:, 1]
    dz = ray_directions[:, 2]
    ox, oy, oz = ray_origin

    center_y = R - h

    t = np.full(ray_directions.shape[0], np.inf)

    # Check intersection with each rim wall (left and right)
    for wall_z in [-W/2.0, W/2.0]:
        # Plane equation: z = wall_z
        # Ray: origin + t * direction = point
        # So: oz + t * dz = wall_z
        # t = (wall_z - oz) / dz

        # Avoid division by zero (ray parallel to walls)
        valid_dz = np.abs(dz) > 1e-12
        t_wall = np.full(dz.shape, np.inf)
        t_wall[valid_dz] = (wall_z - oz) / dz[valid_dz]

        # Check if intersection is within wall bounds
        if np.any(valid_dz):
            # Only compute hit positions for finite t_wall values
            finite_t = np.isfinite(t_wall) & (t_wall > 1e-6)
            hit_x = np.full_like(t_wall, 0.0)
            hit_y = np.full_like(t_wall, 0.0)
            if np.any(finite_t):
                hit_x[finite_t] = ox + t_wall[finite_t] * dx[finite_t]
                hit_y[finite_t] = oy + t_wall[finite_t] * dy[finite_t]

            # Wall exists where radial distance from center is between R - H_wall and R
            r_sq = hit_x**2 + (hit_y - center_y)**2
            wall_mask = finite_t & (r_sq >= (R - H_wall)**2) & (r_sq <= R**2)

            # Update minimum t where wall intersection is valid
            update_mask = wall_mask & (t_wall < t)
            t[update_mask] = t_wall[update_mask]

    return t[0] if is_single else t
