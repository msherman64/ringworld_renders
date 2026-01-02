"""
Data structures and interfaces for the Ringworld rendering pipeline.
"""
from dataclasses import dataclass
import numpy as np
from enum import Enum
from ringworld_renders import constants


class HitType(Enum):
    """Enumeration of possible intersection types."""
    SUN = "sun"
    RING = "ring"
    RIM_WALL = "rim_wall"
    SHADOW_SQUARE = "shadow_square"
    SKY = "sky"


@dataclass
class HitResult:
    """
    Result of ray intersection calculations.

    Attributes:
        distance: Distance to hit point (N,) shape array
        hit_type: Type of surface hit (N,) HitType enum values
        hit_point: 3D coordinates of hit point (N, 3) shape
        surface_normal: Surface normal at hit point (N, 3) shape
    """
    distance: np.ndarray  # (N,) shape
    hit_type: np.ndarray  # (N,) HitType values
    hit_point: np.ndarray  # (N, 3) shape
    surface_normal: np.ndarray | None = None  # (N, 3) shape, optional for now

    def __post_init__(self):
        """Validate array shapes and types."""
        if self.distance.ndim != 1:
            raise ValueError(f"distance must be 1D array, got shape {self.distance.shape}")
        if self.hit_type.ndim != 1:
            raise ValueError(f"hit_type must be 1D array, got shape {self.hit_type.shape}")
        if self.hit_point.ndim != 2 or self.hit_point.shape[1] != 3:
            raise ValueError(f"hit_point must be (N,3) array, got shape {self.hit_point.shape}")

        n_rays = self.distance.shape[0]
        if self.hit_type.shape[0] != n_rays:
            raise ValueError(f"hit_type shape {self.hit_type.shape} doesn't match distance shape {self.distance.shape}")
        if self.hit_point.shape[0] != n_rays:
            raise ValueError(f"hit_point shape {self.hit_point.shape} doesn't match distance shape {self.distance.shape}")

        # Validate hit_type values
        valid_types = {ht.value for ht in HitType}
        unique_types = set(self.hit_type)
        invalid_types = unique_types - valid_types
        if invalid_types:
            raise ValueError(f"Invalid hit types: {invalid_types}. Valid types: {valid_types}")


class HitSelector:
    """
    Responsible for determining which surface each ray hits first.

    This encapsulates the intersection logic and priority ordering from the original
    get_color() method.
    """

    def __init__(self, renderer):
        """
        Initialize with a reference to the renderer for access to intersection methods.

        Args:
            renderer: Renderer instance with intersection methods
        """
        self.renderer = renderer

    def select_primary(self, ray_origin: np.ndarray, ray_directions: np.ndarray,
                      time_sec: float) -> HitResult:
        """
        Find the primary intersection for each ray.

        Args:
            ray_origin: (3,) origin point (typically observer position)
            ray_directions: (N, 3) array of ray directions
            time_sec: Current time in seconds for shadow square positioning

        Returns:
            HitResult with primary intersections for all rays
        """
        is_single = ray_directions.ndim == 1
        if is_single:
            ray_directions = ray_directions[None, :]

        n_rays = ray_directions.shape[0]

        # Perform all intersection tests
        t_sun = self.renderer.intersect_sun(ray_origin, ray_directions)
        t_ring = self.renderer.intersect_ring(ray_origin, ray_directions)
        t_wall = self.renderer.intersect_rim_walls(ray_origin, ray_directions)
        t_ss = self.renderer.intersect_shadow_squares(ray_origin, ray_directions, time_sec)

        # Find primary hit for each ray (minimum finite distance)
        t_candidates = np.stack([t_sun, t_ring, t_ss, t_wall], axis=1)  # (N, 4)
        t_min = np.min(np.where(np.isfinite(t_candidates), t_candidates, np.inf), axis=1)

        # Determine hit types - use object dtype to preserve full enum values
        hit_types = np.full(n_rays, HitType.SKY.value, dtype=object)
        hit_types[t_sun == t_min] = HitType.SUN.value
        hit_types[t_ring == t_min] = HitType.RING.value
        hit_types[t_wall == t_min] = HitType.RIM_WALL.value
        hit_types[t_ss == t_min] = HitType.SHADOW_SQUARE.value

        # Calculate hit points
        hit_points = np.full((n_rays, 3), np.nan)
        valid_hits = np.isfinite(t_min)
        if np.any(valid_hits):
            hit_points[valid_hits] = (ray_origin[None, :] +
                                     t_min[valid_hits, None] * ray_directions[valid_hits])

        # Convert to HitResult
        result = HitResult(
            distance=t_min,
            hit_type=hit_types,
            hit_point=hit_points
        )

        return result


class MaterialSystem:
    """
    Responsible for determining surface colors based on hit results.

    This encapsulates the material/shading logic from the original get_color() method.
    """

    def __init__(self, renderer):
        """
        Initialize with a reference to the renderer for access to shadow calculations.

        Args:
            renderer: Renderer instance with shadow calculation methods
        """
        self.renderer = renderer

    def get_surface_color(self, hits: HitResult, time_sec: float,
                         use_shadows: bool = True, use_ring_shine: bool = True,
                         debug_shadow_squares: bool = False) -> np.ndarray:
        """
        Calculate surface colors for all hits.

        Args:
            hits: HitResult from intersection calculations
            time_sec: Current time for shadow and ring-shine calculations
            use_shadows: Whether to apply shadow factors
            use_ring_shine: Whether to apply ring-shine effect
            debug_shadow_squares: Whether to use debug colors for shadow squares

        Returns:
            (N, 3) array of RGB surface colors
        """
        n_rays = hits.distance.shape[0]
        surface_colors = np.zeros((n_rays, 3))

        # Sun material
        sun_mask = hits.hit_type == HitType.SUN.value
        if np.any(sun_mask):
            surface_colors[sun_mask] = np.array([1.0, 1.0, 0.8])

        # Shadow Square material
        ss_mask = hits.hit_type == HitType.SHADOW_SQUARE.value
        if np.any(ss_mask):
            if debug_shadow_squares:
                # False colors for shadow squares
                valid_hits = hits.hit_point[ss_mask]
                center_y = self.renderer.center_y
                theta = np.arctan2(valid_hits[:, 0], -(valid_hits[:, 1] - center_y))

                omega = self.renderer.omega_ss
                half_width = self.renderer.angular_width / 2.0

                ss_colors = np.zeros((np.sum(ss_mask), 3))
                for i in range(self.renderer.N_ss):
                    ss_center_theta = (i + 0.5) * (2.0 * np.pi / self.renderer.N_ss) + omega * time_sec
                    d_theta = (theta - ss_center_theta + np.pi) % (2.0 * np.pi) - np.pi

                    # Check if in this square (with tolerance)
                    mask_in_sq = np.abs(d_theta) <= (half_width * 1.5)
                    col = np.array(constants.SS_COLORS[i % len(constants.SS_COLORS)])
                    ss_colors[mask_in_sq] = col

                surface_colors[ss_mask] = ss_colors
            else:
                # Shadow squares are opaque black
                surface_colors[ss_mask] = np.array([0.0, 0.0, 0.0])

        # Rim Wall material
        wall_mask = hits.hit_type == HitType.RIM_WALL.value
        if np.any(wall_mask):
            surface_colors[wall_mask] = np.array([0.1, 0.1, 0.15])

        # Ring material
        ring_mask = hits.hit_type == HitType.RING.value
        if np.any(ring_mask):
            # Dynamic Ring-shine: peaking at midnight when the arch is overhead
            time_angle = 2.0 * np.pi * time_sec / constants.SOLAR_DAY_SECONDS
            # Scale from 0.02 (noon) to 0.10 (midnight)
            ambient_shine = 0.06 - 0.04 * np.cos(time_angle) if use_ring_shine else 0.0

            s_factor = np.ones(np.sum(ring_mask))
            if use_shadows:
                valid_hits = hits.hit_point[ring_mask]
                s_factor = self.renderer.get_shadow_factor(valid_hits, time_sec)
                s_factor = np.atleast_1d(s_factor)

            # Mock green surface with physical light interaction
            base_color = np.array([0.2, 0.5, 0.2]) * (s_factor + ambient_shine)[:, None]
            surface_colors[ring_mask] = base_color

        return surface_colors


class AtmosphericModel:
    """
    Responsible for atmospheric effects: transmittance and in-scattering.

    This encapsulates the complex atmospheric physics from get_atmospheric_effects().
    """

    def __init__(self, renderer):
        """
        Initialize with renderer parameters needed for atmospheric calculations.

        Args:
            renderer: Renderer instance with atmospheric parameters
        """
        self.renderer = renderer

        # Cache frequently used atmospheric parameters
        self.H_a = renderer.H_a  # 100 mile Effective Ceiling
        self.H_scale = renderer.H_scale  # Scale height for 1G
        self.tau_zenith = renderer.tau_zenith  # Rayleigh spectral scaling
        self.sky_color = renderer.sky_color  # Typical Rayleigh blue

    def get_atmospheric_effects(self, t_hits: np.ndarray, ray_origin: np.ndarray,
                               ray_directions: np.ndarray, time_sec: float = 0.0,
                               use_shadows: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate atmospheric transmittance and in-scattering.

        Args:
            t_hits: Distance to surface hits (or large value for sky)
            ray_origin: Ray origin point
            ray_directions: Ray directions (N, 3)
            time_sec: Current time for shadow calculations
            use_shadows: Whether to apply shadows to scattering

        Returns:
            Tuple of (transmittance, in_scattering) arrays, both shape (N, 3)
        """
        is_single = ray_directions.ndim == 1
        if is_single:
            ray_directions = ray_directions[None, :]
            t_hits = np.atleast_1d(t_hits)

        dx, dy, dz = ray_directions[:, 0], ray_directions[:, 1], ray_directions[:, 2]
        ox, oy, oz = ray_origin

        # Calculate starting height relative to Ring Floor
        yc = self.renderer.ring_center[1]
        dist_sq_from_center_axis = ox**2 + (oy - yc)**2
        r_start = np.sqrt(dist_sq_from_center_axis)
        h_start = self.renderer.R - r_start

        R_minus_h = self.renderer.R - self.renderer.h

        # 1. Geometric Boundaries
        # Radial boundaries: inner (R-Ha) and outer (R)
        oy_prime = oy - R_minus_h
        a = dx**2 + dy**2
        b = 2.0 * (ox * dx + oy_prime * dy)
        c_term = ox**2 + oy_prime**2

        # Inner cylinder (R - Ha)
        r_inner = self.renderer.R - self.H_a
        c_inner = c_term - r_inner**2

        t_i1, t_i2, mask_i = self.renderer._solve_quadratic_vectorized(a, b, c_inner)

        if not np.any(mask_i):
             t_i1 = np.full_like(t_hits, np.inf)
             t_i2 = np.full_like(t_hits, np.inf)

        # Width boundary (Rim Walls at +/- W/2)
        t_z_exit = np.full_like(t_hits, np.inf)

        # Exit through width boundaries
        term = np.zeros_like(dz)
        mask_pos = dz > 1e-12
        mask_neg = dz < -1e-12

        term[mask_pos] = (self.renderer.W / 2.0 - oz) / dz[mask_pos]
        term[mask_neg] = (-self.renderer.W / 2.0 - oz) / dz[mask_neg]

        t_z_exit[mask_pos | mask_neg] = term[mask_pos | mask_neg]

        # 2. Segment Identification
        looking_up = dy > 0
        seg_near_end = np.minimum(t_hits, t_z_exit)
        seg_near_end[looking_up] = np.minimum(seg_near_end[looking_up], t_i1[looking_up])

        # Far segment: enters at t_i2, exits at t_hit, clipped by Z
        far_mask = looking_up & (t_hits > t_i2) & np.isfinite(t_i2)
        if np.any(far_mask):
            z_at_i2 = oz + t_i2[far_mask] * dz[far_mask]
            within_z = np.abs(z_at_i2) <= self.renderer.W / 2.0
            updated_mask = far_mask.copy()
            updated_mask[far_mask] = within_z
            far_mask = updated_mask

        # 3. Analytical Exponential Integration
        def integrated_tau(h_start, L, cos_theta):
            H = self.H_scale
            h_end = h_start + L * cos_theta

            with np.errstate(divide='ignore', invalid='ignore'):
                v1 = -h_start / H
                v2 = -h_end / H
                v1 = np.clip(v1, -700, 700)
                v2 = np.clip(v2, -700, 700)

                term = np.where(np.abs(cos_theta) < 1e-6,
                               np.exp(v1) * (L / H),
                               (np.exp(v1) - np.exp(v2)) / cos_theta)
            return term

        # NEAR SEGMENT Optical Depth
        itau = integrated_tau(h_start, seg_near_end, dy)
        tau_near = self.tau_zenith[None, :] * itau[:, None]
        T_near = np.exp(-tau_near)

        # FAR SEGMENT Optical Depth
        tau_far = np.zeros_like(tau_near)
        if np.any(far_mask):
            p_far = t_i2[far_mask][:, None] * ray_directions[far_mask]
            vec_to_center = self.renderer.ring_center - p_far
            up_far = vec_to_center / np.linalg.norm(vec_to_center, axis=1, keepdims=True)
            cos_theta_far = np.sum(ray_directions[far_mask] * up_far, axis=1)

            l_far = np.minimum(t_hits[far_mask], t_z_exit[far_mask]) - t_i2[far_mask]
            l_far = np.maximum(l_far, 0.0)

            # Cap far-side path by reaching the ground (h=0)
            m_down = cos_theta_far < -1e-6
            if np.any(m_down):
                l_limit = -self.H_a / cos_theta_far[m_down]
                l_far[m_down] = np.minimum(l_far[m_down], l_limit)

            tau_far[far_mask] = self.tau_zenith[None, :] * integrated_tau(self.H_a, l_far, cos_theta_far)[:, None]

        T_far = np.exp(-tau_far)

        # 4. Phase Function and Shadowing
        sun_dir = self.renderer.ring_center / np.linalg.norm(self.renderer.ring_center)
        cos_theta = np.sum(ray_directions * sun_dir[None, :], axis=1)
        phase = 0.75 * (1.0 + cos_theta**2)

        # Shadowing: Sample at scale height
        s_near = np.ones(tau_near.shape[0])
        if use_shadows:
            p_near = (self.H_scale / np.maximum(np.abs(dy), 0.1))[:, None] * ray_directions
            s_near = np.atleast_1d(self.renderer.get_shadow_factor(p_near, time_sec))

        scat_near = self.sky_color[None, :] * (1.0 - T_near) * (s_near + 0.1)[:, None] * phase[:, None]

        scat_far = np.zeros_like(scat_near)
        if np.any(far_mask):
            t_mid_far = t_i2[far_mask] + l_far / 2.0
            p_far = t_mid_far[:, None] * ray_directions[far_mask]
            s_far = np.atleast_1d(self.renderer.get_shadow_factor(p_far, time_sec))
            scat_far[far_mask] = self.sky_color[None, :] * (1.0 - T_far[far_mask]) * (s_far + 0.1)[:, None] * phase[far_mask, None]

        total_scat = scat_near + (scat_far * T_near)
        total_trans = T_near * T_far

        if is_single:
            return total_trans[0, :], total_scat[0, :]
        return total_trans, total_scat
