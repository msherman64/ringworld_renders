"""
Rendering pipeline module for Ringworld Renderer.

This module provides the core rendering pipeline components, including
hit detection, material shading, and atmospheric effects.
"""

from dataclasses import dataclass
from enum import IntEnum
import numpy as np
from ringworld_renders import constants
from ringworld_renders.intersections import solve_quadratic_vectorized


class HitType(IntEnum):
    """Enumeration of possible ray hit types."""
    SUN = 0
    RING = 1
    WALL = 2
    SHADOW_SQUARE = 3
    SKY = 4


@dataclass
class HitResult:
    """
    Result of ray intersection calculations.
    
    All arrays have shape (N,) or (N, 3) where N is the number of rays.
    """
    distance: np.ndarray  # (N,) shape - distance to hit point
    hit_type: np.ndarray  # (N,) HitType enum values
    hit_point: np.ndarray  # (N, 3) shape - 3D position of hit
    surface_normal: np.ndarray  # (N, 3) shape - surface normal at hit (for future use)
    
    def __post_init__(self):
        """Validate array shapes after initialization."""
        n = self.distance.shape[0]
        assert self.hit_type.shape == (n,), f"hit_type shape mismatch: {self.hit_type.shape} != ({n},)"
        assert self.hit_point.shape == (n, 3), f"hit_point shape mismatch: {self.hit_point.shape} != ({n}, 3)"
        assert self.surface_normal.shape == (n, 3), f"surface_normal shape mismatch: {self.surface_normal.shape} != ({n}, 3)"


class HitSelector:
    """
    Selects the primary hit from multiple intersection results.
    
    Priority order: SUN > RING > WALL > SHADOW_SQUARE > SKY
    """
    
    def __init__(self, ring_center):
        """
        Initialize the hit selector.
        
        Args:
            ring_center: np.array([x, y, z]) - Center of the ring in observer coordinates
        """
        self.ring_center = ring_center
    
    def select_primary(self, ray_origin, ray_directions, t_sun, t_ring, t_wall, t_ss):
        """
        Select the primary hit from intersection distances.

        Args:
            ray_origin: np.array([x, y, z]) - Origin of all rays
            ray_directions: np.ndarray of shape (N, 3) - Direction vectors for N rays
            t_sun: np.ndarray of shape (N,) - Distances to sun intersections (inf if no hit)
            t_ring: np.ndarray of shape (N,) - Distances to ring intersections (inf if no hit)
            t_wall: np.ndarray of shape (N,) - Distances to wall intersections (inf if no hit)
            t_ss: np.ndarray of shape (N,) - Distances to shadow square intersections (inf if no hit)

        Returns:
            HitResult with distance, hit_type, hit_point, and surface_normal
        """
        num_rays = ray_directions.shape[0]
        
        # Find primary hit with priority: sun > ring > wall > shadow square
        hit_t = np.minimum(t_sun, np.minimum(t_ring, np.minimum(t_ss, t_wall)))
        
        # Determine hit type with priority ordering
        hit_is_sun = (t_sun <= hit_t) & (t_sun < np.inf)
        hit_is_ring = (t_ring <= hit_t) & (t_ring < np.inf) & ~hit_is_sun
        hit_is_wall = (t_wall <= hit_t) & (t_wall < np.inf) & ~hit_is_sun & ~hit_is_ring
        hit_is_ss = (t_ss <= hit_t) & (t_ss < np.inf) & ~hit_is_sun & ~hit_is_ring & ~hit_is_wall
        hit_is_sky = hit_t == np.inf
        
        # Convert to HitType enum values
        hit_type = np.full(num_rays, HitType.SKY, dtype=np.int32)
        hit_type[hit_is_sun] = HitType.SUN
        hit_type[hit_is_ring] = HitType.RING
        hit_type[hit_is_wall] = HitType.WALL
        hit_type[hit_is_ss] = HitType.SHADOW_SQUARE
        
        # Calculate hit points
        hit_point = np.zeros((num_rays, 3))
        finite_mask = np.isfinite(hit_t)
        if np.any(finite_mask):
            hit_point[finite_mask] = ray_origin[None, :] + hit_t[finite_mask, None] * ray_directions[finite_mask]
        # For infinite distances (sky rays), hit_point remains zero (undefined but safe)

        # Calculate surface normals (placeholder for now - will be properly calculated in future)
        # For now, set to zeros as they're marked "for future use"
        surface_normal = np.zeros((num_rays, 3))
        
        return HitResult(
            distance=hit_t,
            hit_type=hit_type,
            hit_point=hit_point,
            surface_normal=surface_normal
        )


class MaterialSystem:
    """
    Handles surface material shading for different hit types.
    
    Provides methods to calculate surface colors based on hit type,
    shadow factors, and material properties.
    """
    
    def __init__(self, shadow_model, ring_center, center_y, N_ss):
        """
        Initialize the material system.
        
        Args:
            shadow_model: ShadowModel instance for shadow factor calculations
            ring_center: np.array([x, y, z]) - Center of the ring
            center_y: float - Y coordinate of ring center
            N_ss: int - Number of shadow squares
        """
        self.shadow_model = shadow_model
        self.ring_center = ring_center
        self.center_y = center_y
        self.N_ss = N_ss
    
    def get_surface_color(self, hit_result, ray_origin, ray_directions, time_sec=0.0,
                          use_shadows=True, use_ring_shine=True, debug_shadow_squares=False):
        """
        Calculate surface colors for all hits.
        
        Args:
            hit_result: HitResult with distance, hit_type, hit_point, surface_normal
            ray_origin: np.array([x, y, z]) - Origin of rays
            ray_directions: np.ndarray of shape (N, 3) - Direction vectors
            time_sec: float - Current time in seconds
            use_shadows: bool - Whether to apply shadow factors
            use_ring_shine: bool - Whether to apply ring-shine ambient
            debug_shadow_squares: bool - Whether to use debug colors for shadow squares
        
        Returns:
            np.ndarray of shape (N, 3) - Surface colors for each ray
        """
        num_rays = hit_result.distance.shape[0]
        surface_colors = np.zeros((num_rays, 3))
        
        # Handle each hit type
        sun_mask = hit_result.hit_type == HitType.SUN
        if np.any(sun_mask):
            surface_colors[sun_mask] = self._get_sun_color()
        
        ring_mask = hit_result.hit_type == HitType.RING
        if np.any(ring_mask):
            surface_colors[ring_mask] = self._get_ring_color(
                hit_result.hit_point[ring_mask], time_sec, use_shadows, use_ring_shine
            )
        
        wall_mask = hit_result.hit_type == HitType.WALL
        if np.any(wall_mask):
            surface_colors[wall_mask] = self._get_wall_color()
        
        ss_mask = hit_result.hit_type == HitType.SHADOW_SQUARE
        if np.any(ss_mask):
            surface_colors[ss_mask] = self._get_shadow_square_color(
                hit_result.hit_point[ss_mask], time_sec, debug_shadow_squares
            )
        
        # Sky has no surface color (already zero)
        
        return surface_colors
    
    def _get_sun_color(self):
        """Get color for sun hits."""
        return np.array(constants.SUN_COLOR)

    def _get_wall_color(self):
        """Get color for rim wall hits."""
        return np.array(constants.WALL_COLOR)
    
    def _get_ring_color(self, hit_points, time_sec, use_shadows, use_ring_shine):
        """
        Get color for ring hits with shadow and ring-shine effects.
        
        Args:
            hit_points: np.ndarray of shape (M, 3) - Hit points on ring
            time_sec: float - Current time
            use_shadows: bool - Whether to apply shadows
            use_ring_shine: bool - Whether to apply ring-shine
        """
        # Base color
        base_color = np.array(constants.RING_COLOR)

        # Shadow factor
        if use_shadows:
            s_factor = self.shadow_model.get_shadow_factor(hit_points, time_sec)
            s_factor = np.atleast_1d(s_factor)
        else:
            s_factor = np.ones(hit_points.shape[0])

        # Ring-shine ambient
        if use_ring_shine:
            time_angle = 2.0 * np.pi * time_sec / constants.SOLAR_DAY_SECONDS
            ambient_shine = constants.RING_SHINE_BASE - constants.RING_SHINE_AMPLITUDE * np.cos(time_angle)
        else:
            ambient_shine = 0.0

        # Apply shadow and ambient
        color = base_color * (s_factor + ambient_shine)[:, None]
        return color
    
    def _get_shadow_square_color(self, hit_points, time_sec, debug_shadow_squares):
        """
        Get color for shadow square hits.
        
        Args:
            hit_points: np.ndarray of shape (M, 3) - Hit points on shadow squares
            time_sec: float - Current time
            debug_shadow_squares: bool - Whether to use debug colors
        """
        if debug_shadow_squares:
            # False colors for debugging
            theta = np.arctan2(hit_points[:, 0], -(hit_points[:, 1] - self.center_y))
            
            omega = self.shadow_model.omega_ss_val
            half_width = self.shadow_model.angular_width_val / 2.0
            
            ss_colors_mapped = np.zeros((hit_points.shape[0], 3))
            
            for i in range(self.N_ss):
                ss_center_theta = (i + 0.5) * (2.0 * np.pi / self.N_ss) + omega * time_sec
                d_theta = (theta - ss_center_theta + np.pi) % (2.0 * np.pi) - np.pi
                
                # Check if in this square (with tolerance)
                mask_in_sq = np.abs(d_theta) <= (half_width * 1.5)
                col = np.array(constants.SS_COLORS[i % len(constants.SS_COLORS)])
                ss_colors_mapped[mask_in_sq] = col
            
            return ss_colors_mapped
        else:
            # Shadow squares are opaque and black (back side)
            return np.zeros((hit_points.shape[0], 3))


class AtmosphericModel:
    """
    Handles atmospheric effects including transmittance and in-scattering.
    
    Implements analytical exponential density integration for Rayleigh scattering.
    """
    
    def __init__(self, R, h, W, H_a, H_scale, tau_zenith, sky_color, ring_center, shadow_model):
        """
        Initialize the atmospheric model.
        
        Args:
            R: Ring radius (meters)
            h: Observer height (meters)
            W: Ring width (meters)
            H_a: Effective atmospheric ceiling (meters)
            H_scale: Scale height for exponential density (meters)
            tau_zenith: np.array([r, g, b]) - Optical depth at zenith for each channel
            sky_color: np.array([r, g, b]) - Sky scattering color
            ring_center: np.array([x, y, z]) - Center of the ring
            shadow_model: ShadowModel instance for shadow factor calculations
        """
        self.R = R
        self.h = h
        self.W = W
        self.H_a = H_a
        self.H_scale = H_scale
        self.tau_zenith = tau_zenith
        self.sky_color = sky_color
        self.ring_center = ring_center
        self.shadow_model = shadow_model
    
    def get_atmospheric_color(self, t_hits, ray_origin, ray_directions, time_sec=0.0, use_shadows=True):
        """
        Calculate atmospheric transmittance and in-scattering.

        Args:
            t_hits: np.ndarray of shape (N,) - Distances to hit points (or large value for sky)
            ray_origin: np.array([x, y, z]) - Origin of all rays
            ray_directions: np.ndarray of shape (N, 3) - Direction vectors for N rays
            time_sec: float - Current time in seconds
            use_shadows: bool - Whether to apply shadow factors to scattering

        Returns:
            tuple: (transmittance, in_scattering) both of shape (N, 3)
        """
            
        dx, dy, dz = ray_directions[:, 0], ray_directions[:, 1], ray_directions[:, 2]
        ox, oy, oz = ray_origin
        
        # Calculate starting height relative to Ring Floor for atmospheric density.
        # Ring Center is at self.ring_center (0, R-h, 0).
        # Radius r = distance(ray_origin, ring_center)
        # Height H = R - r
        yc = self.ring_center[1]
        dist_sq_from_center_axis = ox**2 + (oy - yc)**2
        r_start = np.sqrt(dist_sq_from_center_axis)
        h_start = self.R - r_start
        
        R_minus_h = self.R - self.h
        
        # 1. Geometric Boundaries
        # Radial boundaries: inner (R-Ha) and outer (R)
        # origin (ox, oy)
        # ray P(t) = O + tD
        # Intersection with cylinder |P(t) - C|^2 = r^2
        # C = (0, R-h)
        # (ox + t*dx)^2 + (oy + t*dy - (R-h))^2 = r^2
        
        oy_prime = oy - R_minus_h
        a = dx**2 + dy**2
        b = 2.0 * (ox * dx + oy_prime * dy)
        c_term = ox**2 + oy_prime**2
        
        # Inner cylinder (R - Ha)
        r_inner = self.R - self.H_a
        c_inner = c_term - r_inner**2
        
        t_i1, t_i2, mask_i = solve_quadratic_vectorized(a, b, c_inner)
        
        if not np.any(mask_i):
             t_i1 = np.full_like(t_hits, np.inf)
             t_i2 = np.full_like(t_hits, np.inf)

        # Width boundary (Rim Walls at +/- W/2)
        
        t_z_exit = np.full_like(t_hits, np.inf)
        
        # We only care about positive t that exits the volume.
        # t = (W/2 - oz) / dz if dz > 0
        # t = (-W/2 - oz) / dz if dz < 0
            
        term = np.zeros_like(dz)
        mask_pos = dz > 1e-12
        mask_neg = dz < -1e-12
        
        term[mask_pos] = (self.W / 2.0 - oz) / dz[mask_pos]
        term[mask_neg] = (-self.W / 2.0 - oz) / dz[mask_neg]
        
        t_z_exit[mask_pos | mask_neg] = term[mask_pos | mask_neg]


        # 2. Segment Identification
        looking_up = dy > 0
        seg_near_end = np.minimum(t_hits, t_z_exit)
        seg_near_end[looking_up] = np.minimum(seg_near_end[looking_up], t_i1[looking_up])
        
        # Far segment: enters at t_i2, exits at t_hit, clipped by Z
        # We ensure t_i2 is finite to avoid NaN in the width check
        far_mask = looking_up & (t_hits > t_i2) & np.isfinite(t_i2)
        if np.any(far_mask):
            z_at_i2 = oz + t_i2[far_mask] * dz[far_mask]
            # Use a temporary mask to avoid indexing errors
            within_z = np.abs(z_at_i2) <= self.W / 2.0
            updated_mask = far_mask.copy()
            updated_mask[far_mask] = within_z
            far_mask = updated_mask
        
        # 3. Analytical Exponential Integration
        def integrated_tau(h_start, L, cos_theta):
            # Integral of e^(-h(s)/H) ds from 0 to L
            # = [ e^(-h_start/H) - e^(-h_end/H) ] / (cos_theta/H)
            H = self.H_scale
            h_end = h_start + L * cos_theta
            
            with np.errstate(divide='ignore', invalid='ignore'):
                # Handle horizontal rays (cos_theta near 0)
                # For small cos_theta, it's roughly L * e^(-h_avg/H)
                # But the analytical form is stable if we use the right terms.
                v1 = -h_start / H
                v2 = -h_end / H
                # Cap exponents to prevent overflow
                v1 = np.clip(v1, -700, 700)
                v2 = np.clip(v2, -700, 700)
                
                term = np.where(np.abs(cos_theta) < 1e-6,
                                np.exp(v1) * (L / H),
                                (np.exp(v1) - np.exp(v2)) / cos_theta)
            return term

        # NEAR SEGMENT Optical Depth
        # Use simple calculated h_start
        itau = integrated_tau(h_start, seg_near_end, dy)
        tau_near = self.tau_zenith[None, :] * itau[:, None]
        T_near = np.exp(-tau_near)

        
        # FAR SEGMENT Optical Depth
        tau_far = np.zeros_like(tau_near)
        if np.any(far_mask):
            # Recalculate local vertical at the far side entry point P_far = t_i2 * D
            p_far = t_i2[far_mask][:, None] * ray_directions[far_mask]
            # Local UP at P_far is normalize(R_center - P_far)
            vec_to_center = self.ring_center - p_far
            up_far = vec_to_center / np.linalg.norm(vec_to_center, axis=1, keepdims=True)
            cos_theta_far = np.sum(ray_directions[far_mask] * up_far, axis=1)
            
            l_far = np.minimum(t_hits[far_mask], t_z_exit[far_mask]) - t_i2[far_mask]
            l_far = np.maximum(l_far, 0.0)
            
            # Cap far-side path by reaching the ground (h=0) if looking down
            # Dist from H_a to 0 at cos_theta_far is -H_a / cos_theta_far
            m_down = cos_theta_far < -1e-6
            if np.any(m_down):
                l_limit = -self.H_a / cos_theta_far[m_down]
                l_far[m_down] = np.minimum(l_far[m_down], l_limit)
            
            # Start far integration at the Far Ceiling (h = H_a)
            tau_far[far_mask] = self.tau_zenith[None, :] * integrated_tau(self.H_a, l_far, cos_theta_far)[:, None]

        
        T_far = np.exp(-tau_far)
        
        # 4. Phase Function and Shadowing
        # Rayleigh Phase Function: P(theta) = 3/4 * (1 + cos^2 theta)
        # Sun is at self.ring_center, which is effectively (0, 1, 0) from the origin
        sun_dir = self.ring_center / np.linalg.norm(self.ring_center)
        cos_theta = np.sum(ray_directions * sun_dir[None, :], axis=1)
        phase = 0.75 * (1.0 + cos_theta**2)
        
        # Shadowing: Sample at a representative height (Scale Height)
        # to capture volumetric light even if the observer is in a shadow.
        s_near = np.ones(tau_near.shape[0])
        if use_shadows:
            # Point at approx scale height in the ray direction
            p_near = (self.H_scale / np.maximum(np.abs(dy), 0.1))[:, None] * ray_directions
            s_near = np.atleast_1d(self.shadow_model.get_shadow_factor(p_near, time_sec))
            
        scat_near = self.sky_color[None, :] * (1.0 - T_near) * (s_near + 0.1)[:, None] * phase[:, None]
        
        scat_far = np.zeros_like(scat_near)
        if np.any(far_mask):
            # Sample far-side shadow at the midpoint of the air segment
            t_mid_far = t_i2[far_mask] + l_far / 2.0
            p_far = t_mid_far[:, None] * ray_directions[far_mask]
            s_far = np.atleast_1d(self.shadow_model.get_shadow_factor(p_far, time_sec))
            # Phase is the same (ray direction doesn't change)
            scat_far[far_mask] = self.sky_color[None, :] * (1.0 - T_far[far_mask]) * (s_far + 0.1)[:, None] * phase[far_mask, None]




        total_scat = scat_near + (scat_far * T_near)
        total_trans = T_near * T_far

        return total_trans, total_scat

