import numpy as np
import functools
from ringworld_renders import constants
from ringworld_renders.rendering import HitSelector, MaterialSystem, AtmosphericModel
from ringworld_renders.intersections import (
    intersect_ring, intersect_sun, intersect_rim_walls, solve_quadratic_vectorized
)
from ringworld_renders.shadows import ShadowModel

class Renderer:
    def __init__(self, radius_miles=None, width_miles=None, eye_height_miles=None):
        """
        Initialize the Ringworld renderer with physical parameters.
        
        Coordinate System (Observer-Centric):
        - Origin (0,0,0): The Observer's position.
        - Ring Center: Located at (0, R - h, 0).
        - Y-Axis: "Up" relative to the floor local to the observer (prior to curvature).
        - R: Ring Radius.
        - h: Observer height above floor.
        """
        # Defaults if not provided
        r_miles = radius_miles if radius_miles is not None else constants.DEFAULT_RADIUS_MILES
        w_miles = width_miles if width_miles is not None else constants.DEFAULT_WIDTH_MILES
        h_miles = eye_height_miles if eye_height_miles is not None else constants.DEFAULT_EYE_HEIGHT_MILES
        
        self.R = r_miles * constants.MILES_TO_METERS
        self.W = w_miles * constants.MILES_TO_METERS
        self.h = h_miles * constants.MILES_TO_METERS
        
        # Atmosphere parameters (Rayleigh 1/lambda^4 scaling)
        self.H_a = 100.0 * constants.MILES_TO_METERS # 100 mile Effective Ceiling
        self.H_scale = 5.3 * constants.MILES_TO_METERS # Scale height for 1G
        # Rayleigh spectral scaling relative to blue (450nm): [650nm, 550nm, 450nm]
        self.tau_zenith = np.array([0.0138, 0.027, 0.06])
        self.sky_color = np.array([0.5, 0.7, 1.0]) # Typical Rayleigh blue
        
        # Shadow Square parameters
        self.N_ss = constants.NUM_SHADOW_SQUARES
        self.R_ss = constants.SS_RADIUS_METERS
        self.L_ss = constants.SS_LENGTH_METERS
        self.H_ss = constants.SS_HEIGHT_METERS
        
        # Observer-centric coordinate system:
        # Ring center is at (0, R - h, 0).
        self.center_y = self.R - self.h  # Cached for frequent use
        self.ring_center = np.array([0.0, self.center_y, 0.0])

        # Sun parameters
        self.R_sun = constants.SUN_RADIUS_METERS
        # Derive angular diameter from geometry
        # tan(theta/2) = R_sun / (R - R_sun) approx for observer at 0, R_sun at center
        # Distance to sun center ~ R_sun (No, Sun is at center (0, R-h, 0)).
        # Distance from observer (0,0,0) to Center (0, R-h, 0) is approx R.
        dist_to_sun = np.linalg.norm(self.ring_center)
        self.sun_angular_diameter = 2.0 * np.rad2deg(np.arctan(self.R_sun / dist_to_sun))

        # Physics modules
        self.shadow_model = ShadowModel(
            N_ss=self.N_ss,
            R_ss=self.R_ss,
            L_ss=self.L_ss,
            H_ss=self.H_ss,
            sun_angular_diameter=self.sun_angular_diameter
        )

        # Rendering pipeline components
        self.hit_selector = HitSelector(self)
        self.material_system = MaterialSystem(self)
        self.atmospheric_model = AtmosphericModel(self)


    def intersect_ring(self, ray_origin, ray_directions):
        """
        Vectorized Stable intersection solver for ray and Ring (cylinder).
        """
        return intersect_ring(ray_origin, ray_directions, self.R, self.h, self.W)

    def intersect_sun(self, ray_origin, ray_directions):
        """
        Vectorized intersection of ray and Sun (sphere).
        """
        return intersect_sun(ray_origin, ray_directions, self.ring_center, self.R_sun)

    def intersect_shadow_squares(self, ray_origin, ray_directions, time_sec):
        """
        Vectorized intersection of ray and the shadow square cylindrical shell.
        """
        return self.shadow_model.intersect_shadow_squares(ray_origin, ray_directions, time_sec, self.center_y)

    def get_shadow_factor(self, hit_points, time_sec):
        """
        Vectorized shadow factor calculation.
        """
        return self.shadow_model.get_shadow_factor(hit_points, time_sec, self.center_y)

    def intersect_rim_walls(self, ray_origin, ray_directions):
        """
        Vectorized intersection with Rim Walls (planes at Z = +/- W/2).
        Walls exist between R - H_w and R.
        """
        return intersect_rim_walls(ray_origin, ray_directions, self.R, self.h, self.W, constants.RIM_WALL_HEIGHT_METERS)

    def get_atmospheric_effects(self, t_hits, ray_origin, ray_directions, time_sec=0.0, use_shadows=True):
        """
        Analytical integral of exponential density.
        """
        is_single = ray_directions.ndim == 1
        if is_single:
            ray_directions = ray_directions[None, :]
            t_hits = np.atleast_1d(t_hits)
            
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
            s_near = np.atleast_1d(self.get_shadow_factor(p_near, time_sec))
            
        scat_near = self.sky_color[None, :] * (1.0 - T_near) * (s_near + 0.1)[:, None] * phase[:, None]
        
        scat_far = np.zeros_like(scat_near)
        if np.any(far_mask):
            # Sample far-side shadow at the midpoint of the air segment
            t_mid_far = t_i2[far_mask] + l_far / 2.0
            p_far = t_mid_far[:, None] * ray_directions[far_mask]
            s_far = np.atleast_1d(self.get_shadow_factor(p_far, time_sec))
            # Phase is the same (ray direction doesn't change)
            scat_far[far_mask] = self.sky_color[None, :] * (1.0 - T_far[far_mask]) * (s_far + 0.1)[:, None] * phase[far_mask, None]




        total_scat = scat_near + (scat_far * T_near)
        total_trans = T_near * T_far
        
        if is_single:
            return total_trans[0, :], total_scat[0, :]
        return total_trans, total_scat






    def get_color(self, ray_origin, ray_directions, time_sec=0.0,
                  use_atmosphere=True, use_shadows=True, use_ring_shine=True,
                  debug_shadow_squares=False):
        """
        Calculate the color for each ray using the rendering pipeline.
        vectorized for N rays.
        """
        is_single = ray_directions.ndim == 1
        if is_single:
            ray_directions = ray_directions[None, :]

        # Stage 1: Intersection → HitResult
        hits = self.hit_selector.select_primary(ray_origin, ray_directions, time_sec)

        # Stage 2: HitResult → SurfaceColor
        surface_colors = self.material_system.get_surface_color(
            hits, time_sec, use_shadows, use_ring_shine, debug_shadow_squares
        )

        # Stage 3: Ray + HitResult → Atmospheric Effects
        # Use large distance for sky rays to bound atmospheric sampling
        eff_t = hits.distance.copy()
        sky_mask = hits.hit_type == "sky"
        eff_t[sky_mask] = self.R * 3.0

        transmittance, in_scattering = self.atmospheric_model.get_atmospheric_effects(
            eff_t, ray_origin, ray_directions, time_sec, use_shadows
        )

        # Stage 4: Composition
        transmittance_applied = transmittance if use_atmosphere else np.ones_like(transmittance)
        scattering_applied = in_scattering if use_atmosphere else np.zeros_like(in_scattering)

        final_colors = surface_colors * transmittance_applied + scattering_applied

        if is_single:
            return np.clip(final_colors[0], 0.0, 1.0)
        return np.clip(final_colors, 0.0, 1.0)

    @functools.lru_cache(maxsize=32)
    def _render_cached(self, width, height, fov, yaw, pitch, time_sec, 
                       use_atmosphere, use_shadows, use_ring_shine, debug_shadow_squares):
        """
        Internal cached render call using hashable arguments.
        Yaw 0 = +X (Spinward), Yaw 90 = +Z (Axial)
        Pitch 90 = +Y (Zenith)
        """
        # 1. Construct Basis from Spherical Coordinates (Robust)
        y_rad = np.deg2rad(yaw)
        p_rad = np.deg2rad(pitch)
        
        # Look vector (Forward)
        lx = np.cos(p_rad) * np.cos(y_rad)
        ly = np.sin(p_rad)
        lz = np.cos(p_rad) * np.sin(y_rad)
        forward = np.array([lx, ly, lz])
        
        # Right vector: Always horizontal, perpendicular to zenith and forward.
        right = np.array([-np.sin(y_rad), 0.0, np.cos(y_rad)])
        
        # Up vector: perpendicular to Forward and Right
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)
        
        aspect_ratio = width / height
        tan_half_fov = np.tan(np.deg2rad(fov / 2.0))
        
        # Grid of pixel coordinates
        x = np.linspace(-1, 1, width) * aspect_ratio * tan_half_fov
        y = np.linspace(1, -1, height) * tan_half_fov
        px, py = np.meshgrid(x, y)
        
        # Ray directions
        ray_dirs = (forward[None, None, :] + 
                    px[:, :, None] * right[None, None, :] + 
                    py[:, :, None] * up[None, None, :])
        ray_dirs = ray_dirs / np.linalg.norm(ray_dirs, axis=2, keepdims=True)
        
        # Flatten and shade
        flat_ray_dirs = ray_dirs.reshape(-1, 3)
        ray_origin = np.array([0.0, 0.0, 0.0])
        colors = self.get_color(ray_origin, flat_ray_dirs, time_sec, use_atmosphere, 
                                use_shadows, use_ring_shine, debug_shadow_squares)
        
        return (colors.reshape(height, width, 3) * 255).astype(np.uint8)


    def render(self, width=400, height=300, fov=95.0, 
               yaw=0.0, pitch=45.0, look_at=None,
               time_sec=0.0, use_atmosphere=True, 
               use_shadows=True, use_ring_shine=True,
               debug_shadow_squares=False):
        """
        Render a single image of the Ringworld using NumPy vectorization.
        Uses spherical coordinates (yaw/pitch) for a robust camera basis.
        """
        # Handle look_at override for backward compatibility
        if look_at is not None:
            # Convert look_at vector back to yaw/pitch
            mag = np.linalg.norm(look_at)
            pitch = np.rad2deg(np.arcsin(np.clip(look_at[1] / mag, -1.0, 1.0)))
            yaw = np.rad2deg(np.arctan2(look_at[2], look_at[0]))
        
        return self._render_cached(
            width, height, fov, yaw, pitch, time_sec,
            use_atmosphere, use_shadows, use_ring_shine, debug_shadow_squares
        )
