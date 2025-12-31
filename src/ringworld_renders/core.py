import numpy as np

class Renderer:
    def __init__(self, radius_miles=92955807.0, width_miles=1000000.0, eye_height_miles=0.001242742):
        """
        Initialize the Ringworld renderer with physical parameters.
        """
        MILES_TO_METERS = 1609.34
        
        self.R = radius_miles * MILES_TO_METERS
        self.W = width_miles * MILES_TO_METERS
        self.h = eye_height_miles * MILES_TO_METERS
        
        # Atmosphere parameters
        self.H_a = 100.0 * MILES_TO_METERS # 100 mile ceiling
        self.tau_zenith = 0.15 # Better Rayleigh blue-out
        self.sky_color = np.array([0.5, 0.7, 1.0]) # Typical Rayleigh blue
        
        # Sun parameters
        self.R_sun = 432474.0 * MILES_TO_METERS
        self.sun_angular_diameter = 0.53 # degrees
        
        # Shadow Square parameters
        self.N_ss = 8
        self.R_ss = 36571994.0 * MILES_TO_METERS
        self.L_ss = 14360000.0 * MILES_TO_METERS
        self.H_ss = 1200000.0 * MILES_TO_METERS
        
        # Observer-centric coordinate system:
        # Ring center is at (0, R - h, 0).
        self.ring_center = np.array([0.0, self.R - self.h, 0.0])

    def intersect_ring(self, ray_origin, ray_directions):
        """
        Vectorized Stable intersection solver for ray and Ring (cylinder).
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
        b = 2.0 * (ox*dx + oy*dy - dy * R_minus_h)
        c = ox**2 + oy**2 - 2.0 * oy * R_minus_h - h * (2.0 * R - h)
        
        discriminant = b**2 - 4.0 * a * c
        valid_mask = discriminant >= 0
        
        t = np.full(ray_directions.shape[0], np.inf)
        
        if np.any(valid_mask):
            sqrt_disc = np.sqrt(discriminant[valid_mask])
            t1 = (-b[valid_mask] - sqrt_disc) / (2.0 * a[valid_mask])
            t2 = (-b[valid_mask] + sqrt_disc) / (2.0 * a[valid_mask])
            
            # Select smallest positive t
            best_t = np.full(t1.shape, np.inf)
            m1 = t1 > 1e-6
            best_t[m1] = t1[m1]
            m2 = (t2 > 1e-6) & (t2 < best_t)
            best_t[m2] = t2[m2]
            
            # Check width
            hit_z = ray_origin[2] + best_t * ray_directions[valid_mask, 2]
            width_mask = np.abs(hit_z) <= self.W / 2.0
            
            final_t = np.full(t1.shape, np.inf)
            final_t[width_mask] = best_t[width_mask]
            
            t[valid_mask] = final_t
            
        return t[0] if is_single else t

    def intersect_sun(self, ray_origin, ray_directions):
        """
        Vectorized intersection of ray and Sun (sphere).
        """
        is_single = ray_directions.ndim == 1
        if is_single:
            ray_directions = ray_directions[None, :]
            
        center = self.ring_center
        radius = self.R_sun
        
        oc = ray_origin - center
        # a = np.dot(ray_directions, ray_directions)
        a = np.sum(ray_directions**2, axis=1)
        b = 2.0 * np.sum(oc * ray_directions, axis=1)
        c = np.sum(oc**2) - radius**2
        
        discriminant = b**2 - 4.0 * a * c
        valid_mask = discriminant >= 0
        
        t = np.full(ray_directions.shape[0], np.inf)
        
        if np.any(valid_mask):
            t1 = (-b[valid_mask] - np.sqrt(discriminant[valid_mask])) / (2.0 * a[valid_mask])
            m = t1 > 1e-6
            t_valid = np.full(t1.shape, np.inf)
            t_valid[m] = t1[m]
            t[valid_mask] = t_valid
            
        return t[0] if is_single else t

    def get_shadow_factor(self, hit_points, time_sec):
        """
        Vectorized shadow factor calculation.
        """
        is_single = hit_points.ndim == 1
        if is_single:
            hit_points = hit_points[None, :]
            
        # Center of the ring in observer-centric frame
        center_y = self.R - self.h
        
        # Angular position theta
        theta = np.arctan2(hit_points[:, 0], -(hit_points[:, 1] - center_y))
        
        SOLAR_DAY = 24.0 * 3600.0
        T_assembly = self.N_ss * SOLAR_DAY
        omega_ss = -2.0 * np.pi / T_assembly
        
        angular_width = self.L_ss / self.R_ss
        sigma_rad = np.deg2rad(self.sun_angular_diameter)
        penumbra_width = sigma_rad
        
        # Start of full umbra
        hu = (angular_width - penumbra_width) / 2.0
        # Start of penumbra
        hw = (angular_width + penumbra_width) / 2.0
        
        shadow_factor = np.ones(hit_points.shape[0])
        
        for i in range(self.N_ss):
            ss_center_theta = (i + 0.5) * (2.0 * np.pi / self.N_ss) + omega_ss * time_sec
            
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

    def get_atmospheric_effects(self, t_hits, ray_directions):
        """
        Vectorized atmospheric effects using a robust shell intersection.
        """
        is_single = ray_directions.ndim == 1
        if is_single:
            ray_directions = ray_directions[None, :]
            t_hits = np.atleast_1d(t_hits)
            
        dx = ray_directions[:, 0]
        dy = ray_directions[:, 1]
        
        # Intersection with atmosphere ceiling (Cylinder)
        R_minus_h = self.R - self.h
        r_c = self.R - self.H_a
        
        # a*t^2 + 2*b*t + c = 0 (origin at 0,0,0)
        a = dx**2 + dy**2
        b = -dy * R_minus_h
        c = R_minus_h**2 - r_c**2
        
        # Since we are inside the ceiling radius (r_c < R-h), 
        # looking 'up' (dy > 0) will eventually hit the ceiling.
        discriminant = b**2 - a*c
        valid_mask = (a > 1e-9) & (discriminant >= 0)
        
        t_ceiling = np.full_like(t_hits, 1e12) # Far away if no hit
        if np.any(valid_mask):
            # We want the intersection point between observer and center
            t_vals = (-b[valid_mask] - np.sqrt(discriminant[valid_mask])) / a[valid_mask]
            # If t is negative, we are looking away from the shell
            t_ceiling[valid_mask] = np.maximum(t_vals, 0.0)

        # Path length is minimum of hit distance and ceiling distance
        path_in_atmosphere = np.minimum(t_hits, t_ceiling)
        
        tau = self.tau_zenith * (path_in_atmosphere / self.H_a)
        transmittance = np.exp(-tau)
        in_scattering = self.sky_color[None, :] * (1.0 - transmittance[:, None])
        
        if is_single:
            return transmittance[0], in_scattering[0]
        return transmittance, in_scattering

    def get_color(self, ray_origin, ray_directions, time_sec=0.0, 
                  use_scattering=True, use_extinction=True, 
                  use_shadows=True, use_ring_shine=True):
        """
        Fully vectorized shader.
        """
        is_single = ray_directions.ndim == 1
        if is_single:
            ray_directions = ray_directions[None, :]
        num_rays = ray_directions.shape[0]
        
        t_sun = self.intersect_sun(ray_origin, ray_directions)
        t_ring = self.intersect_ring(ray_origin, ray_directions)
        
        # Primary hit logic
        hit_t = np.minimum(t_sun, t_ring)
        hit_is_sun = (t_sun <= t_ring) & (t_sun < np.inf)
        hit_is_ring = (t_ring < t_sun) & (t_ring < np.inf)
        hit_is_sky = hit_t == np.inf
        
        surface_colors = np.zeros((num_rays, 3))
        
        # 1. Sun Color
        sun_mask = hit_is_sun
        if np.any(sun_mask):
            sun_p = ray_origin + t_sun[sun_mask][:, None] * ray_directions[sun_mask]
            s_sun = self.get_shadow_factor(sun_p, time_sec) if use_shadows else 1.0
            s_sun = np.atleast_1d(s_sun)
            surface_colors[sun_mask] = np.array([1.0, 1.0, 0.8]) * s_sun[:, None]
            
        # 2. Ring Color
        ring_mask = hit_is_ring
        if np.any(ring_mask):
            hit_p = ray_origin + t_ring[ring_mask][:, None] * ray_directions[ring_mask]
            
            # Ring-shine: proportional to visible sunlit arch
            # Simplified: scale by how much the 'average' sky is lit
            # Ring-shine is roughly albedo * (view factor of sunlit arch)
            # A simple proxy: 1.0 at noon, 0.05 at midnight (as specified by user's 'multi moons' goal)
            # We derive this from the global shadow state
            s_zenith = self.get_shadow_factor(self.ring_center[None, :], time_sec)
            s_zenith = np.atleast_1d(s_zenith)[0]
            ambient = 0.05 * s_zenith if use_ring_shine else 0.0
            
            s_factor = self.get_shadow_factor(hit_p, time_sec) if use_shadows else 1.0
            s_factor = np.atleast_1d(s_factor)
            
            # Mock green surface with physical light interaction
            surface_colors[ring_mask] = np.array([0.2, 0.5, 0.2]) * (s_factor + ambient)[:, None]
            
        # 3. Atmospheric Effects
        # We process ALL rays (including sky)
        # Use a large distance for sky rays to get full scattering
        eff_t = hit_t.copy()
        eff_t[hit_is_sky] = 1e12
        
        transmittance, in_scattering = self.get_atmospheric_effects(eff_t, ray_directions)
        
        if use_shadows:
            # Volumetric shadowing for in-scattering
            # Sample at midpoint of atmosphere
            cos_theta_z = np.maximum(ray_directions[:, 1], 0.05)
            sky_sample_p = ray_origin + (self.H_a / (2.0 * cos_theta_z))[:, None] * ray_directions
            s_sky = self.get_shadow_factor(sky_sample_p, time_sec)
            # 0.1 ambient sky glow from the rest of the atmosphere (physically based scattering)
            in_scattering *= (s_sky + 0.1)[:, None]
            
        final_colors = surface_colors * transmittance[:, None] + in_scattering
        
        if is_single:
            return np.clip(final_colors[0], 0.0, 1.0)
        return np.clip(final_colors, 0.0, 1.0)

    def render(self, width=400, height=300, fov=110.0, look_at=np.array([1.0, 0.3, 0.0]), 
               time_sec=0.0, use_scattering=True, use_extinction=True, 
               use_shadows=True, use_ring_shine=True):
        """
        Render a single image of the Ringworld using NumPy vectorization.
        """
        # Camera basis
        forward = look_at / np.linalg.norm(look_at)
        up_ref = np.array([0.0, 1.0, 0.0])
        if abs(np.dot(forward, up_ref)) > 0.999:
            right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
        else:
            right = np.cross(forward, up_ref)
        right = right / np.linalg.norm(right)
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
        colors = self.get_color(ray_origin, flat_ray_dirs, time_sec, use_scattering, 
                                use_extinction, use_shadows, use_ring_shine)
        
        return (colors.reshape(height, width, 3) * 255).astype(np.uint8)
