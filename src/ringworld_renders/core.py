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
        
        # Atmosphere parameters (Rayleigh 1/lambda^4 scaling)
        self.H_a = 1000.0 * MILES_TO_METERS # 1000 mile Rim Walls
        self.H_scale = 5.3 * MILES_TO_METERS # Scale height for 1G
        # Rayleigh spectral scaling relative to blue (450nm): [650nm, 550nm, 450nm]
        # powers = (450 / [650, 550, 450])^4
        self.tau_zenith = np.array([0.0138, 0.027, 0.06])
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
        valid_mask = (a > 1e-12) & (discriminant >= 0)
        
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
        valid_mask = (a > 1e-12) & (discriminant >= 0)
        
        t = np.full(ray_directions.shape[0], np.inf)
        
        if np.any(valid_mask):
            sqrt_disc = np.sqrt(discriminant[valid_mask])
            t1 = (-b[valid_mask] - sqrt_disc) / (2.0 * a[valid_mask])
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

    def get_atmospheric_effects(self, t_hits, ray_directions, time_sec=0.0, use_shadows=True):
        """
        Analytical integral of exponential density: tau = tau_z * integral[e^(-h(s)/H) ds] / H.
        This provides physical grounding and eliminates banding artifacts.
        """
        is_single = ray_directions.ndim == 1
        if is_single:
            ray_directions = ray_directions[None, :]
            t_hits = np.atleast_1d(t_hits)
            
        dx, dy, dz = ray_directions[:, 0], ray_directions[:, 1], ray_directions[:, 2]
        r_obs = self.h # Observer distance from floor
        R_minus_h = self.R - self.h
        
        # 1. Geometric Boundaries
        # Radial boundaries: inner (R-Ha) and outer (R)
        a = dx**2 + dy**2
        b = -dy * R_minus_h
        r_inner = self.R - self.H_a
        c_inner = R_minus_h**2 - r_inner**2
        disc_i = b**2 - a*c_inner
        mask_i = (a > 1e-12) & (disc_i >= 0)
        t_i1 = np.full_like(t_hits, np.inf)
        t_i2 = np.full_like(t_hits, np.inf)
        if np.any(mask_i):
            sq_i = np.sqrt(disc_i[mask_i])
            t_i1[mask_i] = (-b[mask_i] - sq_i) / a[mask_i]
            t_i2[mask_i] = (-b[mask_i] + sq_i) / a[mask_i]

        # Width boundary (Rim Walls at +/- W/2)
        t_z_exit = np.full_like(t_hits, np.inf)
        abs_dz = np.abs(dz)
        m_z = abs_dz > 1e-12
        t_z_exit[m_z] = (self.W / 2.0) / abs_dz[m_z]

        # 2. Segment Identification
        looking_up = dy > 0
        seg_near_end = np.minimum(t_hits, t_z_exit)
        seg_near_end[looking_up] = np.minimum(seg_near_end[looking_up], t_i1[looking_up])
        
        # Far segment: enters at t_i2, exits at t_hit, clipped by Z
        # We ensure t_i2 is finite to avoid NaN in the width check
        far_mask = looking_up & (t_hits > t_i2) & np.isfinite(t_i2)
        if np.any(far_mask):
            z_at_i2 = t_i2[far_mask] * dz[far_mask]
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
        itau = integrated_tau(r_obs, seg_near_end, dy)
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
                  use_atmosphere=True, use_shadows=True, use_ring_shine=True):
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
            
            # Dynamic Ring-shine: peaking at midnight (12h) when the arch is overhead.
            # At Noon (0s), the arch is backlit/blocked.
            SOLAR_DAY = 24.0 * 3600.0
            time_angle = 2.0 * np.pi * time_sec / SOLAR_DAY
            # Scale from 0.02 (noon) to 0.10 (midnight)
            ambient_shine = 0.06 - 0.04 * np.cos(time_angle) if use_ring_shine else 0.0
            
            s_factor = self.get_shadow_factor(hit_p, time_sec) if use_shadows else 1.0
            s_factor = np.atleast_1d(s_factor)
            
            # Mock green surface with physical light interaction
            surface_colors[ring_mask] = np.array([0.2, 0.5, 0.2]) * (s_factor + ambient_shine)[:, None]


            
        # 3. Atmospheric Effects
        # We process ALL rays (including sky)
        # Use a large but finite distance for sky rays to bound shadow sampling
        eff_t = hit_t.copy()
        eff_t[hit_is_sky] = self.R * 3.0

        
        transmittance, in_scattering = self.get_atmospheric_effects(eff_t, ray_directions, time_sec, use_shadows)
        
        # Final blend

        transmittance_applied = transmittance if use_atmosphere else np.ones_like(transmittance)
        scattering_applied = in_scattering if use_atmosphere else np.zeros_like(in_scattering)
        
        final_colors = surface_colors * transmittance_applied + scattering_applied

        
        if is_single:
            return np.clip(final_colors[0], 0.0, 1.0)
        return np.clip(final_colors, 0.0, 1.0)

    def render(self, width=400, height=300, fov=110.0, look_at=np.array([1.0, 0.3, 0.0]), 
               time_sec=0.0, use_atmosphere=True, 
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
        colors = self.get_color(ray_origin, flat_ray_dirs, time_sec, use_atmosphere, 
                                use_shadows, use_ring_shine)

        
        return (colors.reshape(height, width, 3) * 255).astype(np.uint8)
