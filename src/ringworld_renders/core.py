import numpy as np
import functools
from ringworld_renders import constants

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

    def _solve_quadratic_vectorized(self, a, b, c):
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
    @property
    def omega_ss(self):
        """Angular velocity of Shadow Squares (rad/s)."""
        T_assembly = self.N_ss * constants.SOLAR_DAY_SECONDS
        return -2.0 * np.pi / T_assembly

    @property
    def angular_width(self):
        """Angular width of a single Shadow Square (radians)."""
        return self.L_ss / self.R_ss

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
        
        t1, t2, valid_mask = self._solve_quadratic_vectorized(a, b, c)
        
        t = np.full(ray_directions.shape[0], np.inf)
        
        if np.any(valid_mask):

            
            # Select smallest positive t
            best_t = np.full(t1.shape, np.inf)
            m1 = t1 > 1e-6
            best_t[m1] = t1[m1]
            m2 = (t2 > 1e-6) & (t2 < best_t)
            best_t[m2] = t2[m2]
            
            # Check width
            # hit_z = ray_origin[2] + best_t * ray_directions[:, 2]
            # But we must avoid Inf * 0 or similar NaN issues if direction is 0?
            # best_t is Inf where invalid. 
            # Inf * 0 is NaN. NaN <= W/2 is False. So it's safe.
            hit_z = ray_origin[2] + best_t * ray_directions[:, 2]
            width_mask = np.abs(hit_z) <= self.W / 2.0
            
            final_t = np.full(t1.shape, np.inf)
            final_t[width_mask] = best_t[width_mask]
            
            t = final_t
            
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
        
        t1, _, valid_mask = self._solve_quadratic_vectorized(a, b, c)
        
        t = np.full(ray_directions.shape[0], np.inf)
        
        if np.any(valid_mask):
            m = t1 > 1e-6
            t_valid = np.full(t1.shape, np.inf)
            t_valid[m] = t1[m]
            t = t_valid

            
        return t[0] if is_single else t

    def intersect_shadow_squares(self, ray_origin, ray_directions, time_sec):
        """
        Vectorized intersection of ray and the shadow square cylindrical shell.
        """
        is_single = ray_directions.ndim == 1
        if is_single:
            ray_directions = ray_directions[None, :]
            
        dx = ray_directions[:, 0]
        dy = ray_directions[:, 1]
        ox, oy, oz = ray_origin
        
        R_ss = self.R_ss
        center_y = self.center_y
        
        # Intersection with cylinder (r = R_ss)
        # (x - cx)^2 + (y - cy)^2 = R^2
        # cx = 0, cy = center_y
        # dx^2*t^2 + 2*dx*ox*t + ox^2 + dy^2*t^2 + 2*dy*(oy-center_y)*t + (oy-center_y)^2 = R_ss^2
        
        a = dx**2 + dy**2
        b = 2.0 * (ox*dx + (oy - center_y)*dy)
        c = ox**2 + (oy - center_y)**2 - R_ss**2
        
        t1, t2, valid_mask = self._solve_quadratic_vectorized(a, b, c)
        
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

    def get_shadow_factor(self, hit_points, time_sec):
        """
        Vectorized shadow factor calculation.
        """
        is_single = hit_points.ndim == 1
        if is_single:
            hit_points = hit_points[None, :]
            
        # Center of the ring in observer-centric frame
        center_y = self.center_y
        
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
                valid_r = (r_sq <= self.R**2) & (r_sq >= (self.R - constants.RIM_WALL_HEIGHT_METERS)**2)
                
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
                valid_r = (r_sq <= self.R**2) & (r_sq >= (self.R - constants.RIM_WALL_HEIGHT_METERS)**2)
                
                subset_t = t_cand[mask_neg]
                subset_t[valid_r] = tn[valid_r]
                t_cand[mask_neg] = subset_t
            
            # Filter valid
            valid_t = (t_cand > 1e-6)
            t[valid_t] = t_cand[valid_t]
            
        return t[0] if is_single else t

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
        
        t_i1, t_i2, mask_i = self._solve_quadratic_vectorized(a, b, c_inner)
        
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
        Calculate the color for each ray.
        vectorized for N rays.
        """
        is_single = ray_directions.ndim == 1
        if is_single:
            ray_directions = ray_directions[None, :]
        num_rays = ray_directions.shape[0]
        
        # 0. Intersections
        # Note: intersect_sun uses sphere logic. Not quite physically perfectly aligned with shadow squares?
        # But for this purpose, it works.
        
        t_sun = self.intersect_sun(ray_origin, ray_directions)
        t_ring = self.intersect_ring(ray_origin, ray_directions)
        t_wall = self.intersect_rim_walls(ray_origin, ray_directions)
        t_ss = self.intersect_shadow_squares(ray_origin, ray_directions, time_sec) if use_shadows or debug_shadow_squares else np.full(num_rays, np.inf)
        
        # Primary hit logic
        hit_t = np.minimum(t_sun, np.minimum(t_ring, np.minimum(t_ss, t_wall)))
        hit_is_sun = (t_sun <= hit_t) & (t_sun < np.inf)
        hit_is_ring = (t_ring <= hit_t) & (t_ring < np.inf) & ~hit_is_sun
        hit_is_wall = (t_wall <= hit_t) & (t_wall < np.inf) & ~hit_is_sun & ~hit_is_ring
        hit_is_ss = (t_ss <= hit_t) & (t_ss < np.inf) & ~hit_is_sun & ~hit_is_ring & ~hit_is_wall
        hit_is_sky = hit_t == np.inf
        
        surface_colors = np.zeros((num_rays, 3))
        
        # 1. Sun Color
        sun_mask = hit_is_sun
        if np.any(sun_mask):
            # Sun is an emitter; in a unified geometric model, if we hit the sun, 
            # we are by definition not occluded by a shadow square (because t_ss > t_sun).
            surface_colors[sun_mask] = np.array([1.0, 1.0, 0.8])
            
        # 2. Shadow Square Color (Occlusion)
        ss_mask = hit_is_ss
        if np.any(ss_mask):
            if debug_shadow_squares:
                # False Colors
                valid_dirs = ray_directions[ss_mask]
                hits = ray_origin + t_ss[ss_mask][:, None] * valid_dirs
                
                center_y = self.center_y
                theta = np.arctan2(hits[:, 0], -(hits[:, 1] - center_y))
                
                omega = self.omega_ss
                half_width = self.angular_width / 2.0
                
                ss_colors_mapped = np.zeros((np.sum(ss_mask), 3))
                
                for i in range(self.N_ss):
                    ss_center_theta = (i + 0.5) * (2.0 * np.pi / self.N_ss) + omega * time_sec
                    d_theta = (theta - ss_center_theta + np.pi) % (2.0 * np.pi) - np.pi
                    
                    # Check if in this square (with tolerance)
                    mask_in_sq = np.abs(d_theta) <= (half_width * 1.5)
                    col = np.array(constants.SS_COLORS[i % len(constants.SS_COLORS)])
                    ss_colors_mapped[mask_in_sq] = col
                
                surface_colors[ss_mask] = ss_colors_mapped
            else:
                # Shadow squares are opaque and black (back side)
                surface_colors[ss_mask] = np.array([0.0, 0.0, 0.0])
            
        # 3. Rim Wall Color
        wall_mask = hit_is_wall
        if np.any(wall_mask):
            # Dark grey rock
            surface_colors[wall_mask] = np.array([0.1, 0.1, 0.15])
            
        # 3. Ring Color
        ring_mask = hit_is_ring
        if np.any(ring_mask):
            hit_p = ray_origin + t_ring[ring_mask][:, None] * ray_directions[ring_mask]
            
            # Dynamic Ring-shine: peaking at midnight (12h) when the arch is overhead.
            # At Noon (0s), the arch is backlit/blocked.
            time_angle = 2.0 * np.pi * time_sec / constants.SOLAR_DAY_SECONDS
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

        
        transmittance, in_scattering = self.get_atmospheric_effects(eff_t, ray_origin, ray_directions, time_sec, use_shadows)
        
        # Final blend

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
