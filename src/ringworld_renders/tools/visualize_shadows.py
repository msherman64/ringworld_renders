import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Wedge, Polygon, Arrow
from ringworld_renders.core import Renderer

# Distinct colors for shadow squares (consistent across views)
SS_COLORS = [
    [1.0, 0.2, 0.2], # Red
    [0.2, 1.0, 0.2], # Green
    [0.2, 0.2, 1.0], # Blue
    [1.0, 1.0, 0.2], # Yellow
    [1.0, 0.2, 1.0], # Magenta
    [0.2, 1.0, 1.0], # Cyan
    [1.0, 0.6, 0.0], # Orange
    [0.6, 0.0, 1.0]  # Purple
]

class DebugRenderer(Renderer):
    def get_debug_color(self, ray_origin, ray_directions, time_sec):
        # Similar to get_color but returns debug colors for shadow squares
        t_sun = self.intersect_sun(ray_origin, ray_directions)
        t_ring = self.intersect_ring(ray_origin, ray_directions)
        t_wall = self.intersect_rim_walls(ray_origin, ray_directions)
        t_ss = self.intersect_shadow_squares(ray_origin, ray_directions, time_sec)
        
        # Determine hits logic
        # If we hit shadow square AND it's closer than sun/ring/wall
        hit_t = np.minimum(t_sun, np.minimum(t_ring, np.minimum(t_ss, t_wall)))
        
        # Masks (Prioritize simple visibility)
        # Note: intersect_ring returns 'inf' if we are inside? No, it hits the other side.
        
        hit_is_ss = (t_ss <= hit_t) & (t_ss < np.inf) & ~((t_sun < t_ss) | (t_ring < t_ss) | (t_wall < t_ss))
        hit_is_ring = (t_ring <= hit_t) & (t_ring < np.inf) & ~(hit_is_ss) & ~((t_sun < t_ring) | (t_wall < t_ring))
        hit_is_sun = (t_sun <= hit_t) & (t_sun < np.inf) & ~hit_is_ss & ~hit_is_ring
        hit_is_wall = (t_wall <= hit_t) & (t_wall < np.inf) & ~hit_is_ss & ~hit_is_ring & ~hit_is_sun
        
        colors = np.zeros((len(ray_directions), 3))
        
        # 1. Shadow Squares (Distinct Colors)
        if np.any(hit_is_ss):
            valid_dirs = ray_directions[hit_is_ss]
            hits = ray_origin + t_ss[hit_is_ss][:, None] * valid_dirs
            
            center_y = self.R - self.h
            theta = np.arctan2(hits[:, 0], -(hits[:, 1] - center_y))
            
            SOLAR_DAY = 24.0 * 3600.0
            T_assembly = self.N_ss * SOLAR_DAY
            omega_ss = -2.0 * np.pi / T_assembly
            
            angular_width = self.L_ss / self.R_ss
            half_width = angular_width / 2.0
            
            ss_cols_mapped = np.zeros((np.sum(hit_is_ss), 3))
            
            for i in range(self.N_ss):
                ss_center_theta = (i + 0.5) * (2.0 * np.pi / self.N_ss) + omega_ss * time_sec
                d_theta = (theta - ss_center_theta + np.pi) % (2.0 * np.pi) - np.pi
                
                # Check if in this square (with tolerance)
                mask = np.abs(d_theta) <= (half_width * 1.5) 
                
                col = np.array(SS_COLORS[i % len(SS_COLORS)])
                ss_cols_mapped[mask] = col
                
            colors[hit_is_ss] = ss_cols_mapped

        # 2. Ring Surface (3 Brightness Levels)
        if np.any(hit_is_ring):
            valid_dirs = ray_directions[hit_is_ring]
            hits = ray_origin + t_ring[hit_is_ring][:, None] * valid_dirs
            
            factors = self.get_shadow_factor(hits, time_sec)
            
            # 3 Levels: Daylight, Penumbra, Umbra
            # Base color: Greenish Ground
            base_color = np.array([0.4, 0.8, 0.4]) 
            
            ring_cols = np.zeros((len(factors), 3))
            
            # Vectorized assignment
            # Daylight: factor > 0.99
            mask_day = factors > 0.99
            ring_cols[mask_day] = base_color * 1.0 # Full Brightness
            
            # Umbra: factor < 0.01
            mask_umbra = factors < 0.01
            ring_cols[mask_umbra] = base_color * 0.2 # Dark
            
            # Penumbra: Between
            mask_pen = (~mask_day) & (~mask_umbra)
            ring_cols[mask_pen] = base_color * 0.5 # Medium
            
            colors[hit_is_ring] = ring_cols
            
        # Sun
        colors[hit_is_sun] = [1.0, 1.0, 0.0]
        
        # Wall
        colors[hit_is_wall] = [0.2, 0.2, 0.25]
        
        return colors

    def render_debug(self, width, height, fov, time_sec):
        # yaw=0, pitch=45 (Standard View)
        yaw, pitch = 0.0, 45.0
        
        # Basis
        y_rad = np.deg2rad(yaw)
        p_rad = np.deg2rad(pitch)
        lx = np.cos(p_rad) * np.cos(y_rad)
        ly = np.sin(p_rad)
        lz = np.cos(p_rad) * np.sin(y_rad)
        forward = np.array([lx, ly, lz])
        right = np.array([-np.sin(y_rad), 0.0, np.cos(y_rad)])
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)
        
        aspect = width / height
        tan_fov = np.tan(np.deg2rad(fov / 2.0))
        x = np.linspace(-1, 1, width) * aspect * tan_fov
        y = np.linspace(1, -1, height) * tan_fov
        px, py = np.meshgrid(x, y)
        
        ray_dirs = (forward[None, None, :] + 
                    px[:, :, None] * right[None, None, :] + 
                    py[:, :, None] * up[None, None, :])
        ray_dirs = ray_dirs / np.linalg.norm(ray_dirs, axis=2, keepdims=True)
        
        flat_dirs = ray_dirs.reshape(-1, 3)
        origin = np.array([0.0, 0.0, 0.0])
        
        colors = self.get_debug_color(origin, flat_dirs, time_sec)
        return colors.reshape(height, width, 3)

def create_visualization():
    renderer = DebugRenderer()
    
    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(1, 2, 1) # Top Down
    ax2 = fig.add_subplot(1, 2, 2) # View
    
    # Precompute constants
    R = renderer.R
    R_ss = renderer.R_ss
    N_ss = renderer.N_ss
    L_ss = renderer.L_ss
    ang_width = L_ss / R_ss
    ang_width_deg = np.rad2deg(ang_width)
    
    SOLAR_DAY = 24.0 * 3600.0
    T_assembly = N_ss * SOLAR_DAY
    omega_ss = -2.0 * np.pi / T_assembly
    
    # 2D Raycasting for Wedge
    def compute_wedge_poly(time_sec):
        # System View Coordinates: 
        # C = (0, 0) (Ring Center).
        # Obs = (0, -R).
        # Horizon/Tangent at Obs is +X (Angle 0).
        # Zenith/Inward is +Y (Angle 90).
        # FOV = 95 deg. Pitch 45.
        # Angles: 45 +/- 47.5 => [-2.5, 92.5] degrees relative to Tangent.
        # So rays are strictly (cos a, sin a).
        
        num_rays = 100
        angles = np.linspace(-2.5, 92.5, num_rays) # Degrees
        rads = np.deg2rad(angles)
        dx = np.cos(rads)
        dy = np.sin(rads)
        
        obs_pos = np.array([0, -R])
        points = [obs_pos.tolist()] 
        
        C = np.array([0, 0])
        
        # Iterate rays
        for i in range(num_rays):
            D = np.array([dx[i], dy[i]])
            
            # Intersection Candidates
            t_candidates = []
            
            # 1. Ring (Far side)
            # |(Obs + tD) - C|^2 = R^2
            # Obs = (0, -R). C = (0,0).
            # |(t dx, -R + t dy)|^2 = R^2
            # t^2 dx^2 + (-R + t dy)^2 = R^2
            # t^2 dx^2 + R^2 - 2tR dy + t^2 dy^2 = R^2
            # t^2 (dx^2 + dy^2) - 2tR dy = 0
            # t^2 - 2tR dy = 0.
            # Roots: t=0 (Self), t = 2 R dy.
            
            if dy[i] > 1e-6:
                t_ring = 2.0 * dy[i] * R
                # Only valid if we look "up" into the ring.
                t_candidates.append(t_ring)
            else:
                # Look down -> Floor.
                t_candidates.append(0.0)
            
            # 2. Sun
            # Circle centered at C(0,0), radius R_sun
            # |O + tD|^2 = R_sun^2
            # t^2 + 2t(O.D) + O.O - R_sun^2 = 0
            # O.D = -R * dy
            # O.O = R^2
            qa = 1.0
            qb = -2.0 * R * dy[i]
            qc = R**2 - renderer.R_sun**2
            disc = qb**2 - 4*qa*qc
            
            if disc >= 0:
                sq = np.sqrt(disc)
                t1 = (-qb - sq) / (2*qa)
                if t1 > 1e-3: t_candidates.append(t1)
            
            # 3. Shadow Squares
            # Centered at C(0,0), radius R_ss
            qc_ss = R**2 - R_ss**2
            disc_ss = qb**2 - 4*qa*qc_ss
            
            if disc_ss >= 0:
                sq = np.sqrt(disc_ss)
                t1 = (-qb - sq) / (2*qa)
                
                # Check angular validity for t1
                if t1 > 1e-3:
                    P = obs_pos + t1 * D
                    # Angle relative to C(0,0)
                    # theta = atan2(x, -y) ?? 
                    # Core logic: theta = atan2(x, -(y - cy))
                    # cy = 0 in our shifted coords? No, let's be careful.
                    # In core: Center is (0, R-h, 0). Obs(0,0,0).
                    # Here: Center (0,0). Obs (0, -R).
                    # The "y" in core is distance from Center?
                    # Let's just use the angle definition consistent with Shadow Square placement.
                    # SS placed at theta_center.
                    # theta = atan2(x, -y).
                    # x = P[0]. y = P[1].
                    # -y corresponds to "Up" from center?
                    # C=(0,0). Up=+Y.
                    # Core: theta=0 is Noon.
                    # Noon is Obs -> Sun. Obs at Bottom (6 oclock). Sun at Center.
                    # Vector Obs->Sun is +Y.
                    # if P is roughly (0, -R_ss), then x=0, y=-R_ss.
                    # -y = R_ss. atan2(0, R_ss) = 0.
                    # So theta = atan2(x, -y) works for "Noon = 0".
                    
                    theta_hit = np.arctan2(P[0], -P[1])
                    
                    # Check against squares
                    half_w = ang_width / 2.0
                    
                    in_ss = False
                    for k in range(N_ss):
                        theta_center = (k + 0.5) * (2.0 * np.pi / N_ss) + omega_ss * time_sec
                        dt = (theta_hit - theta_center + np.pi) % (2.0 * np.pi) - np.pi
                        if np.abs(dt) <= half_w:
                            in_ss = True
                            break
                    
                    if in_ss:
                        t_candidates.append(t1)
            
            # Pick min
            t_min = min(t_candidates)
            points.append((obs_pos + t_min * D).tolist())
            
        points.append(obs_pos.tolist())
        return points

    def update(frame):
        time_sec = frame * (SOLAR_DAY / 50.0) # 50 frames per day
        
        # View 1: Top Down
        ax1.clear()
        ax1.set_title("System View (Occluded Wedge)")
        ax1.set_aspect('equal')
        ax1.set_xlim(-1.1*R, 1.1*R)
        ax1.set_ylim(-1.1*R, 1.1*R)
        
        # 1. Ring (Green)
        # Center (0,0)
        ring_circle = plt.Circle((0, 0), R, color='green', fill=False, linewidth=2, label='Ring')
        ax1.add_patch(ring_circle)
        
        # 2. Shadow Squares & Shadows
        for i in range(N_ss):
            theta_center = (i + 0.5) * (2.0 * np.pi / N_ss) + omega_ss * time_sec
            theta_deg = np.rad2deg(theta_center)
            
            # Map theta to matplotlib Wedge
            # Core theta=0 is +X (Tangent)? No, it's atan2(x, -y).
            # If x=0, y=-R (Obs directly below sun). theta=0.
            # Matplotlib 0 is +X. 90 is +Y.
            # If theta=0 (Noon), we are at (0, -R_ss). 6 o'clock.
            # Matplotlib angle should be -90.
            # If theta=90 (Spinward), we are at (R_ss, 0). 3 o'clock.
            # Matplotlib angle should be 0.
            # So Matplotlib Angle = Theta_deg - 90?
            # 90 - 90 = 0. Correct.
            # 0 - 90 = -90. Correct.
            
            w_theta = theta_deg - 90
            
            t1 = w_theta - ang_width_deg/2
            t2 = w_theta + ang_width_deg/2
            
            col = SS_COLORS[i % len(SS_COLORS)]
            
            # Shadow Square itself (Centered at (0,0))
            wedge = Wedge((0, 0), R_ss, t1, t2, width=3e6*1609, color=col, alpha=0.8)
            ax1.add_patch(wedge)
            
            # Projected Shadow on Ring (Penumbra/Umbra simplfied)
            # Centered at (0, 0), on Ring Radius
            # Dark Green
            ring_shadow_arc = Wedge((0, 0), R, t1, t2, width=0.08*R, color='black', alpha=0.5)
            ax1.add_patch(ring_shadow_arc)

        # 3. Sun
        sun = plt.Circle((0, 0), renderer.R_sun*5, color='yellow', zorder=5)
        ax1.add_patch(sun)
        
        # 4. Observer at (0, -R)
        obs_y = -R
        ax1.plot(0, obs_y, 'ro', markersize=6, zorder=10)
        
        # 5. Occluded Wedge
        poly_points = compute_wedge_poly(time_sec)
        wedge_poly = Polygon(poly_points, closed=True, color='red', alpha=0.3, zorder=8)
        ax1.add_patch(wedge_poly)

        
        # View 2: Surface Render
        ax2.clear()
        ax2.set_title("Surface View")
        ax2.axis('off')
        
        img = renderer.render_debug(width=320, height=240, fov=95.0, time_sec=time_sec)
        ax2.imshow(img)
    
    # Animate
    frames = 50
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=100)
    
    print("Saving animation to animation_shadows_final.gif...")
    ani.save('output/animation_shadows_final.gif', writer='pillow', fps=10)
    print("Done.")

if __name__ == "__main__":
    create_visualization()
