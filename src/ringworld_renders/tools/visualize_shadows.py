import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Polygon
from io import BytesIO
import PIL.Image

from ringworld_renders import constants
from ringworld_renders.shadows import omega_ss, angular_width
from ringworld_renders.core import Renderer

def render_system_plot(time_sec, renderer):
    """
    Render the Top-Down System View frame for a given time.
    Returns a PIL Image.
    """
    # Create figure (smaller size for UI side-view?)
    # or reasonable size
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1)
    
    R = renderer.R
    R_ss = renderer.R_ss
    N_ss = renderer.N_ss
    
    # Use shadow calculation functions
    ang_width = angular_width(renderer.L_ss, renderer.R_ss)
    ang_width_deg = np.rad2deg(ang_width)
    omega_ss_val = omega_ss(renderer.N_ss)
    
    # Setup plot
    ax.set_title("System View (Top-Down)")
    ax.set_aspect('equal')
    ax.set_xlim(-1.1*R, 1.1*R)
    ax.set_ylim(-1.1*R, 1.1*R)
    
    # 1. Ring (Green)
    ring_circle = plt.Circle((0, 0), R, color='green', fill=False, linewidth=2, label='Ring')
    ax.add_patch(ring_circle)
    
    # 2. Shadow Squares & Shadows
    for i in range(N_ss):
        theta_center = (i + 0.5) * (2.0 * np.pi / N_ss) + omega_ss_val * time_sec
        theta_deg = np.rad2deg(theta_center)
        
        # Map theta to matplotlib Wedge (South aligned)
        w_theta = theta_deg - 90
        
        t1 = w_theta - ang_width_deg/2
        t2 = w_theta + ang_width_deg/2
        
        col = constants.SS_COLORS[i % len(constants.SS_COLORS)]
        
        # Shadow Square itself (Centered at (0,0))
        wedge = Wedge((0, 0), R_ss, t1, t2, width=3e6 * constants.MILES_TO_METERS, color=col, alpha=0.8)
        ax.add_patch(wedge)
        
        # Projected Shadow on Ring (Penumbra/Umbra simplfied)
        # Centered at (0, 0), on Ring Radius
        # Dark Green
        ring_shadow_arc = Wedge((0, 0), R, t1, t2, width=0.08*R, color='black', alpha=0.5)
        ax.add_patch(ring_shadow_arc)

    # 3. Sun
    sun = plt.Circle((0, 0), renderer.R_sun*5, color='yellow', zorder=5)
    ax.add_patch(sun)
    
    # 4. Observer at (0, -R)
    obs_y = -R
    ax.plot(0, obs_y, 'ro', markersize=6, zorder=10)
    
    # 5. Occluded Wedge
    # Raycast helper
    def compute_wedge_poly(t_sec):
        num_rays = 100
        angles = np.linspace(-2.5, 92.5, num_rays) # Degrees relative to Horizon (+X)
        rads = np.deg2rad(angles)
        dx = np.cos(rads)
        dy = np.sin(rads)
        
        obs_pos = np.array([0, -R])
        points = [obs_pos.tolist()] 
        
        # Iterate rays
        for r_idx in range(num_rays):
            D = np.array([dx[r_idx], dy[r_idx]])
            t_candidates = []
            
            # 1. Ring (Far side)
            # t = 2 * dy * R if dy > 0
            if dy[r_idx] > 1e-6:
                t_ring = 2.0 * dy[r_idx] * R
                if t_ring > 1e-3: t_candidates.append(t_ring)
            else:
                t_candidates.append(0.0) # Floor
            
            # 2. Sun
            qa = 1.0
            qb = -2.0 * R * dy[r_idx]
            qc = R**2 - renderer.R_sun**2
            disc = qb**2 - 4*qa*qc
            
            if disc >= 0:
                sq = np.sqrt(disc)
                t1 = (-qb - sq) / (2*qa)
                if t1 > 1e-3: t_candidates.append(t1)
            
            # 3. Shadow Squares
            qc_ss = R**2 - R_ss**2
            disc_ss = qb**2 - 4*qa*qc_ss
            
            if disc_ss >= 0:
                sq = np.sqrt(disc_ss)
                t1 = (-qb - sq) / (2*qa)
                if t1 > 1e-3:
                    P = obs_pos + t1 * D
                    theta_hit = np.arctan2(P[0], -P[1])
                    half_w = ang_width / 2.0
                    
                    in_ss = False
                    for k in range(N_ss):
                        theta_c = (k + 0.5) * (2.0 * np.pi / N_ss) + omega_ss * t_sec
                        dt = (theta_hit - theta_c + np.pi) % (2.0 * np.pi) - np.pi
                        if np.abs(dt) <= half_w:
                            in_ss = True
                            break
                    if in_ss:
                        t_candidates.append(t1)
            
            t_min = min(t_candidates)
            points.append((obs_pos + t_min * D).tolist())
            
        points.append(obs_pos.tolist())
        return points

    poly_points = compute_wedge_poly(time_sec)
    wedge_poly = Polygon(poly_points, closed=True, color='red', alpha=0.3, zorder=8)
    ax.add_patch(wedge_poly)
    
    # Save to buffer
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    plt.close(fig)
    buf.seek(0)
    return PIL.Image.open(buf)


def create_visualization():
    """Entry point for ringworld-visualize command."""
    print("Launching Shadow Visualization Tool...")
    # For now, create a simple interactive plot
    # TODO: Implement full interactive visualization
    renderer = Renderer()

    # Create a sample plot at noon
    img = render_system_plot(0.0, renderer)

    # Display the plot
    img.show()

    print("Visualization created. Close the image window to exit.")
