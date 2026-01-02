import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Polygon
from io import BytesIO
import PIL.Image

from ringworld_renders import constants


def compute_visible_region_polar(time_sec, renderer, fov_deg=95.0, yaw_deg=0.0, pitch_deg=45.0):
    """
    Compute the visible region using polar coordinates centered on the observer.
    
    This is much simpler: for each angular direction, we just find the distance
    to the first occluding object (shadow square or ring).
    
    Returns:
        List of (x, y) points in Cartesian coordinates for plotting
    """
    R = renderer.R
    R_ss = renderer.R_ss
    N_ss = renderer.N_ss
    ang_width = renderer.shadow_model.angular_width
    omega_ss = renderer.shadow_model.omega_ss
    
    # Observer position in Cartesian
    obs_pos = np.array([0.0, -R])
    
    # Convert camera parameters
    yaw_rad = np.deg2rad(yaw_deg)
    fov_rad = np.deg2rad(fov_deg)
    pitch_rad = np.deg2rad(pitch_deg)
    
    # Camera look direction (in observer's frame, straight ahead is toward center)
    # In global frame, this is +y direction (angle = π/2)
    center_look_angle = np.pi / 2 + yaw_rad
    
    # Field of view boundaries
    half_fov = fov_rad / 2.0
    pitch_factor = np.sin(pitch_rad)
    pitch_extension = pitch_factor * np.deg2rad(20)
    
    angle_min = center_look_angle - half_fov - pitch_extension
    angle_max = center_look_angle + half_fov + pitch_extension
    
    # ========================================================================
    # Compute shadow square positions at this time
    # ========================================================================
    
    shadow_squares = []
    for i in range(N_ss):
        theta_center = (i + 0.5) * (2.0 * np.pi / N_ss) + omega_ss * time_sec
        half_width = ang_width / 2.0
        
        shadow_squares.append({
            'theta_min': theta_center - half_width,
            'theta_max': theta_center + half_width,
        })
    
    # ========================================================================
    # Sample angles uniformly and find occlusion distance for each
    # ========================================================================
    
    n_samples = 2000
    angles = np.linspace(angle_min, angle_max, n_samples)
    
    # For each angle, find the distance to first occlusion
    polygon_points = [obs_pos.copy()]
    
    for angle in angles:
        # Ray direction from observer
        ray_dir = np.array([np.cos(angle), np.sin(angle)])
        
        # Check what this ray hits:
        # 1. Shadow squares at R_ss?
        # 2. Ring at R?
        
        # To check shadow square intersection:
        # Ray from obs_pos = (0, -R) in direction ray_dir
        # Hits circle at R_ss if it intersects
        
        # Parametric ray: P(t) = obs_pos + t * ray_dir
        # Circle: |P|² = R_ss²
        # Solve: |obs_pos + t * ray_dir|² = R_ss²
        
        a = np.dot(ray_dir, ray_dir)  # Should be 1 since normalized
        b = 2 * np.dot(obs_pos, ray_dir)
        c = np.dot(obs_pos, obs_pos) - R_ss**2
        
        discriminant = b**2 - 4*a*c
        
        hit_distance = None
        
        if discriminant >= 0:
            # Ray intersects the R_ss circle
            sqrt_disc = np.sqrt(discriminant)
            t1 = (-b - sqrt_disc) / (2*a)
            t2 = (-b + sqrt_disc) / (2*a)
            
            # Check both intersection points
            for t in [t1, t2]:
                if t > 1e-6:  # Forward along ray
                    hit_point = obs_pos + t * ray_dir
                    # What angle is this hit point at (in global frame)?
                    hit_angle = np.arctan2(hit_point[1], hit_point[0])
                    
                    # Is this within any shadow square?
                    for ss in shadow_squares:
                        # Check if hit_angle is within [theta_min, theta_max]
                        # Handle wraparound
                        theta_min = ss['theta_min']
                        theta_max = ss['theta_max']
                        
                        # Normalize to same range
                        while hit_angle < theta_min - np.pi:
                            hit_angle += 2*np.pi
                        while hit_angle > theta_min + np.pi:
                            hit_angle -= 2*np.pi
                        
                        if theta_min <= hit_angle <= theta_max:
                            # Hit a shadow square!
                            if hit_distance is None or t < hit_distance:
                                hit_distance = t
                            break
        
        # If no shadow square hit, check ring at R
        if hit_distance is None:
            # Ray-circle intersection with ring at R
            c_ring = np.dot(obs_pos, obs_pos) - R**2
            discriminant_ring = b**2 - 4*a*c_ring
            
            if discriminant_ring >= 0:
                sqrt_disc = np.sqrt(discriminant_ring)
                t1 = (-b - sqrt_disc) / (2*a)
                t2 = (-b + sqrt_disc) / (2*a)
                
                # Take the furthest positive intersection (far side of ring)
                valid_t = [t for t in [t1, t2] if t > 1e-6]
                if valid_t:
                    hit_distance = max(valid_t)
        
        # Add the hit point to polygon
        if hit_distance is not None:
            hit_point = obs_pos + hit_distance * ray_dir
            polygon_points.append(hit_point)
    
    # Close polygon
    polygon_points.append(obs_pos.copy())
    
    return [p.tolist() for p in polygon_points]


def render_system_plot(time_sec, renderer, fov=95.0, yaw=0.0, pitch=45.0):
    """
    Render the Top-Down System View frame for a given time and camera settings.
    Returns a PIL Image.
    """
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1)
    
    R = renderer.R
    R_ss = renderer.R_ss
    N_ss = renderer.N_ss
    
    ang_width = renderer.shadow_model.angular_width
    ang_width_deg = np.rad2deg(ang_width)
    omega_ss = renderer.shadow_model.omega_ss
    
    # Setup plot
    ax.set_title("System View (Top-Down) - Polar Method")
    ax.set_aspect('equal')
    ax.set_xlim(-1.1*R, 1.1*R)
    ax.set_ylim(-1.1*R, 1.1*R)
    ax.grid(True, alpha=0.3)
    
    # 1. Ring (Green)
    ring_circle = plt.Circle((0, 0), R, color='green', fill=False, linewidth=2, label='Ring')
    ax.add_patch(ring_circle)
    
    # 2. Shadow Square orbit (dashed circle at R_ss)
    ss_orbit = plt.Circle((0, 0), R_ss, color='gray', fill=False, 
                          linewidth=1, linestyle='--', alpha=0.5)
    ax.add_patch(ss_orbit)
    
    # 3. Shadow Squares & their shadows on the ring
    for i in range(N_ss):
        theta_center = (i + 0.5) * (2.0 * np.pi / N_ss) + omega_ss * time_sec
        theta_deg = np.rad2deg(theta_center)
        
        # Map theta to matplotlib Wedge (South aligned)
        w_theta = theta_deg - 90
        
        t1 = w_theta - ang_width_deg/2
        t2 = w_theta + ang_width_deg/2
        
        col = constants.SS_COLORS[i % len(constants.SS_COLORS)]
        
        # Shadow Square - represented as thin lines at the edges
        theta_left_rad = np.deg2rad(t1 + 90)  # Convert back to standard angle
        theta_right_rad = np.deg2rad(t2 + 90)
        
        # Inner and outer radius of shadow square
        r_inner = R_ss - 3e6*1609
        r_outer = R_ss
        
        # Left edge line
        x_left = [r_inner * np.cos(theta_left_rad), r_outer * np.cos(theta_left_rad)]
        y_left = [r_inner * np.sin(theta_left_rad), r_outer * np.sin(theta_left_rad)]
        ax.plot(x_left, y_left, color=col, linewidth=2, alpha=0.9)
        
        # Right edge line
        x_right = [r_inner * np.cos(theta_right_rad), r_outer * np.cos(theta_right_rad)]
        y_right = [r_inner * np.sin(theta_right_rad), r_outer * np.sin(theta_right_rad)]
        ax.plot(x_right, y_right, color=col, linewidth=2, alpha=0.9)
        
        # Outer arc connecting the two edges
        arc_angles = np.linspace(theta_left_rad, theta_right_rad, 20)
        x_arc = r_outer * np.cos(arc_angles)
        y_arc = r_outer * np.sin(arc_angles)
        ax.plot(x_arc, y_arc, color=col, linewidth=2, alpha=0.9)
        
        # Projected Shadow on Ring (darker wedge)
        ring_shadow_arc = Wedge((0, 0), R, t1, t2, width=0.08*R, color='black', alpha=0.5)
        ax.add_patch(ring_shadow_arc)

    # 4. Sun (at center)
    sun = plt.Circle((0, 0), renderer.R_sun*5, color='yellow', zorder=5)
    ax.add_patch(sun)
    
    # 5. Observer at (0, -R)
    obs_y = -R
    ax.plot(0, obs_y, 'ro', markersize=8, zorder=10, label='Observer')
    
    # 6. Visible region polygon
    poly_points = compute_visible_region_polar(time_sec, renderer, fov, yaw, pitch)
    wedge_poly = Polygon(poly_points, closed=True, color='red', alpha=0.25, 
                        edgecolor='red', linewidth=1.5, zorder=8, 
                        label='Visible Region')
    ax.add_patch(wedge_poly)
    
    ax.legend(loc='upper right', fontsize=8)
    
    # Save to buffer
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    plt.close(fig)
    buf.seek(0)
    return PIL.Image.open(buf)


def create_system_animation(renderer, fov=95.0, yaw=0.0, pitch=45.0, 
                           duration_hours=6, fps=2, time_step_seconds = 60):
    """
    Create an animated matplotlib figure showing the Ringworld system over time.
    
    Args:
        renderer: The ringworld renderer object
        fov: Field of view in degrees (default 95.0)
        yaw: Camera yaw in degrees (default 0.0)
        pitch: Camera pitch in degrees (default 45.0)
        duration_hours: Total animation duration in hours (default 6)
        fps: Frames per second (default 2)

    Returns:
        fig, anim: The matplotlib figure and animation objects
    """
    import matplotlib.animation as animation

    R = renderer.R
    R_ss = renderer.R_ss
    N_ss = renderer.N_ss
    ang_width = renderer.shadow_model.angular_width
    ang_width_deg = np.rad2deg(ang_width)
    omega_ss = renderer.shadow_model.omega_ss

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title("Ringworld Shadow Square System (Animated) - Polar Method")
    ax.set_aspect('equal')
    ax.set_xlim(-1.1*R, 1.1*R)
    ax.set_ylim(-1.1*R, 1.1*R)
    ax.set_xlabel("X Position (miles)")
    ax.set_ylabel("Y Position (miles)")
    ax.grid(True, alpha=0.3)

    # Plot the static elements
    ring_circle = plt.Circle((0, 0), R, color='green', fill=False, linewidth=2, label='Ringworld')
    ax.add_patch(ring_circle)
    
    # Shadow square orbit
    ss_orbit = plt.Circle((0, 0), R_ss, color='gray', fill=False, 
                          linewidth=1, linestyle='--', alpha=0.5, label='SS Orbit')
    ax.add_patch(ss_orbit)

    sun = plt.Circle((0, 0), renderer.R_sun*5, color='yellow', zorder=5, label='Sun')
    ax.add_patch(sun)

    # Observer position
    obs_y = -R
    observer_plot = ax.plot(0, obs_y, 'ro', markersize=8, zorder=10, label='Observer')[0]

    # Add legend
    ax.legend(loc='upper right', fontsize=10)

    # Add text for time
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=14,
                       verticalalignment='top', 
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Animation parameters - calculate time step in seconds
    # time_step_seconds = 60  # Update every minute of simulation time
    total_frames = int(duration_hours * 3600 / time_step_seconds)
    interval_ms = int(1000 / fps)  # Milliseconds between frames
    
    # Store references to dynamic elements (lines and patches)
    dynamic_elements = []

    def animate(frame):
        nonlocal dynamic_elements
        
        # Calculate time in seconds
        time_sec = frame * time_step_seconds
        hours = time_sec / 3600
        minutes = (time_sec % 3600) / 60

        # Remove previous dynamic elements
        for elem in dynamic_elements:
            if hasattr(elem, 'remove'):
                elem.remove()
            elif isinstance(elem, list):  # Line2D objects
                for line in elem:
                    line.remove()
        dynamic_elements = []

        # Update shadow squares for this time
        for i in range(N_ss):
            theta_center = (i + 0.5) * (2.0 * np.pi / N_ss) + omega_ss * time_sec
            theta_deg = np.rad2deg(theta_center)

            # Map theta to matplotlib Wedge (South aligned)
            w_theta = theta_deg - 90

            t1 = w_theta - ang_width_deg/2
            t2 = w_theta + ang_width_deg/2

            col = constants.SS_COLORS[i % len(constants.SS_COLORS)]

            # Shadow Square - represented as thin lines at the edges
            theta_left_rad = np.deg2rad(t1 + 90)  # Convert back to standard angle
            theta_right_rad = np.deg2rad(t2 + 90)
            
            # Inner and outer radius of shadow square
            r_inner = R_ss - 3e6*1609
            r_outer = R_ss
            
            # Left edge line
            x_left = [r_inner * np.cos(theta_left_rad), r_outer * np.cos(theta_left_rad)]
            y_left = [r_inner * np.sin(theta_left_rad), r_outer * np.sin(theta_left_rad)]
            line_left = ax.plot(x_left, y_left, color=col, linewidth=2, alpha=0.9)
            dynamic_elements.extend(line_left)
            
            # Right edge line
            x_right = [r_inner * np.cos(theta_right_rad), r_outer * np.cos(theta_right_rad)]
            y_right = [r_inner * np.sin(theta_right_rad), r_outer * np.sin(theta_right_rad)]
            line_right = ax.plot(x_right, y_right, color=col, linewidth=2, alpha=0.9)
            dynamic_elements.extend(line_right)
            
            # Outer arc connecting the two edges
            arc_angles = np.linspace(theta_left_rad, theta_right_rad, 20)
            x_arc = r_outer * np.cos(arc_angles)
            y_arc = r_outer * np.sin(arc_angles)
            line_arc = ax.plot(x_arc, y_arc, color=col, linewidth=2, alpha=0.9)
            dynamic_elements.extend(line_arc)

            # Projected Shadow on Ring
            ring_shadow_arc = Wedge((0, 0), R, t1, t2, width=0.08*R, color='black', alpha=0.5)
            ax.add_patch(ring_shadow_arc)
            dynamic_elements.append(ring_shadow_arc)

        # Add shaded visible area
        visible_points = compute_visible_region_polar(time_sec, renderer, fov, yaw, pitch)
        if visible_points and len(visible_points) > 2:
            visible_wedge = Polygon(visible_points, closed=True, color='red', alpha=0.25,
                                   edgecolor='red', linewidth=1.5, zorder=8)
            ax.add_patch(visible_wedge)
            dynamic_elements.append(visible_wedge)

        # Update time text
        time_text.set_text(f'Time: {hours:.2f}h ({int(hours)}h {int(minutes)}m)')

        return dynamic_elements + [time_text]

    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=total_frames,
                                 interval=interval_ms, blit=False, repeat=True)

    return fig, anim