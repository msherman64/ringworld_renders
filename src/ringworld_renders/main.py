import argparse
import os
import sys
import time
import numpy as np
import PIL.Image
from ringworld_renders.core import Renderer
from ringworld_renders.ui import create_ui, CSS


def generate_samples(renderer, resolution=1024):
    """Generate high-resolution production samples."""
    print(f"\n--- Generating Production Samples ({resolution}x{resolution}) ---")
    os.makedirs("output", exist_ok=True)
    
    samples = [
        ("High Noon", 0, "production_noon_1024.png"),
        ("Sunset", 6 * 3600, "production_sunset_1024.png"),
        ("Midnight", 12 * 3600, "production_night_1024.png"),
    ]
    
    for name, t_sec, filename in samples:
        print(f"Rendering {name}...")
        t0 = time.time()
        img = renderer.render(width=resolution, height=resolution, fov=95, 
                             look_at=np.array([1.0, 1.0, 0.0]), time_sec=t_sec)

        print(f"  Complete in {time.time() - t0:.2f}s")
        PIL.Image.fromarray(img).save(os.path.join("output", filename))

def run_physical_verification(renderer):
    """Run physical consistency checks."""
    print("\n--- Physical Verification ---")
    center_y = renderer.R - renderer.h
    hit_p = np.array([0.0, center_y - renderer.R, 0.0]) # Observer ground
    
    # Sunset checking
    sunset_start = None
    sunset_end = None
    for m in range(0, 40):
        t_sec = (6 * 3600) + (m - 20) * 60
        s = renderer.get_shadow_factor(hit_p, t_sec)
        if sunset_start is None and s < 0.999:
            sunset_start = m
        if sunset_start is not None and sunset_end is None and s < 0.001:
            sunset_end = m
    
    if sunset_start is not None and sunset_end is not None:
        duration = sunset_end - sunset_start
        print(f"Verified Sunset Duration: {duration} minutes (Target: ~26 min)")
    else:
        print("Error: Could not determine sunset duration.")

def main():
    parser = argparse.ArgumentParser(
        description="Ringworld Renderer - Physically accurate Ringworld visualization",
        prog="ringworld"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # UI command
    ui_parser = subparsers.add_parser("ui", help="Launch the interactive Gradio UI")
    ui_parser.set_defaults(func=run_ui_command)

    # Samples command
    samples_parser = subparsers.add_parser("samples", help="Generate high-quality sample images")
    samples_parser.add_argument("--res", type=int, default=1024,
                               help="Resolution for sample generation (default: 1024)")
    samples_parser.set_defaults(func=run_samples_command)

    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Run physical consistency checks")
    verify_parser.set_defaults(func=run_verify_command)

    # Visualize command
    visualize_parser = subparsers.add_parser("visualize", help="Launch shadow square visualization")
    visualize_parser.set_defaults(func=run_visualize_command)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    # Call the appropriate function
    args.func(args)


def run_ui_command(args):
    """Launch the interactive Gradio UI."""
    print("Launching Ringworld Renderer UI...")
    demo = create_ui()
    demo.launch(css=CSS)


def run_samples_command(args):
    """Generate high-quality sample images."""
    renderer = Renderer()
    generate_samples(renderer, args.res)


def run_verify_command(args):
    """Run physical consistency checks."""
    renderer = Renderer()
    run_physical_verification(renderer)


def run_visualize_command(args):
    """Launch shadow square visualization."""
    launch_visualization()

# Legacy entry points for backward compatibility
def run_ui():
    """Legacy entry point for ringworld-ui command."""
    print("Note: ringworld-ui is deprecated. Use 'ringworld ui' instead.")
    run_ui_command(None)

def run_verify():
    """Legacy entry point for ringworld-verify command."""
    print("Note: ringworld-verify is deprecated. Use 'ringworld verify' instead.")
    run_verify_command(None)

def run_samples():
    """Legacy entry point for ringworld-samples command."""
    print("Note: ringworld-samples is deprecated. Use 'ringworld samples' instead.")
    # For legacy compatibility, parse --res argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--res", type=int, default=1024)
    legacy_args, _ = parser.parse_known_args()
    run_samples_command(legacy_args)


def launch_visualization():
    """Launch standalone shadow square visualization."""
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from ringworld_renders.tools.visualize_shadows import render_system_plot

    print("Launching Shadow Square Visualization...")
    print("Close the window to exit.")

    renderer = Renderer()
    R = renderer.R

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title("Ringworld Shadow Square System (Animated)")
    ax.set_aspect('equal')
    ax.set_xlim(-1.1*R, 1.1*R)
    ax.set_ylim(-1.1*R, 1.1*R)
    ax.set_xlabel("X Position (miles)")
    ax.set_ylabel("Y Position (miles)")

    # Plot the static ring
    ring_circle = plt.Circle((0, 0), R, color='green', fill=False, linewidth=2, label='Ringworld')
    ax.add_patch(ring_circle)

    # Add text for time
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=12,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Animation parameters
    fps = 10
    total_duration_hours = 24
    total_frames = fps * total_duration_hours * 3600 // 3600  # 1 frame per hour

    def animate(frame):
        # Calculate time in seconds (1 frame = 1 hour)
        time_sec = frame * 3600
        hours = time_sec / 3600

        # Clear previous shadow squares
        for patch in ax.patches[:]:
            if patch != ring_circle:  # Keep the ring
                patch.remove()

        # Get system plot for this time
        system_img = render_system_plot(time_sec, renderer)

        # Convert PIL image to matplotlib
        import matplotlib.image as mpimg
        from io import BytesIO

        buf = BytesIO()
        system_img.save(buf, format='png')
        buf.seek(0)
        img_array = mpimg.imread(buf)

        # Display the system plot
        ax.imshow(img_array, extent=[-1.1*R, 1.1*R, -1.1*R, 1.1*R], alpha=0.7, zorder=1)

        # Update time text
        time_text.set_text('.1f')

        return []

    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=total_frames,
                                 interval=1000//fps, blit=False, repeat=True)

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
