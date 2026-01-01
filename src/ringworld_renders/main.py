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
    parser = argparse.ArgumentParser(description="Ringworld Renderer CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # UI command
    ui_parser = subparsers.add_parser("ui", help="Launch the interactive Gradio UI")
    ui_parser.set_defaults(func=lambda args, renderer: launch_ui())

    # Samples command
    samples_parser = subparsers.add_parser("samples", help="Generate production-quality samples")
    samples_parser.add_argument("--res", type=int, default=1024, help="Resolution for sample generation")
    samples_parser.set_defaults(func=lambda args, renderer: generate_samples(renderer, args.res))

    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Run physical consistency checks")
    verify_parser.set_defaults(func=lambda args, renderer: run_physical_verification(renderer))

    # Visualize command
    visualize_parser = subparsers.add_parser("visualize", help="Launch the shadow visualization tool")
    visualize_parser.set_defaults(func=lambda args, renderer: launch_visualization())

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    renderer = Renderer()
    args.func(args, renderer)

def launch_ui():
    """Launch the interactive Gradio UI."""
    print("Launching UI...")
    demo = create_ui()
    demo.launch(css=CSS)

def launch_visualization():
    """Launch the shadow visualization tool."""
    from ringworld_renders.tools.visualize_shadows import create_visualization
    create_visualization()

if __name__ == "__main__":
    main()
