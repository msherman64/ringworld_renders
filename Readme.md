Ringworld Visualization Project: Architecture & Design Document
1. Motivation & Philosophy
The primary goal is to visualize the "Arch" (the distant side of the Ringworld) in a way that feels physically grounded. By placing familiar terrestrial terrain (e.g., the White Mountains) inside the ring, we provide the viewer with a necessary frame of reference.
Core Philosophy:
 * Geometric Truth over Photorealism: We prioritize accurate angular size, horizon curvature, and atmospheric extinction over texture resolution.
 * Transparency: The physics and math should be readable in the code (NumPy) rather than hidden in a black-box engine.
 * Relatability: The user must feel they standing on a real mountain, looking up at an impossible sky.
2. Design Considerations
The "Scale Gap" Challenge
We are dealing with numbers that generally do not play well together:
 * Ring Radius: ~93,000,000 miles.
 * Terrain Detail: ~0.001 miles (approx. 5 feet).
Robustness Strategy:
We must stick to 64-bit floating point precision (standard in Python float / NumPy float64). We will define a global coordinate system where the origin (0,0,0) is the local camera position on the ring floor, rather than the sun. This prevents floating-point jitter where large coordinates swallow small details.
Coordinate Systems
To make the math understandable, we will distinguish between:
 * Ring Space (Cylindrical): (r, \theta, z) — Useful for defining the macro structure (Arch, Shadow Squares, Rim Walls).
 * Local Tangent Space (Cartesian): (x, y, z) — Useful for the terrain, ray marching, and camera orientation. (0,1,0) is "Up" (towards the hub).
3. Modular Architecture Plan
We will refactor the monolithic script into three logical domains.
Module A: The Macro-Structure (The Ring)
This module defines the "sky" and the "horizon."
 * Inputs: Radius (R), Width (W), Wall Height (H), Sun position.
 * Responsibilities:
   * Analytically calculate where a view ray intersects the distant ring floor (The Arch).
   * Calculate intersections with Shadow Squares (for lighting/eclipses).
   * High Impact: This handles the "Arch fade" into the atmosphere.
Module B: The Micro-Structure (The Terrain)
This module defines the ground beneath our feet.
 * Inputs: A function or heightmap H(x, z).
 * Responsibilities:
   * Map (x, z) coordinates to height y.
   * Upgradability: Currently, you use Gaussian peaks. This module allows us to easily swap that for a HeightmapLoader that reads real NASA SRTM data for the Presidential Range without breaking the renderer.
   * Curvature Correction: It must slightly curve the flat terrain data to match the Ring's huge radius (optional, but rigorous).
Module C: The Engine (The Ray Tracer)
The simplified rendering loop.
 * Setup: Camera position (vector), Orientation (quaternions or Euler angles), FOV.
 * Loop:
   * Ray Generation: Create vectors for every pixel.
   * Local March: Ray-march nearby terrain (0 to ~100 miles).
   * Global Cast: If the ray misses local terrain, ray-trace against the Macro-Structure (The Arch).
   * Atmosphere Pass: Apply Beer-Lambert fog based on the total distance the ray traveled.
4. Visual Focus Areas (High Impact)
To achieve the visual goals you listed, we should prioritize these specific rendering features:
A. The "Impossible" Horizon
On Earth, the horizon drops away due to curvature. On the Ring, the horizon rises.
 * The Math: We need a function that explicitly handles the "vanishing point" where the floor meets the sky. This is visually distinct from Earth and is the primary cue that we are on a ring.
B. Atmospheric Depth (The "Blue" Arch)
The Arch is 180 million miles away across the diameter. It shouldn't look like solid ground; it should look like a sliver of solidified sky.
 * Refinement: We need a multi-layer atmosphere model. Thicker air near the floor, thinner air high up. This ensures the top of the Arch looks different than the bottom (the "feet" of the Arch).
C. Shadow Square Transit
This creates "day" and "night."
 * Implementation: A function is_in_shadow(point_on_ring) that checks the line of sight to the central sun against the periodic shadow squares. This allows us to render the "edge of night" sweeping across the landscape.
