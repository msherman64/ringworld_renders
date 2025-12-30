# Ringworld Physics Engine: Technical Design Document

## Section 1: Motivation, Visual Goals, and Physical Parameters

### 1.1 Engineering Motivation: The Precision Gap
The primary driver for this engine is the reconciliation of astronomical scales with human-scale interaction. Standard 3D rendering architectures are ill-equipped for a structure where a single surface spans eleven orders of magnitude.

* **Floating Point Constraints:** In a Sun-centric (Global) system, a 64-bit float allocating bits to represent $93,000,000$ miles leaves insufficient mantissa bits to represent an observer moving by inches. This leads to "Geometric Jitter."
* **The Intersection Stability Problem:** Calculating intersections via $(R-h)^2 - R^2$ leads to "Catastrophic Cancellation," where the most significant digits of the eye-height $h$ are erased.
* **The Horizon Continuity Goal:** We require a mathematical framework that treats the "Local" ground and the "Distant" Arch as a single, continuous geometric solution, ensuring no seams or precision shifts occur at the point where the ground meets the sky.

### 1.2 Visual Goals (The "Niven" Aesthetic)
The success of the visualization is measured by the emergence of the following phenomena:

* **The Upward Horizon:** A subversion of planetary perspective. The ground must appear to rise as distance increases, eventually becoming the "Sky." 
* **Grounded Contextualization:** To provide a visceral sense of scale, the viewpoint is placed within a **familiar, Earth-like terrain** (meadows, hills, or valleys). This "Earth-bridge" allows the viewer to use local references (like a nearby hill) to comprehend the sheer impossibility of the Arch rising from behind it.
* **Atmospheric "Blue-Out":** Instead of a hard geometric horizon, the landscape must be "swallowed" by the atmosphere. Looking "forward" traverses thousands of miles of air, creating a gradient where the brown/green of the terrain fades into the blue/haze of the sky.
* **Night Walls & Light Columns:** Vertical pillars of darkness cast by the Shadow Squares. A key goal is the **Emergent Night-Sky**: when the observer is in shadow, the sky should not be a void but should reveal the stars and the illuminated Arch overhead.
* **Ring-shine (Ambient Illumination):** The "Night" side of the Ring is never truly dark. Because 95% of the Ring is in direct sunlight at any given moment, the shadowed regions must receive "Ring-shine"—secondary illumination reflected from the distant, lit parts of the Arch.
* **Rim Walls:** Vertical barriers rising 1,000 miles at the lateral edges ($z = \pm 500,000$ miles). These serve as the "frames" of the world, visible as massive, distant mountain ranges on the left and right horizons.
* **Volumetric Shadowing:** When the observer is in shadow, the air itself must reflect this. The engine must distinguish between a sunlit landscape seen through shadowed air and a shadowed landscape seen through sunlit air.

### 1.3 Physical Parameters (The Source of Truth)
All calculations must use `float64` precision for the following constants.

#### 1.3.1 The Ring Structure
| Parameter | Symbol | Value | Unit |
| :--- | :--- | :--- | :--- |
| **Radius** | $R$ | $92,955,807.0$ | miles (1 AU) |
| **Circumference** | $C$ | $\approx 584,000,000.0$ | miles |
| **Width** | $W$ | $1,000,000.0$ | miles |
| **Rim Wall Height** | $H_w$ | $1,000.0$ | miles |
| **Gravity (Centrifugal)**| $g$ | $1.0$ | G ($9.8 m/s^2$) |
| **Rotational Velocity** | $v_{rot}$ | $770.0$ | miles/sec |

#### 1.3.2 The Atmosphere & Optics
| Parameter | Symbol | Value | Unit |
| :--- | :--- | :--- | :--- |
| **Effective Ceiling** | $H_a$ | $100.0$ | miles |
| **Zenith Opacity** | $\tau_{zenith}$ | $0.06$ | unitless |
| **Scattering Model** | - | Rayleigh ($1/\lambda^4$) | - |
| **Sun Angular Diameter**| $\sigma$ | $0.53^\circ$ | degrees |

#### 1.3.3 The Shadow Square Assembly
| Parameter | Symbol | Value | Unit |
| :--- | :--- | :--- | :--- |
| **Square Count** | $N$ | $20$ | units |
| **Orbital Radius** | $R_{ss}$ | $2,500,000.0$ | miles |
| **Square Length** | $L_{ss}$ | $1,000,000.0$ | miles |
| **Orbital Velocity** | $v_{ss}$ | $670.0$ | miles/sec |

### 1.4 Observer Specification
* **Standard Eye Height ($h$):** $0.00189394$ miles ($10.0$ feet).
* **Field of View (FOV):** $110.0^\circ$ (Optimized for Arch and Rim Wall visibility).
* **Local Coordinate Frame:** * **Origin $(0,0,0)$:** The Observer's Eyes.
    * **$+y$ Axis:** Zenith (pointing toward the Sun/Arch).
    * **$-y$ Axis:** Nadir (pointing toward the local ground).

### 1.5 Temporal State Management
* **Problem to Solve:** Maintaining synchronized motion of shadow squares and ring rotation over long durations without floating-point drift.
* **Importance:** At $670$ miles/s, a $0.0001$ deviation in time accumulation results in miles of positional error.
* **Selected Approach: Absolute Epoch Accumulation.**
    * **Description:** All positions are calculated as a function of $T_{total}$ (double precision seconds since start). $T=0$ defines a "High Noon" alignment at the observer's longitude.
    * **How it solves the problem:** It prevents incremental error. Instead of adding `delta_t` to a position, we derive position from an absolute time scalar.

## Section 2: Geometric Logic and Observer-Centric Architecture

### 2.1 The Coordinate Strategy: Origin Placement
* **Problem to Solve:** Establishing a coordinate origin that maintains numerical stability across eleven orders of magnitude (from inches to hundreds of millions of miles).
* **Importance:** A 64-bit float (`float64`) provides approximately 15–17 significant decimal digits. In a system where the radius $R \approx 10^8$ miles, the available precision for local movement in a Sun-centric system is severely limited. High-precision grounding is required to prevent "jitter" in the foreground terrain and ensuring the observer's interaction with the surface is physically consistent.
* **Selected Approach: Observer-Centric Tangent Frame.** * **Description:** We place the origin $(0,0,0)$ at the camera's location. The structure's center is displaced to $(0, R-h, 0)$.
    * **How it solves the problem:** By centering the origin on the observer, the geometry with the highest visual frequency (the local ground, hills, and immediate surroundings) is calculated with maximum floating-point density. 
    * **Why we selected it:** It pushes the inevitable precision loss to the center of the solar system (the sun), where it is visually irrelevant, while keeping the observer's immediate surroundings mathematically "stiff" and stable.
* **Considered & Discarded: Global Cartesian (Sun-Centric).**
    * **Why not:** At $1$ AU from the origin, performing additions or subtractions (like moving a camera by 1 inch) results in **Catastrophic Cancellation**. The large magnitude of the distance "swallows" the small magnitude of the movement, leading to jagged, vibrating geometry and "Z-fighting" artifacts.

### 2.2 The Intersection Kernel: The Delta-R Solver
* **Problem to Solve:** Solving the quadratic equation for a ray-cylinder intersection without losing the significance of the observer's height ($h$).
* **Importance:** The constant term in a standard quadratic intersection ($C = \text{dist}^2 - R^2$) involves subtracting two massive, nearly identical numbers. If the precision of $h$ is lost during this subtraction, the "Ground" will either disappear or jitter with an invisible plane at $h=0$.
* **Selected Approach: The "Delta-R" Expanded Coefficient.** * **Description:** We solve the quadratic $At^2 + Bt + C = 0$ using the expanded form for the constant term: $C = -2Rh + h^2$.
    * **How it solves the problem:** This formula avoids squaring $R$ ($9.3 \times 10^7$) and $R-h$ separately. Instead, it calculates the *difference* directly using the observer's height as the primary variable.
    * **Why we selected it:** It preserves the significant digits of the 10-foot eye-level even when the structure radius is 93 million miles, ensuring the ground remains exactly where the observer expects it.
* **Considered & Discarded: Naive Analytical Intersection.**
    * **Why not:** Implementation of $(R-h)^2 - R^2$ results in the most significant bits of $h$ being truncated during the subtraction of the massive $R^2$ terms.

### 2.3 Solving for Distance: The Stable Quadratic (q-method)
* **Problem to Solve:** Finding two roots ($t_{ground}$ and $t_{arch}$) accurately when one value is extremely small ($\approx 0$) and the other is extremely large ($2R$).
* **Importance:** Using the standard quadratic formula can be numerically unstable when $B$ is much larger than $A$ or $C$, leading to a loss of precision in the smaller root (the ground).
* **Selected Approach: The $q$-Method Solver.** * **Description:** We calculate an intermediate value $q = -0.5 \cdot (B + \text{sign}(B)\sqrt{B^2 - 4AC})$, then derive $t_1 = q / A$ and $t_2 = C / q$.
    * **How it solves the problem:** This method avoids the subtraction of nearly equal terms in the numerator, which is the primary source of rounding error in the standard formula.
    * **Why we selected it:** It provides two stable, independent distances: one for the local floor beneath the observer and one for the far side of the ring (the Arch).
* **Considered & Discarded: Standard Quadratic Formula.**
    * **Why not:** When $B$ is large, the small root ($t_{ground}$) becomes highly susceptible to rounding errors, which would cause the ground at the observer's feet to flicker, tilt, or register as a "miss."

### 2.4 Lateral Constraints: Rim Wall Geometry
* **Problem to Solve:** Truncating the infinite analytical cylinder at the 1-million-mile width boundaries and adding 1,000-mile-high containment walls.
* **Importance:** Without these constraints, the world would have no lateral edges, and the iconic visual effect of the Rim Walls rising as distant mountains on the left and right would be missing.
* **Selected Approach: Plane-Intersection with Radial Validation.** * **Description:** We solve for the ray's intersection with the planes $z = \pm W/2$. If a hit occurs, we validate if that hit is within the wall height $R + H_w$.
    * **How it solves the problem:** It provides a secondary "hit" condition that overrides the cylinder hit if the ray reaches the lateral edge of the structure before the surface curvature.
    * **Why we selected it:** It is an $O(1)$ analytical solution that perfectly maintains the cylindrical symmetry of the structure.
* **Considered & Discarded: Ray Marching (SDFs) for Walls.**
    * **Why not:** While SDFs are flexible for complex shapes, they are too computationally expensive for a distance of 500,000 miles, requiring thousands of sampling steps that an analytical plane-solver handles in a single operation.

### 2.5 Atmospheric Shell Intersection
* **Problem to Solve:** Defining the geometric boundary where the "Sky" ends and "Space" (Starfield) begins.
* **Importance:** Without a ceiling, a ray pointing into the sky that misses the Arch would calculate infinite airmass. We need a physical boundary to terminate the volume integration.
* **Selected Approach: Internal Sphere/Cylinder Intersection.**
    * **Description:** We solve for the intersection with a secondary cylinder at $R_{atmos} = R - H_a$.
    * **How it solves the problem:** This provides a "hit" distance $t_{sky}$ for rays that don't hit the structure. If a ray misses the structure but hits this shell, we calculate the scattering from $0$ to $t_{sky}$ and then render the background starfield.
    * **Why we selected it:** It treats the atmosphere as a physical volume with a defined edge, allowing for crisp transitions between the blue horizon and the black of space.

## Section 3: Surface Mapping and Procedural Data

### 3.1 The Mapping Challenge: Astronomical Surface Area
* **Problem to Solve:** Creating a consistent coordinate system to map textures and terrain onto a surface area of 584 trillion square miles.
* **Importance:** Standard UV mapping techniques fail at this scale. A single texture covering the Ringworld would require a resolution that exceeds modern hardware limits by several orders of magnitude, and linear tiling would create visible patterns and seams.
* **Selected Approach: Normalized Angular-Longitudinal $(\theta, z)$ Mapping.**
    * **Description:** Every 3D hit point $\mathbf{P}$ is converted into two normalized coordinates: $\theta$ (the angle around the ring's circumference) and $z_{norm}$ (the position across the ring's width).
    * **How it solves the problem:** This transform creates a 1:1 mapping between a 3D point and a 2D "map space" that is agnostic to the observer's distance. $\theta$ is derived via `atan2(Px, -(Py - Cy))` and $z_{norm}$ via `Pz / (W/2)`.
    * **Why we selected it:** It allows for "Infinite Resolution" through procedural noise functions. Instead of reading a pixel from a file, the engine samples a mathematical function at $(\theta, z)$, ensuring detail is maintained whether the viewer is looking at the ground or the Arch.
* **Considered & Discarded: Tri-planar Mapping.**
    * **Why not:** While effective for rocks and small terrains, tri-planar mapping in an astronomical cylindrical context creates projection stretching at the "shoulders" of the cylinder and is computationally more expensive than a simple angular transform.

### 3.2 Shadow Square State: Temporal Angular Lookup
* **Problem to Solve:** Determining if a specific point $(\theta, z)$ on the ring is in sunlight or shadow, accounting for the relativistic speed of the shadow squares.
* **Importance:** Shadows (the "Night Walls") move at $670$ miles/s. The calculation must be precise enough to handle the transition at the observer's location (local night) while simultaneously calculating the shadow state of the Arch $1.8 \times 10^8$ miles away.
* **Selected Approach: Phase-Shifted Angular Modulo.**
    * **Description:** The shadow state is calculated by taking the point's angle $\theta$, subtracting the current orbital rotation of the shadow squares $(\omega \cdot t)$, and checking the result against the square/gap ratio using a modulo operator: `(theta - orbit_pos) % (2*pi / N)`.
    * **How it solves the problem:** It reduces a complex multi-body occlusion problem into a single $O(1)$ mathematical check.
    * **Why we selected it:** It is perfectly stable over time and requires no "shadow maps" or depth buffers, which would suffer from precision loss over the millions of miles between the sun and the ring.
* **Considered & Discarded: Shadow Mapping / Ray-Traced Occlusion.**
    * **Why not:** Standard shadow maps cannot handle the depth range (1 AU). Ray-tracing the shadow squares as actual geometry would require additional intersection tests per pixel, whereas the angular modulo is virtually free.



### 3.3 Procedural Terrain: Perturbation of the Radial Root
* **Problem to Solve:** Adding mountains, valleys, and "Earth-like" terrain to the perfectly smooth analytical cylinder.
* **Importance:** A perfectly smooth cylinder lacks the visual cues necessary to communicate scale. Terrain provides the "familiar context" required to ground the observer.
* **Selected Approach: Radial Distance Displacement.**
    * **Description:** We do not modify the geometry; we modify the distance result. After calculating the analytical hit $t$, we sample a multi-octave noise function at the hit's $(\theta, z)$ coordinates and offset $t$ by the noise value.
    * **How it solves the problem:** It allows for the emergence of complex geography without the need for billions of polygons. The "Mountains" on the Arch are generated by the same logic as the "Hills" in the foreground.
    * **Why we selected it:** It maintains the performance of an analytical solver while providing the visual complexity of a high-poly mesh.
* **Considered & Discarded: Geometry Displacing (Vertex Displacement).**
    * **Why not:** Moving actual vertices would require a level of mesh subdivision that would crash any current system. At this scale, the "Geometry" must be implicit in the distance calculation.

### 3.4 Discarded Data Approaches: Tiled Textures
* **Problem to Solve:** Texturing the surface.
* **Considered & Discarded: Seamless Tiling Textures.**
    * **Why not:** Tiling a texture across a 584-million-mile circumference would result in "Pattern Moire"—at a distance, the repetition of the tiles would create a grid-like artifact that destroys the illusion of a natural world. Procedural noise (Perlin/Simplex) is required to ensure every square mile is unique.
 
### 3.5 Seamless Procedural Sampling
* **Problem to Solve:** Preventing a visual "seam" in the terrain where the angular coordinate $\theta$ wraps from $2\pi$ back to $0$.
* **Importance:** In a cylindrical world, the observer can look "all the way around." A discontinuity in the noise function would create a vertical line across the entire world.
* **Selected Approach: 3D Periodic Noise Mapping.**
    * **Description:** We do not sample noise using $(\theta, z)$. Instead, we map $(\theta, z)$ onto a virtual 3D cylinder: $X = \cos(\theta)$, $Y = \sin(\theta)$, $Z = z$.
    * **How it solves the problem:** Because $\cos$ and $\sin$ are continuous, the noise function transitions perfectly across the $0/2\pi$ boundary.
    * **Why we selected it:** It is the standard mathematical solution for mapping 2D planar noise onto a manifold without seams.

## Section 4: Radiative Transfer and Atmospheric Optics

### 4.1 Atmospheric Scattering: The Rayleigh Model
* **Problem to Solve:** Simulating a realistic sky color and "blue-out" effect across astronomical distances.
* **Importance:** On a Ringworld, the "Sky" is the primary depth cue. Without atmospheric scattering, the distant Arch would look like a sharp, dark line against black space. We need the "Sky" to emerge from the air between the observer and the structure.
* **Selected Approach: Analytical Single-Pass Volume Integration.**
    * **Description:** We calculate the "Airmass" along the ray path using the zenith angle. The final color is a blend of the surface color (attenuated by distance) and the "In-scattered" sky color.
    * **How it solves the problem:** It creates the "haze" that naturally obscures the distant parts of the ring. As the ray approaches the horizon (low $v_y$), the airmass increases exponentially, causing the terrain to fade into the haze.
    * **Why we selected it:** It allows for a dynamic sky. Because we calculate it per-pixel based on the observer's shadow state, we can render the transition where the ground is in shadow (night) but the upper atmosphere is still catching sunlight.
* **Considered & Discarded: Skybox / Cube-map Textures.**
    * **Why not:** A static skybox cannot account for the "Night Walls" moving across the sky. In a Ringworld, the "Sky" is part of the world geometry and must change in real-time as the shadow squares move.


### 4.2 Light Extinction: The Beer-Lambert Law
* **Problem to Solve:** Dimming the light from the Arch as it passes through the thick atmosphere near the horizons.
* **Importance:** Without light extinction, the Arch would be at full brightness even when viewed through thousands of miles of air. This would look physically "wrong" and flat.
* **Selected Approach: Exponential Transmittance ($e^{-\tau}$).**
    * **Description:** We calculate the optical depth $\tau$ as a function of the path length and the zenith opacity. The surface light is then multiplied by $T = \exp(-\tau)$.
    * **How it solves the problem:** This provides the mathematical basis for the "Blue-out" effect. It ensures that the further away a part of the ring is, the more its light is absorbed and replaced by scattered sky light.
    * **Why we selected it:** It is the standard physical model for atmospheric visibility and produces the soft, realistic transitions seen in high-altitude photography.

### 4.3 Shadow Softening: The Angular Penumbra
* **Problem to Solve:** Eliminating the "razor-sharp" edges of the shadow squares that occur in a point-light simulation.
* **Importance:** The Sun is a disk, not a point. In reality, a shadow edge is a gradient (the penumbra). A sharp 1-pixel line across a 1,000,000-mile-wide structure would look like a rendering error.
* **Selected Approach: Linear Angular Interpolation.**
    * **Description:** We define the angular width of the sun ($\sigma \approx 0.53^\circ$). When checking the shadow state $(\theta \pmod{\text{period}})$, we check if the point is within $\sigma$ of the shadow edge and interpolate the light intensity.
    * **How it solves the problem:** It softens the "Night Walls," making the transition from day to night feel like a natural sunset/sunrise lasting several seconds rather than an instantaneous flicker.
    * **Why we selected it:** It mimics the effect of a disk-light source with negligible computational overhead compared to multi-sample area lighting.

### 4.4 Ambient Illumination: Ring-shine
* **Problem to Solve:** Preventing the "Night" side of the ring from being pitch black.
* **Importance:** On a Ringworld, the night side is illuminated by the enormous, sunlit Arch overhead. This "Ring-shine" should be bright enough to see by—roughly equivalent to a landscape under multiple full moons.
* **Selected Approach: Constant-Hemisphere Ambient term based on Arch Visibility.**
    * **Description:** We add a base "Ambient" light level to the surface shader that scales with the total illuminated area of the Arch currently visible in the sky.
    * **How it solves the problem:** It provides the subtle green/brown detail to the night-side landscape, grounded by the light "reflected" from the distant side of the structure.
    * **Why we selected it:** It captures the unique global-illumination environment of a Ringworld without the need for expensive ray-traced bounces.

### 4.5 Discarded Optical Approaches: Global Illumination (Path Tracing)
* **Problem to Solve:** Calculating light bounces from the Arch to the floor.
* **Considered & Discarded: Real-time Path Tracing.**
    * **Why not:** While highly accurate, path tracing across $10^8$ miles is currently impossible for real-time applications. Our analytical "Ring-shine" approximation provides $90\%$ of the visual benefit for $0.1\%$ of the computational cost.

### 4.6 Shadow-Sky Coupling (Volumetric Shadowing)
* **Problem to Solve:** Correctly rendering the "Night Wall" within the air itself.
* **Importance:** If a shadow square is blocking the sun, the air in that shadow should not be blue; it should be dark. Conversely, looking "out" from a shadow into a sunlit area should show glowing blue air.
* **Selected Approach: Step-Function Shadow Integration.**
    * **Description:** The in-scattering term ($I$) is multiplied by the shadow factor $S_f$ sampled at the midpoint of the atmospheric path.
    * **How it solves the problem:** It ensures that "Night" feels 3D. The sky overhead darkens as the shadow square passes, rather than just the ground turning black.
    * **Why we selected it:** It provides the "Light Column" effect where the edges of the shadow squares are visible in the haze.
