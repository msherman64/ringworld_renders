# Ringworld Physics Engine: Technical Design Document (Rev 8.0)

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
 

# Ringworld Physics Engine: Technical Design Document (Rev 9.0)

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
