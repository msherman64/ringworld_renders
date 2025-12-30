# Design Document: Ringworld Atmospheric Ray-Tracer (R.A.R.T.)

## 1. Project Motivation
The goal is to render a physically accurate view from the surface of a Ringworld megastructure ($R \approx 93,000,000$ miles). The project focuses on simulating the unique visual paradoxes of this geometry, specifically the "Optical Horizon" (where the ground vanishes into haze rather than curving away) and the "Arch" (the distant side of the ring visible in the sky).

## 2. Global Physical Constants
| Parameter | Value | Motivation |
| :--- | :--- | :--- |
| **Radius (R)** | 92,955,807 miles | 1 AU radius for Earth-standard solar flux. |
| **Width (W)** | 1,000,000 miles | Sufficient area for continental-scale terrain. |
| **Rim Wall (H)** | 1,000 miles | Retains atmosphere via centrifugal force. |
| **Rotational Speed** | ~770 miles/sec | Maintains 1G outward acceleration. |

## 3. Atmospheric & Visibility Model
* **Density Profile:** Exponential decay $\rho(h) = \rho_0 e^{-h/5}$ (Scale Height = 5 miles).
* **Optical Horizon:** Set at ~150–200 miles, where cumulative optical depth $\tau \approx 5.0$ (99% extinction).
* **Scattering:** Rayleigh scattering model for sky-tinting and Arch coloration.
* **Vacuum Gap:** Integration logic must cease once a ray exceeds 100 miles in altitude to prevent vacuum glow.
* **Visual Paradox:** The ground remains locally flat but dissolves into a blue-white haze; the "Arch" rises out of this haze at high angles.

## 4. Shadow Square System
To achieve a **12h Day / 12h Night** cycle using non-rigid, stable orbits:
* **Quantity:** 20 independent squares.
* **Orbit Radius:** 2,500,000 miles from Sun center.
* **Orbital Speed:** ~170 miles/sec (Keplerian stable).
* **Relative Velocity:** ~600 miles/sec (relative to Ring surface).
* **Square Dimensions:** ~1,200,000 miles wide to ensure total Umbra across the Ring's width.
* **Transition (Sunrise/Sunset):** ~24 minutes of linear occultation as the square crosses the Sun's disk.

## 5. Implementation Strategy

### 5.1. Precision & Coordinates
* **Render Space:** Camera-centric $(0,0,0)$ using 64-bit floats to prevent jitter.
* **Solar Space:** Sun located at $(0, -93M, 0)$ relative to camera.
* **Camera Placement:** 20,000 miles from the Rim Wall (480,000 miles from centerline).
* **FOV:** 100° looking Spinward.

### 5.2. Hybrid Rendering Engine
* **Module A: RingGeometry:** Analytical ray-cylinder intersection for the distant Arch.
* **Module B: ShadowSystem:** Boolean/Scalar test for solar occultation by squares.
* **Module C: Ray-Marcher:** "Step-and-sample" for local micro-terrain (<200 miles).
* **Module D: Analytical Leap:** For rays not hitting local terrain, "leap" to the Arch intersection point to calculate distant scattering.

## 6. Visual Objectives
* Capture the "infinite horizon" effect where terrain dissolves into sky color.
* Render the Arch passing behind the Sun's disk.
* Simulate the "Wall of Night" moving across the landscape during the 24-minute sunset.
