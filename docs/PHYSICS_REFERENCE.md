# Physics Reference

Quick reference for physical parameters, constants, and key equations used in the Ringworld Renderer. For detailed explanations, see [Technical Design](TECHNICAL_DESIGN.md).

## Physical Parameters

### The Ring Structure

| Parameter | Symbol | Value | Unit |
| :--- | :--- | :--- | :--- |
| **Radius** | $R$ | $92,955,807.0$ | miles (1 AU) |
| **Circumference** | $C$ | $\approx 584,000,000.0$ | miles |
| **Width** | $W$ | $1,000,000.0$ | miles |
| **Rim Wall Height** | $H_w$ | $1,000.0$ | miles |
| **Gravity (Centrifugal)**| $g$ | $1.0$ | G ($9.8 m/s^2$) |
| **Rotational Velocity** | $v_{rot}$ | $770.0$ | miles/sec |

### The Atmosphere & Optics

| Parameter | Symbol | Value | Unit |
| :--- | :--- | :--- | :--- |
| **Sun Diameter** | $D_{sun}$ | $864,948$ | miles |
| **Sun Radius** | $R_{sun}$ | $432,474$ | miles |
| **Sun Angular Diameter**| $\sigma$ | $0.53^\circ$ | degrees |
| **Effective Ceiling** | $H_a$ | $100.0$ | miles |
| **Zenith Opacity** | $\tau_{zenith}$ | $0.06$ | unitless |
| **Scattering Model** | - | Rayleigh ($1/\lambda^4$) | - |

### The Shadow Square Assembly

| Parameter | Symbol | Value | Unit |
| :--- | :--- | :--- | :--- |
| **Square Count** | $N$ | $8$ | units |
| **Orbital Radius** | $R_{ss}$ | $36,571,994$ | miles |
| **Square Width** | $L_{ss}$ | $14,360,000$ | miles |
| **Square Height** | $H_{ss}$ | $1,200,000$ | miles |
| **Orbital Direction**| - | **Retrograde** | - |
| **Solar Day** | $T_{day}$ | $24.0$ | hours |

### Day/Night Cycle

Optimized for a 24-hour cycle:
- **Night** (0-50% solar illumination): 12.0 hours
- **Day** (50-100% solar illumination): 12.0 hours
- **Twilight periods**: Each sunrise and sunset lasts ~26.3 minutes

**Timing Breakdown:**
- Full darkness (umbra): 11.12 hours
- Sunrise (0% → 50% → 100%): 0.44 hours (26.3 min)
- Full daylight: 11.12 hours
- Sunset (100% → 50% → 0%): 0.44 hours (26.3 min)

### Observer Specification

- **Standard Eye Height ($h$):** $0.00189394$ miles ($10.0$ feet)
- **Field of View (FOV):** $110.0^\circ$ (Optimized for Arch and Rim Wall visibility)

**Local Coordinate Frame:**
- **Origin $(0,0,0)$:** The Observer's Eyes
- **$+y$ Axis:** Zenith (pointing toward the Sun/Arch)
- **$-y$ Axis:** Nadir (pointing toward the local ground)

## Key Equations

### Delta-R Intersection Solver

To prevent catastrophic cancellation in ray-cylinder intersection:

$$(R-h)^2 - R^2 = -2Rh + h^2 \approx -2Rh$$

Where:
- $R$ = Ring radius (92,955,807 miles)
- $h$ = Observer eye height (0.00189394 miles)

### Shadow Square Angular Position

Angular position of shadow square $i$ at time $T$:

$$\theta_i(T) = \theta_{i,0} + \omega_{ss} \cdot T$$

Where:
- $\theta_{i,0}$ = Initial angular offset
- $\omega_{ss}$ = Angular velocity of shadow squares
- $T$ = Absolute time (seconds since $T=0$)

### Atmospheric Optical Depth

Optical depth along ray path:

$$\tau = \tau_{zenith} \cdot \text{airmass}$$

Where:
- $\tau_{zenith} = 0.06$ (zenith opacity)
- $\text{airmass}$ = Function of zenith angle (increases near horizon)

### Light Transmittance (Beer-Lambert Law)

Transmittance through atmosphere:

$$T = e^{-\tau}$$

Where $\tau$ is the optical depth along the ray path.

### Penumbra Width

Penumbra width on the Ring from shadow square orbit:

$$W_p = D_{sun} \times \frac{R_{ring}}{R_{ss}}$$

At $R_{ss} = 36.57M$ miles, this creates a **1.34-million-mile-wide** penumbra zone.

### Ring-shine Illumination

While 50% of the Ring's surface is in shadow at any instant, approximately **97% of the visible overhead Arch** is illuminated when viewed from the night side (due to viewing geometry—looking "up and across" to the far side).

## Unit Conversions

- **Miles to Meters**: $1 \text{ mile} = 1,609.34 \text{ meters}$
- **Feet to Miles**: $1 \text{ foot} = 0.000189394 \text{ miles}$

## Precision Requirements

- **All calculations**: `float64` (64-bit floating point)
- **Numerical stability**: Across 11 orders of magnitude (inches to hundreds of millions of miles)
- **Time accumulation**: Double-precision absolute epoch ($T_{total}$)

## Observable Implications

The combination of these parameters produces the target visual phenomena:

| Visual Phenomenon | Physical Basis |
|------------------|----------------|
| **Upward Horizon** | 1 AU radius cylindrical geometry, 10-foot eye height |
| **Atmospheric Blue-Out** | 100-mile ceiling, Rayleigh scattering ($1/\lambda^4$), $\tau_{zenith}=0.06$ |
| **Night Walls** | 8 shadow squares, retrograde orbit, $R_{ss}=36.57M$ mi |
| **Ring-shine** | 50% sunlit surface area, ~97% of visible Arch illuminated from night side |
| **Rim Walls** | $H_w=1,000$ mi at $z=±500,000$ mi lateral distance |
| **Volumetric Shadowing** | Shadow state + atmospheric scattering interaction |

## Related Documentation

- [Technical Design](TECHNICAL_DESIGN.md) - Detailed explanations of physical basis and design rationale
- [Architecture](ARCHITECTURE.md) - How these parameters are used in the system
- [Contributing](../CONTRIBUTING.md) - Development workflow and contribution guidelines
- [README](../README.md) - Project overview and quick start

