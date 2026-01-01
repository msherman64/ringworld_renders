# System Architecture

This document provides a high-level overview of the Ringworld Renderer's architecture, components, and design decisions. For detailed technical specifications, see [Technical Design](TECHNICAL_DESIGN.md).

## System Overview

The renderer follows a ray-tracing architecture with analytical intersection solvers, designed to handle astronomical scales (1 AU radius) while maintaining numerical precision for human-scale interactions (10-foot eye height).

## Core Components

### Renderer (`core.py`)

The main orchestration class that:
- Manages physical parameters and configuration
- Coordinates the rendering pipeline: intersection → shading → atmosphere
- Handles camera transforms and coordinate systems
- Provides the public API for rendering frames

### Physics Modules

**Intersections** (`intersections.py`):
- Ray-cylinder intersection with Delta-R solver (prevents catastrophic cancellation)
- Ray-sphere intersection for shadow squares
- Analytical solvers maintaining numerical stability

**Shadows** (`shadows.py`):
- Angular shadow calculations for 8 shadow squares
- Penumbra softening based on solar angular diameter
- Temporal synchronization using absolute epoch accumulation

**Atmospheric Effects** (in `core.py`):
- Rayleigh scattering with $1/\lambda^4$ wavelength dependence
- Beer-Lambert light extinction
- Volumetric shadowing (shadow-sky coupling)

**Constants** (`constants.py`):
- Physical parameters (Ring, Sun, Shadow Squares, Atmosphere)
- Unit conversions
- Configuration values

### User Interface (`ui.py`)

Gradio-based interactive interface providing:
- Real-time camera control
- Time-of-day adjustment
- Visual effects toggles
- System visualization

## Rendering Pipeline

The renderer follows this pipeline for each pixel:

```
1. Ray Generation
   └─> Generate ray from observer's eye position in observer-centric frame

2. Intersection
   └─> Test ray against analytical cylinder (Delta-R solver)
   └─> Returns: hit distance (t), angular position (θ), lateral position (z)

3. Terrain Displacement
   └─> Sample multi-octave procedural noise at (θ, z)
   └─> Offset hit distance t by noise value

4. Shadow Evaluation
   └─> Calculate angular position θ of hit point
   └─> Check against 8 shadow squares (angular modulo)
   └─> Apply penumbra softening based on solar angular diameter

5. Atmospheric Integration
   └─> Calculate airmass along ray path
   └─> Compute in-scattering (Rayleigh model)
   └─> Apply light extinction (Beer-Lambert)
   └─> Sample shadow state at atmospheric path midpoint

6. Illumination
   └─> Direct sunlight (with penumbra factor)
   └─> Ring-shine ambient (scaled by visible illuminated Arch)
   └─> Combine with shadow factor

7. Final Color
   └─> Composite: surface color × transmittance + sky color × shadow factor
```

## Coordinate Systems

### Observer-Centric Frame (Rendering)

- **Origin (0, 0, 0)**: Observer's eye position
- **+Y Axis**: Zenith (pointing toward Sun/Arch)
- **-Y Axis**: Nadir (pointing toward local ground)
- **+X Axis**: Horizon (lateral direction)
- **+Z Axis**: Lateral (along ring width)

**Ring Center**: Displaced to `(0, R-h, 0)` where:
- `R` = Ring radius (92,955,807 miles)
- `h` = Observer eye height (0.00189394 miles ≈ 10 feet)

### Ring Coordinate System (Physics)

- **Angular Position (θ)**: Circumferential angle (0 to 2π)
- **Lateral Position (z)**: Distance along ring width (-W/2 to +W/2)
- **Radial Distance (r)**: Distance from ring center

### Shadow Calculations

- Angular positions in ring coordinate system
- Time-based absolute epoch accumulation
- Modulo arithmetic for wrapping around ring

## Key Design Decisions

### 1. Observer-Centric Coordinates

**Problem**: Floating-point precision loss when representing astronomical scales (1 AU) and human scales (inches) in the same coordinate system.

**Solution**: Place origin at observer's eye position. This maximizes floating-point density for local geometry (highest visual frequency) while pushing precision loss to visually irrelevant solar center.

**Result**: Numerical stability across 11 orders of magnitude (inches to hundreds of millions of miles).

### 2. Delta-R Intersection Solver

**Problem**: Standard quadratic formula `(R-h)² - R²` causes catastrophic cancellation when `h << R`.

**Solution**: Factor to `-2Rh + h² ≈ -2Rh` form, preserving significance of eye height `h`.

**Result**: Stable terrain intersection directly underfoot, no "shimmering ground" artifacts.

### 3. Batch-First Architecture

**Problem**: Performance requirements for real-time rendering.

**Solution**: All physics calculations expect 2D arrays (batch inputs). Single-ray convenience exists only at public API boundaries.

**Result**: Vectorized NumPy operations, 30+ fps interactive rendering.

### 4. Angular Shadow Calculations

**Problem**: Determining shadow state for any point on a 584-million-mile circumference.

**Solution**: Angular modulo calculation - compute θ position, check against shadow square angular spans. No shadow maps or ray-traced occlusion needed.

**Result**: Computationally cheap (single modulo per shadow square), handles astronomical scale.

### 5. Procedural Terrain Displacement

**Problem**: Adding terrain detail without billions of polygons.

**Solution**: Modify intersection distance `t` by sampling procedural noise at hit coordinates. Geometry remains analytical.

**Result**: Rich visual detail (hills, valleys, mountains) with analytical intersection performance.

### 6. Absolute Epoch Time Accumulation

**Problem**: Incremental time updates (`position += velocity * delta_t`) accumulate rounding errors over long simulations.

**Solution**: Calculate all positions as function of absolute time `T_total`: `θ(T) = θ₀ + ω·T`.

**Result**: Long-term temporal stability, no positional drift.

## Data Flow

```
Time (T_total)
    ↓
Shadow Square Positions (angular, 8 squares)
    ↓
Ray (origin, direction) in observer frame
    ↓
Cylinder Intersection → (θ, z, t)
    ↓
Procedural Noise → Displaced t
    ↓
Hit Position (θ, z, r)
    ↓
Shadow State (angular modulo, penumbra)
    ↓
Atmospheric Path (airmass, zenith angle)
    ↓
Scattering + Extinction + Shadow Factor
    ↓
Surface Color + Sky Color
    ↓
Final Pixel Color
```

## Module Responsibilities

| Module | Responsibility |
|--------|---------------|
| `core.py` | Orchestration, parameter management, atmospheric effects, public API |
| `intersections.py` | Ray-geometry intersections (cylinder, sphere) |
| `shadows.py` | Shadow square physics, angular calculations |
| `constants.py` | Physical constants, configuration |
| `ui.py` | Interactive Gradio interface |

## Performance Characteristics

### Computational Bottlenecks

1. **Per-pixel ray-cylinder intersection** (Delta-R solver)
2. **Multi-octave procedural noise sampling** (terrain generation)
3. **Atmospheric integration** (single-pass volume integral)
4. **Shadow state evaluation** (8 angular modulo checks per pixel)

### Optimization Strategies

- **Early Ray Termination**: Rays missing cylinder terminate immediately
- **Vectorized Operations**: NumPy broadcasting for batch calculations
- **Caching**: UI uses `lru_cache` on render functions
- **Shadow Culling**: Only check shadow squares within angular range

### Target Performance

- **Resolution**: 1920×1080 or higher
- **Frame Rate**: 30+ fps for interactive exploration
- **Scene Complexity**: Full Ringworld visible, no artificial culling

## Related Documentation

- [Technical Design](TECHNICAL_DESIGN.md) - Detailed technical specifications and design rationale
- [Physics Reference](PHYSICS_REFERENCE.md) - Physical parameters, constants, and equations
- [Contributing](../CONTRIBUTING.md) - Development workflow and contribution guidelines
- [README](../README.md) - Project overview and quick start

