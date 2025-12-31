# Ringworld Renderer: Code Cleanup & Refactoring Plan

A structured approach to improve maintainability, clarity, and conciseness across the codebase.

---

## Phase 1: Centralize Derived Constants
**Estimated Effort**: Small | **Risk**: Low

The quickest win with immediate clarity improvements. Fixes magic numbers and duplicate calculations.

### [MODIFY] [constants.py](file://src/ringworld_renders/constants.py)

Add missing physical constants:
- `RIM_WALL_HEIGHT_MILES = 1000.0` (currently hardcoded in `core.py:317,334`)
- `RIM_WALL_HEIGHT_METERS = RIM_WALL_HEIGHT_MILES * MILES_TO_METERS`

### [MODIFY] [core.py](file://src/ringworld_renders/core.py)

1. **Remove duplicate line** at 384-385: `r_inner = self.R - self.H_a` appears twice
2. **Replace hardcoded rim wall height** with `constants.RIM_WALL_HEIGHT_METERS`
3. **Cache `center_y`** as `self.center_y` in [__init__](file://src/ringworld_renders/core.py#6-51) (used 8+ times across methods)

---

## Phase 2: Consolidate Entry Points
**Estimated Effort**: Small | **Risk**: Low

Fixes a broken script reference and simplifies the CLI architecture.

### [MODIFY] [pyproject.toml](file://pyproject.toml)

Current state has 5 entry points, one of which is broken:
```toml
ringworld-visualize = "ringworld_renders.tools.visualize_shadows:create_visualization"  # ❌ Function doesn't exist
```

Consolidate to a single CLI with subcommands
```toml
[project.scripts]
ringworld = "ringworld_renders.main:main"
```

### [MODIFY] [main.py](file://src/ringworld_renders/main.py)

- Remove [run_ui()](file://src/ringworld_renders/main.py#76-80), [run_verify()](file://src/ringworld_renders/main.py#81-85), [run_samples()](file://src/ringworld_renders/main.py#86-90) wrapper functions (hacky `sys.argv` manipulation)
- Add `visualize` subcommand to existing argparse

### [MODIFY] [visualize_shadows.py](file://src/ringworld_renders/tools/visualize_shadows.py)

Add missing entry point function:
```python
def create_visualization():
    """Entry point for ringworld-visualize command."""
    # Standalone matplotlib visualization
    pass
```

---

## Phase 3: Remove Dead Comments & Fix Inconsistencies
**Estimated Effort**: Small | **Risk**: Low

Cleanup pass for stale comments, numbering errors, and misleading documentation.

### [MODIFY] [core.py](file://src/ringworld_renders/core.py)

| Line | Issue | Fix |
|------|-------|-----|
| 108-109 | Empty lines between if statement and logic | Remove blank line |
| 593 | Comment "3. Rim Wall Color" | Renumber to "2." |
| 598 | Comment "3. Ring Color" | Renumber to "3." |
| 618 | Comment "3. Atmospheric Effects" | Renumber to "4." |

### [MODIFY] [ui.py](file://src/ringworld_renders/ui.py)

| Lines | Issue | Fix |
|-------|-------|-----|
| 99-131 | Verbose Gradio animation workaround comments | Condense to 2-3 lines |

### [MODIFY] [visualize_shadows.py](file://src/ringworld_renders/tools/visualize_shadows.py)

| Line | Issue | Fix |
|------|-------|-----|
| 52 | Magic number `3e6*1609` | Use `constants.MILES_TO_METERS * 3e6` or define named constant |

### [MODIFY] [context_handoff.md](file://context_handoff.md)

| Line | Issue | Fix |
|------|-------|-----|
| 19 | Stale "NOTE: The codebase has shifted" | Remove or update to reflect current state |

---

## Phase 4: Extract [core.py](file://src/ringworld_renders/core.py) into Separate Modules
**Estimated Effort**: Large | **Risk**: High

During Phase 4 (module extraction), we can design cleaner interfaces where extracted modules always expect batch inputs, and single-ray convenience lives only in the public [Renderer](file://src/ringworld_renders/core.py#5-707) API.

> [!WARNING]
> This is a significant structural change. Should be done incrementally with full test coverage before and after each step.

### Proposed Structure

```
src/ringworld_renders/
├── __init__.py
├── constants.py           # (existing)
├── main.py                # (existing)
├── ui.py                  # (existing)
├── renderer.py            # [NEW] Main Renderer class (orchestration + render())
├── intersections.py       # [NEW] All intersect_* methods
├── atmosphere.py          # [NEW] get_atmospheric_effects, integrated_tau
├── shadows.py             # [NEW] get_shadow_factor, omega_ss, angular_width
└── tools/
    └── visualize_shadows.py  # (existing)
```

### Design Decisions

**[get_color()](file://src/ringworld_renders/core.py#525-638) stays in [renderer.py](file://tests/test_renderer.py)** — It's orchestration logic, not physics:
1. Calls intersection methods → determines what was hit
2. Assigns surface colors based on hit type  
3. Applies atmospheric blending

After refactor, it delegates to extracted modules:
```python
def get_color(self, ...):
    t_sun = self.intersector.sun(ray_origin, ray_directions)
    t_ring = self.intersector.ring(ray_origin, ray_directions)
    # ... determine hit priorities ...
    transmittance, scattering = self.atmosphere.get_effects(...)
    # ... blend and return ...
```

**Single-ray convenience in public API only** — Extracted modules always expect batch inputs (2D arrays). The `Renderer.get_color()` and `Renderer.render()` methods handle the `is_single` wrapping/unwrapping at the boundary.

### Extraction Order

1. **Extract [shadows.py](file://src/ringworld_renders/tools/visualize_shadows.py)** (lowest coupling)
   - [omega_ss](file://src/ringworld_renders/core.py#70-75) property → function [omega_ss(N_ss)](file://src/ringworld_renders/core.py#70-75)
   - [angular_width](file://src/ringworld_renders/core.py#76-80) property → function [angular_width(L_ss, R_ss)](file://src/ringworld_renders/core.py#76-80)
   - [get_shadow_factor()](file://src/ringworld_renders/core.py#232-275) method → `ShadowModel` class

2. **Extract `intersections.py`**
   - [_solve_quadratic_vectorized()](file://src/ringworld_renders/core.py#52-70) → module-level utility
   - [intersect_ring()](file://src/ringworld_renders/core.py#81-131), [intersect_sun()](file://src/ringworld_renders/core.py#132-161), [intersect_shadow_squares()](file://src/ringworld_renders/core.py#162-231), [intersect_rim_walls()](file://src/ringworld_renders/core.py#276-346) → `Intersector` class

3. **Extract `atmosphere.py`**
   - [get_atmospheric_effects()](file://src/ringworld_renders/core.py#347-519) → `AtmosphereModel` class
   - [integrated_tau()](file://src/ringworld_renders/core.py#429-449) (nested function → module function)

4. **Refactor [renderer.py](file://tests/test_renderer.py)**
   - `Renderer.__init__()` — instantiates helper classes
   - [get_color()](file://src/ringworld_renders/core.py#525-638) — orchestrates the pipeline (unchanged logic, cleaner calls)
   - [_render_cached()](file://src/ringworld_renders/core.py#639-685) / [render()](file://src/ringworld_renders/core.py#687-707) — camera logic

### Migration Pattern

```python
# renderer.py (after refactor)
from ringworld_renders.intersections import Intersector
from ringworld_renders.atmosphere import AtmosphereModel
from ringworld_renders.shadows import ShadowModel

class Renderer:
    def __init__(self, ...):
        # Physical params...
        self.intersector = Intersector(self.R, self.h, self.W, ...)
        self.atmosphere = AtmosphereModel(self.H_a, self.H_scale, ...)
        self.shadows = ShadowModel(self.N_ss, self.R_ss, ...)
    
    def get_color(self, ray_origin, ray_directions, ...):
        # Handle single-ray convenience at this boundary
        is_single = ray_directions.ndim == 1
        if is_single:
            ray_directions = ray_directions[None, :]
        
        # Delegate to extracted modules (always batch)
        t_sun = self.intersector.sun(ray_origin, ray_directions)
        # ... rest of orchestration ...
        
        return result[0] if is_single else result
```

---

## Verification Plan

### Automated Tests

Run full test suite after each phase:
```bash
uv run pytest -v
```

Key regression tests:
- [test_coordinate_system_physical](file://tests/test_renderer.py#100-122) — validates intersection geometry
- [test_atmospheric_depth_scaling](file://tests/test_renderer.py#64-81) — validates atmosphere extraction
- [test_default_viewpoint_content](file://tests/test_renderer.py#138-162) — E2E rendering sanity check

### Manual Verification

After Phase 4 (structural changes):
```bash
uv run ringworld-render --ui
```
Verify:
- UI launches without errors
- Default viewport shows ground, horizon, and arch
- Shadow squares animate correctly with time slider

---

## Summary

| Phase | Scope | Effort | Risk | Dependencies |
|-------|-------|--------|------|--------------|
| 1. Constants | [constants.py](file://src/ringworld_renders/constants.py), [core.py](file://src/ringworld_renders/core.py) | Small | Low | None |
| 2. Entry Points | [pyproject.toml](file://pyproject.toml), [main.py](file://src/ringworld_renders/main.py), [visualize_shadows.py](file://src/ringworld_renders/tools/visualize_shadows.py) | Small | Low | None |
| 3. Comments | [core.py](file://src/ringworld_renders/core.py), [ui.py](file://src/ringworld_renders/ui.py), [visualize_shadows.py](file://src/ringworld_renders/tools/visualize_shadows.py) | Small | Low | None |
| 4. Module Extraction | New files + [core.py](file://src/ringworld_renders/core.py) → [renderer.py](file://tests/test_renderer.py) | Large | High | Phases 1-3 |

Phases 1-3 can be done independently in any order. Phase 4 should follow after Phase 3 is complete.
