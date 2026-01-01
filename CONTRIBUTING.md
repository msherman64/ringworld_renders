# Contributing to Ringworld Renderer

Thank you for your interest in contributing to the Ringworld Renderer! This project aims to create physically accurate visualizations of a Niven-scale Ringworld, balancing scientific accuracy with visual impact.

## Development Philosophy

This codebase was initially LLM-generated but designed for human validation and maintenance. Our focus is on **accuracy first, performance second, clarity third**. Every visual feature must be scientifically validated.

## Quick Start

### Prerequisites
- **Python 3.13+** (uses modern typing features)
- **uv** package manager (`pip install uv`)
- Basic understanding of computer graphics and atmospheric physics

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd ringworld_renders

# Install dependencies
uv sync

# Run tests to verify setup
uv run pytest

# Launch the interactive UI
uv run ringworld ui
```

## Development Workflow

### 1. Verification First 🔬

**Every visual feature must be scientifically validated before implementation.**

- **Prove correctness**: Use physics equations, reference implementations, or empirical validation
- **Regression testing**: Use `tests/test_visual_features.py` for visual feature validation
- **Numerical validation**: Create small verification scripts to compare RGB values, distances, etc.
- **Physical consistency**: Validate against real atmospheric physics, orbital mechanics, and optics

**Example**: When implementing atmospheric scattering, first validate against Rayleigh scattering theory, then verify the visual result matches expected blue-out effects.

### 2. Performance Guidelines ⚡

This is a real-time renderer - performance matters.

- **No Python loops**: Use NumPy broadcasting for all vectorized operations
- **Vectorized first**: All physics calculations must handle batch inputs (2D arrays)
- **Single-ray convenience**: Convenience for single rays lives only in public APIs
- **Caching**: UI uses `lru_cache` on render functions for interactivity

**Performance targets**: 30+ fps for interactive exploration, real-time shadow animation.

### 3. Communication Standards 📝

- **Be concise**: Don't over-explain standard concepts
- **Focus on results and proof**: "This produces X% more accurate shadows" vs "This is good"
- **Document validation**: Include references to physics equations, test results, or validation methods
- **Visual evidence**: Screenshots, before/after comparisons, numerical metrics

## Testing Strategy

### Test Categories
- **Unit tests**: Individual physics functions (`tests/test_renderer.py`)
- **Integration tests**: End-to-end rendering (`test_default_viewpoint_content`)
- **Visual regression**: Color accuracy, shadow contrast (`tests/test_visual_features.py`)
- **Physical validation**: Orbital mechanics, atmospheric physics (`tests/test_sun_occlusion.py`)

### Running Tests
```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_visual_features.py

# Run with coverage
uv run pytest --cov=src/ringworld_renders
```

### Adding Tests
- New physics features require corresponding tests
- Visual features need regression tests with reference images/values
- Performance-critical code needs benchmarks

## Code Standards

### Architecture
```
src/ringworld_renders/
├── core.py          # Main Renderer class (orchestration, atmospheric effects)
├── intersections.py # Ray-geometry intersection calculations
├── shadows.py       # Shadow square physics and rendering
├── constants.py     # Physical constants and configuration
└── ui.py           # Gradio interface
```

### Style Guidelines
- **Type hints everywhere**: Use modern Python typing
- **Descriptive names**: `shadow_factor` not `sf`, `atmospheric_scattering` not `scatter`
- **Docstrings**: Google/NumPy style with parameter descriptions and units
- **Modular design**: Single responsibility per module/class/method
- **No magic numbers**: Everything in `constants.py` or clearly documented

### Physics Accuracy
- **Float64 precision**: All calculations use 64-bit floats for astronomical scales
- **Observer-centric coordinates**: Prevents floating-point precision loss
- **Validated algorithms**: Reference implementations or peer-reviewed sources
- **Unit consistency**: Miles vs meters clearly documented and converted properly

## Contributing Features

### Process
1. **Open an issue** describing the feature and its scientific basis
2. **Create a verification plan** - how will you prove this is correct?
3. **Implement with tests** - TDD approach preferred
4. **Validate visually** - screenshots, numerical comparisons
5. **Document thoroughly** - physics basis, validation results

### Feature Types

#### Visual Features
- Must demonstrate measurable visual improvement
- Include before/after comparisons
- Validate against real atmospheric/astronomical phenomena

#### Performance Features
- Include benchmarks before/after
- Maintain visual quality
- Document trade-offs

#### Algorithm Improvements
- Prove mathematical correctness
- Maintain or improve accuracy
- Include performance comparison

### Code Review Checklist
- [ ] Tests pass and coverage maintained
- [ ] Type hints complete and accurate
- [ ] Performance impact documented
- [ ] Visual validation included
- [ ] Physics accuracy verified
- [ ] Documentation updated

## Architecture Overview

For detailed architecture information, see:
- **[Architecture](docs/ARCHITECTURE.md)** - System design, components, and data flow
- **[Technical Design](docs/TECHNICAL_DESIGN.md)** - Complete technical specifications
- **[Physics Reference](docs/PHYSICS_REFERENCE.md)** - Physical parameters and equations

### Quick Reference

**Core Components**:
- **Renderer** (`core.py`): Main orchestration class, atmospheric effects
- **Intersections** (`intersections.py`): Ray-geometry intersections
- **Shadows** (`shadows.py`): Shadow square physics

**Key Design Decisions**:
1. Observer-centric coordinates (prevents floating-point precision loss)
2. Batch-first architecture (vectorized NumPy operations)
3. Delta-R solver (prevents catastrophic cancellation)
4. Modular physics (isolated for testing/validation)

## Getting Help

### Resources
- **[README.md](README.md)**: Project overview and quick start
- **[docs/](docs/)**: Detailed technical documentation
- **tests/**: Working examples of all major features

### Communication
- **Issues**: Bug reports, feature requests, physics questions
- **Discussions**: Architecture decisions, validation approaches
- **Pull requests**: Code review and implementation discussion

### Validation Questions
When implementing new features, ask:
- How do I prove this is physically correct?
- What existing tests might break?
- How does this affect performance?
- Is the visual result actually better?

---

## Quality Standards

**Visual Excellence**: "WOW factor" through accuracy, not gimmicks
**Scientific Rigor**: Every pixel backed by physics
**Performance**: Real-time interaction without compromises
**Maintainability**: Code that humans can understand and modify

Thank you for helping make Ringworld visualization more scientifically accurate and visually stunning! 🌌
