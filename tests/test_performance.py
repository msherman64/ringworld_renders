import time
import numpy as np
from ringworld_renders.core import Renderer

def test_render_caching():
    """
    Verify that subsequent renders with identical parameters are significantly faster.
    """
    renderer = Renderer()
    look_at = np.array([1.0, 0.3, 0.0])
    
    # First render (uncached)
    start_time = time.time()
    renderer.render(width=100, height=100, look_at=look_at)
    first_duration = time.time() - start_time
    
    # Second render (should be cached)
    start_time = time.time()
    renderer.render(width=100, height=100, look_at=look_at)
    second_duration = time.time() - start_time
    
    print(f"\nCache performance: First={first_duration:.4f}s, Second={second_duration:.4f}s")
    
    # cached should be at least 10x faster (actually should be near 0s)
    assert second_duration < first_duration / 10.0, f"Caching failed: {second_duration} is not significantly faster than {first_duration}"
