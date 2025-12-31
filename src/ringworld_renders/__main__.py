import numpy as np
from PIL import Image
from .engine import R, WIDTH, TAU_ZENITH

def main():
    # Setup Camera (100 degree FOV)
    WIDTH_PX, HEIGHT_PX = 1200, 600
    x_grid = np.linspace(-1, 1, WIDTH_PX)
    y_grid = np.linspace(-HEIGHT_PX/WIDTH_PX, HEIGHT_PX/WIDTH_PX, HEIGHT_PX)
    xx, yy = np.meshgrid(x_grid, y_grid)
    f = 1.0 / np.tan(np.radians(100.0) / 2.0)

    # Ray Directions (Forward=+X, Up=-Y, Right=+Z)
    dx = np.ones_like(xx) * f
    dy = yy
    dz = xx
    mag = np.sqrt(dx**2 + dy**2 + dz**2)
    dx /= mag; dy /= mag; dz /= mag

    # Intersection with Cylinder: t = -2 * R * dy / (dx^2 + dy^2)
    t_arch = np.where(dy < 0, -2.0 * R * dy / (dx**2 + dy**2), np.nan)

    # Check if ray hits within Ring Width (W)
    z_hit = t_arch * dz
    on_ring = np.abs(z_hit) < (WIDTH / 2.0)

    # Atmospheric Extinction (Analytical Leap)
    tau = TAU_ZENITH / np.abs(dy)
    ext = np.exp(-tau)

    # Colors
    sky = np.array([0.5, 0.7, 1.0])
    arch = np.array([0.4, 0.35, 0.3])
    ground = np.array([0.2, 0.25, 0.2])

    # Compositing
    img = np.zeros((HEIGHT_PX, WIDTH_PX, 3))
    for i in range(3):
        # Atmosphere/Sky base
        img[:,:,i] = sky[i] * (1.0 - np.exp(-TAU_ZENITH / np.maximum(np.abs(dy), 0.01)))
        # Add Arch
        arch_val = arch[i] * ext + sky[i] * (1.0 - ext)
        img[:,:,i] = np.where(on_ring & (dy < 0), arch_val, img[:,:,i])
        # Add Local Ground
        g_fade = 1.0 - np.exp(-0.05 / np.maximum(dy, 0.001))
        g_val = ground[i] * (1.0 - g_fade) + sky[i] * g_fade
        img[:,:,i] = np.where(dy >= 0, g_val, img[:,:,i])

    Image.fromarray((np.clip(img*1.2, 0, 1)*255).astype(np.uint8)).save("rart_mvp_arch.png")
    print("Render complete: rart_mvp_arch.png")

if __name__ == "__main__":
    main()
