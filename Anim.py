import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# --- SHOWCASE CONFIGURATION ---
RESOLUTION = 800  # Crisp high-res grid
MAX_ITER = 150  # Deep iterations for fine, lightning-like detail
FRAMES = 400
ZOOM = 1.4  # Zoomed in closer to the action

# Create the base complex plane grid
Y, X = np.ogrid[-ZOOM:ZOOM:RESOLUTION * 1j, -ZOOM:ZOOM:RESOLUTION * 1j]
Z_BASE = X + Y * 1j

# A custom path for C that slices through highly chaotic fractal regions
t = np.linspace(0, 2 * np.pi, FRAMES, endpoint=False)
# This specific equation traces a wobbling path through "dendrite" and "spiral" zones
C_VALUES = 0.7885 * np.exp(1j * t) * (0.8 + 0.2 * np.cos(2 * t)) + 0.1j

fig, ax = plt.subplots(figsize=(9, 9), facecolor='black')
fig.canvas.manager.set_window_title("Julia Set Showcase")
ax.axis('off')  # Strip away the graph UI for a clean edge-to-edge render

# 'magma' colormap gives a fiery, glowing electric look against a dark background
img = ax.imshow(np.zeros((RESOLUTION, RESOLUTION)), cmap="magma",
                extent=(-ZOOM, ZOOM, -ZOOM, ZOOM), origin="lower",
                vmin=0, vmax=MAX_ITER)


def render_frame(c):
    z = Z_BASE.copy()
    iterations = np.zeros(z.shape, dtype=np.float32)
    active = np.ones(z.shape, dtype=bool)

    for i in range(MAX_ITER):
        z_active = z[active]
        z_active = z_active ** 2 + c
        z[active] = z_active

        # Check if the pixel has escaped the boundary
        escaped = (z_active.real ** 2 + z_active.imag ** 2) > 4

        just_escaped = np.zeros(z.shape, dtype=bool)
        just_escaped[active] = escaped

        iterations[just_escaped] = i
        active[just_escaped] = False

        if not np.any(active):
            break

    # Pixels that never escaped are part of the solid fractal core
    iterations[active] = MAX_ITER
    return iterations


def update(frame):
    c = C_VALUES[frame]
    img.set_data(render_frame(c))
    print(f"Rendering Frame {frame + 1}/{FRAMES} | c = {c.real:.3f}{c.imag:+.3f}i")
    return [img]


print("Launching high-resolution Julia showcase...")

# Tighten the layout so the image fills the entire window
fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
anim = FuncAnimation(fig, update, frames=FRAMES, interval=30, blit=True)

plt.show()