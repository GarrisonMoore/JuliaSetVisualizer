import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, FFMpegWriter
from pathlib import Path

# --- 1. CONFIGURATION ---
RESOLUTION = 800
MAX_ITER = 100
FPS = 30
DURATION = 8
FRAMES = FPS * DURATION
C_RADIUS = 0.7
CMAP = "twilight"
ZOOM = 1.5

output_file = Path(__file__).with_name("JuliaMorph.mp4")

# --- 2. PRE-COMPUTATION ---
Y, X = np.ogrid[-ZOOM:ZOOM:RESOLUTION * 1j, -ZOOM:ZOOM:RESOLUTION * 1j]
Z_BASE = X + Y * 1j

angles = np.linspace(0, 2 * np.pi, FRAMES, endpoint=False)
C_VALUES = -0.54 + (0.08 * np.exp(1j * angles))

# --- 3. OPTIMIZED CALCULATION ---
def render_julia(c):
    z = Z_BASE.copy()
    iterations = np.zeros(z.shape, dtype=np.float32)
    active = np.ones(z.shape, dtype=bool)

    for i in range(MAX_ITER):
        z_active = z[active]
        z_active = z_active ** 3 + c
        z[active] = z_active
        escaped = (z_active.real ** 2 + z_active.imag ** 2) > 4

        just_escaped = np.zeros(z.shape, dtype=bool)
        just_escaped[active] = escaped

        iterations[just_escaped] = i
        active[just_escaped] = False

        if not np.any(active):
            break

    iterations[active] = MAX_ITER
    return iterations


# --- 4. STREAMLINED EXPORT / UI ---
fig, ax = plt.subplots(figsize=(8, 8))

# FIX 1: Re-added vmin and vmax so the colormap knows how to scale the data
img = ax.imshow(np.zeros((RESOLUTION, RESOLUTION)), cmap=CMAP,
                extent=(-ZOOM, ZOOM, -ZOOM, ZOOM), origin="lower",
                vmin=0, vmax=MAX_ITER)

title = ax.set_title("Julia Set Morph")
ax.set_xlabel("Real")
ax.set_ylabel("Imaginary")


def update(frame):
    c = C_VALUES[frame]
    img.set_data(render_julia(c))
    title.set_text(f"Frame {frame + 1}/{FRAMES} | c = {c.real:.3f}{c.imag:+.3f}i")
    return [img, title]


EXPORT_VIDEO = True

if EXPORT_VIDEO:
    print(f"Exporting {FRAMES} frames to {output_file.resolve()}...")
    writer = FFMpegWriter(fps=FPS, codec="libx264", bitrate=5000)
    anim = FuncAnimation(fig, update, frames=FRAMES)
    anim.save(str(output_file), writer=writer, dpi=100)
    print("Export complete.")
else:
    print("Running live preview. Close the window to exit.")
    # FIX 2: Set blit=False to prevent the TkAgg black-screen bug on Linux
    anim = FuncAnimation(fig, update, frames=FRAMES, interval=1000 / FPS, blit=False)
    plt.tight_layout()
    plt.show()