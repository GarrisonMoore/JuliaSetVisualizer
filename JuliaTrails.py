import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# --- 1. CONFIGURATION ---
RESOLUTION = 480
PARTICLES = 50000  # Number of points tracking the boundary
TRAIL_DECAY = 0.7  # How fast the trails fade (0.0 to 1.0)
FPS = 60
DURATION = 8
FRAMES = FPS * DURATION
C_RADIUS = 0.7885
CMAP = "Blues_r"  # Blues reversed looks close to the xscreensaver
ZOOM = 1.8

# --- 2. PRE-COMPUTATION ---
angles = np.linspace(0, 2 * np.pi, FRAMES, endpoint=False)
# Using a slightly off-center path creates more chaotic morphs
C_VALUES = C_RADIUS * np.exp(1j * angles) - 0.1

# Initialize random particles and the image buffer
points = np.random.uniform(-ZOOM, ZOOM, PARTICLES) + 1j * np.random.uniform(-ZOOM, ZOOM, PARTICLES)
image_buffer = np.zeros((RESOLUTION, RESOLUTION), dtype=np.float32)


# --- 3. INVERSE ITERATION LOGIC ---
def step_particles(c):
    global points, image_buffer

    # 1. Backward iteration: Z = sqrt(Z - C)
    points = np.sqrt(points - c)

    # 2. A complex square root has two answers (positive and negative).
    # We must randomly choose which path each particle takes to cover the whole shape.
    signs = np.random.choice([-1, 1], size=PARTICLES)
    points *= signs

    # 3. Fade the old frame to create trails
    image_buffer *= TRAIL_DECAY

    # 4. Map complex coordinates to pixel grid indices
    x_idx = np.clip(((points.real + ZOOM) / (2 * ZOOM)) * RESOLUTION, 0, RESOLUTION - 1).astype(int)
    y_idx = np.clip(((points.imag + ZOOM) / (2 * ZOOM)) * RESOLUTION, 0, RESOLUTION - 1).astype(int)

    # 5. Add new particle positions to the buffer
    np.add.at(image_buffer, (y_idx, x_idx), 1.5)

    return image_buffer


# --- 4. STREAMLINED UI ---
fig, ax = plt.subplots(figsize=(8, 8))
fig.patch.set_facecolor('black')
ax.set_facecolor('black')

# The fractal image
img = ax.imshow(image_buffer, cmap=CMAP, extent=(-ZOOM, ZOOM, -ZOOM, ZOOM),
                origin="lower", vmin=0, vmax=10, animated=True)

# The white control point 'C'
c_marker, = ax.plot([], [], 'wo', markersize=6, fillstyle='none')

# Places a monospace counter in the top-left corner of the render window
counter_text = ax.text(0.03, 0.97, "", transform=ax.transAxes,
                       color="white", fontsize=11, family="monospace", va="top")
ax.set_ylabel("Imaginary", color='white')
ax.tick_params(colors='white')


def update(frame):
    c = C_VALUES[frame]

    img.set_data(step_particles(c))
    c_marker.set_data([c.real], [c.imag])

    status_text = f"Frame {frame + 1}/{FRAMES} | c = {c.real:.3f}{c.imag:+.3f}i"

    # Update the on-screen text and print to console
    counter_text.set_text(status_text)
    print(status_text)

    # Return the new text object for the blitting engine
    return [img, c_marker, counter_text]


print("Running live preview. Close the window to exit.")
anim = FuncAnimation(fig, update, frames=FRAMES, interval=1000 / FPS, blit=True)
plt.tight_layout()
plt.show()