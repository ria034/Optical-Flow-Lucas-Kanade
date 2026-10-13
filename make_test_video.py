"""
Generates a synthetic test video: a few shapes moving across frame with
distinct textures/corners, so Shi-Tomasi corner detection + Lucas-Kanade
optical flow has something real to track. Useful for testing the pipeline
without needing a real camera recording on hand yet.
"""

import cv2
import numpy as np

WIDTH, HEIGHT = 640, 480
NUM_FRAMES = 90
OUTPUT = "test_input.mp4"

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUTPUT, fourcc, 30, (WIDTH, HEIGHT))

rng = np.random.RandomState(1)

# A handful of textured squares (checkerboard pattern gives good corners)
# moving at different velocities/directions.
shapes = []
for i in range(5):
    x = rng.randint(50, WIDTH - 150)
    y = rng.randint(50, HEIGHT - 150)
    vx = rng.uniform(-4, 4)
    vy = rng.uniform(-4, 4)
    size = rng.randint(60, 100)
    shapes.append({"x": x, "y": y, "vx": vx, "vy": vy, "size": size})

for frame_idx in range(NUM_FRAMES):
    frame = np.full((HEIGHT, WIDTH, 3), 30, dtype=np.uint8)  # dark background

    for s in shapes:
        s["x"] += s["vx"]
        s["y"] += s["vy"]
        # bounce off edges
        if s["x"] < 0 or s["x"] + s["size"] > WIDTH:
            s["vx"] *= -1
        if s["y"] < 0 or s["y"] + s["size"] > HEIGHT:
            s["vy"] *= -1

        x, y, size = int(s["x"]), int(s["y"]), s["size"]
        # Checkerboard pattern inside the square -- gives Shi-Tomasi corners
        cell = max(size // 6, 5)
        for cy in range(0, size, cell):
            for cx in range(0, size, cell):
                color = (220, 220, 220) if (cx // cell + cy // cell) % 2 == 0 else (60, 60, 60)
                cv2.rectangle(frame, (x + cx, y + cy),
                               (min(x + cx + cell, x + size), min(y + cy + cell, y + size)),
                               color, -1)

    writer.write(frame)

writer.release()
print(f"Wrote {NUM_FRAMES}-frame synthetic test video to {OUTPUT}")
