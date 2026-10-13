"""
Generates 3 more synthetic test videos, each stressing the tracker in a
different way than the original moving-shapes video:

1. pan_test.mp4      - a large textured background panning uniformly,
                        like a camera panning across a static scene.
2. rotation_test.mp4 - a single textured square rotating in place (tests
                        how LK handles rotation, which pure translation-
                        based tracking historically struggles with more).
3. occlusion_test.mp4 - two shapes crossing paths and occluding each other
                        (tests whether tracking recovers after a feature
                        gets temporarily covered).
"""

import cv2
import numpy as np

WIDTH, HEIGHT = 640, 480
NUM_FRAMES = 90
FPS = 30


def checkerboard(size, cell=12):
    board = np.zeros((size, size, 3), dtype=np.uint8)
    for cy in range(0, size, cell):
        for cx in range(0, size, cell):
            color = (220, 220, 220) if (cx // cell + cy // cell) % 2 == 0 else (60, 60, 60)
            cv2.rectangle(board, (cx, cy), (min(cx + cell, size), min(cy + cell, size)),
                          color, -1)
    return board


# ---------------------------------------------------------------------
# 1. Panning camera over a large static textured background
# ---------------------------------------------------------------------
def make_pan_test():
    bg_size = 1200
    bg = checkerboard(bg_size, cell=20)
    # add some larger random blobs for texture variety (pure checkerboard
    # can alias badly with regular corner detection)
    rng = np.random.RandomState(2)
    for _ in range(40):
        x, y = rng.randint(0, bg_size, size=2)
        r = rng.randint(10, 30)
        color = tuple(int(c) for c in rng.randint(0, 255, size=3))
        cv2.circle(bg, (x, y), r, color, -1)

    writer = cv2.VideoWriter("pan_test.mp4", cv2.VideoWriter_fourcc(*"mp4v"),
                              FPS, (WIDTH, HEIGHT))
    max_offset = bg_size - WIDTH
    for i in range(NUM_FRAMES):
        ox = int((i / NUM_FRAMES) * max_offset)
        oy = int(50 + 20 * np.sin(i / 10))
        oy = max(0, min(oy, bg_size - HEIGHT))
        frame = bg[oy:oy + HEIGHT, ox:ox + WIDTH]
        writer.write(frame)
    writer.release()
    print("Wrote pan_test.mp4")


# ---------------------------------------------------------------------
# 2. Rotating textured square
# ---------------------------------------------------------------------
def make_rotation_test():
    writer = cv2.VideoWriter("rotation_test.mp4", cv2.VideoWriter_fourcc(*"mp4v"),
                              FPS, (WIDTH, HEIGHT))
    square = checkerboard(160, cell=16)
    cx, cy = WIDTH // 2, HEIGHT // 2
    for i in range(NUM_FRAMES):
        frame = np.full((HEIGHT, WIDTH, 3), 30, dtype=np.uint8)
        angle = (i / NUM_FRAMES) * 180  # half rotation over the clip
        M = cv2.getRotationMatrix2D((80, 80), angle, 1.0)
        rotated = cv2.warpAffine(square, M, (160, 160),
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=(30, 30, 30))
        x0, y0 = cx - 80, cy - 80
        frame[y0:y0 + 160, x0:x0 + 160] = rotated
        writer.write(frame)
    writer.release()
    print("Wrote rotation_test.mp4")


# ---------------------------------------------------------------------
# 3. Two shapes crossing paths (occlusion)
# ---------------------------------------------------------------------
def make_occlusion_test():
    writer = cv2.VideoWriter("occlusion_test.mp4", cv2.VideoWriter_fourcc(*"mp4v"),
                              FPS, (WIDTH, HEIGHT))
    sq_a = checkerboard(90, cell=10)
    sq_b = checkerboard(90, cell=14)

    for i in range(NUM_FRAMES):
        frame = np.full((HEIGHT, WIDTH, 3), 30, dtype=np.uint8)
        t = i / NUM_FRAMES

        # A moves left-to-right, B moves right-to-left, crossing in the middle
        ax = int(-90 + t * (WIDTH))
        ay = HEIGHT // 2 - 45
        bx = int(WIDTH - t * (WIDTH + 90))
        by = HEIGHT // 2 - 45

        # draw B first, then A on top, so they visibly occlude each other
        # when overlapping
        for (x, y, sq) in [(bx, by, sq_b), (ax, ay, sq_a)]:
            x0, y0 = max(x, 0), max(y, 0)
            x1, y1 = min(x + 90, WIDTH), min(y + 90, HEIGHT)
            if x1 <= x0 or y1 <= y0:
                continue
            sq_x0, sq_y0 = x0 - x, y0 - y
            sq_x1, sq_y1 = sq_x0 + (x1 - x0), sq_y0 + (y1 - y0)
            frame[y0:y1, x0:x1] = sq[sq_y0:sq_y1, sq_x0:sq_x1]

        writer.write(frame)
    writer.release()
    print("Wrote occlusion_test.mp4")


if __name__ == "__main__":
    make_pan_test()
    make_rotation_test()
    make_occlusion_test()
