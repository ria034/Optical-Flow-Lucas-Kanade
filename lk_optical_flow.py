"""
Sparse optical flow tracking using the Lucas-Kanade method (cv2.calcOpticalFlowPyrLK).

What this actually does:
1. On the first frame, finds good corner features to track (Shi-Tomasi
   corners via cv2.goodFeaturesToTrack).
2. On every subsequent frame, uses pyramidal Lucas-Kanade to estimate where
   each tracked point moved to.
3. Draws each point's trajectory over time, and periodically re-detects
   features so you don't lose track of everything as points drift out of
   frame or fail to track.

Usage:
    python3 lk_optical_flow.py <video_path> [output_video_path]

    If you don't have a video handy, run make_test_video.py first to
    generate a synthetic one (moving shapes) to test against.
"""

import sys
import cv2
import numpy as np


def track_optical_flow(video_path, output_path="optical_flow_output.mp4",
                        max_corners=100, redetect_every=30):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: could not open video at {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Shi-Tomasi corner detection parameters
    feature_params = dict(maxCorners=max_corners, qualityLevel=0.3,
                           minDistance=7, blockSize=7)

    # Lucas-Kanade optical flow parameters
    lk_params = dict(winSize=(15, 15), maxLevel=2,
                      criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                                10, 0.03))

    ret, old_frame = cap.read()
    if not ret:
        print("ERROR: empty video")
        sys.exit(1)

    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    # Random colors for drawing each point's trail
    colors = np.random.randint(0, 255, (max_corners, 3))

    # Accumulates trail lines so trajectories persist across frames
    mask = np.zeros_like(old_frame)

    frame_idx = 0
    total_tracked_points = 0
    total_frames_with_valid_flow = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if p0 is not None and len(p0) > 0:
            p1, status, error = cv2.calcOpticalFlowPyrLK(
                old_gray, frame_gray, p0, None, **lk_params)

            if p1 is not None:
                good_new = p1[status == 1]
                good_old = p0[status == 1]
                total_tracked_points += len(good_new)
                total_frames_with_valid_flow += 1

                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    color = colors[i % len(colors)].tolist()
                    mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)),
                                     color, 2)
                    frame = cv2.circle(frame, (int(a), int(b)), 4, color, -1)

                p0 = good_new.reshape(-1, 1, 2)
            else:
                p0 = None

        img = cv2.add(frame, mask)
        writer.write(img)

        old_gray = frame_gray.copy()
        frame_idx += 1

        # Periodically re-detect features (tracked points thin out over time
        # as some fail to track / leave frame)
        if frame_idx % redetect_every == 0 or p0 is None or len(p0) < max_corners // 3:
            p0_new = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
            p0 = p0_new

    cap.release()
    writer.release()

    avg_points = total_tracked_points / max(total_frames_with_valid_flow, 1)
    print(f"Processed {frame_idx} frames.")
    print(f"Average tracked points per frame: {avg_points:.1f}")
    print(f"Output written to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 lk_optical_flow.py <video_path> [output_path]")
        sys.exit(1)
    video_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "optical_flow_output.mp4"
    track_optical_flow(video_path, output_path)
