# optical_flow

Sparse optical flow tracking using pyramidal Lucas-Kanade (OpenCV).

## What it does
- Detects Shi-Tomasi corner features on the first frame
  (`cv2.goodFeaturesToTrack`)
- Tracks them frame-to-frame using `cv2.calcOpticalFlowPyrLK`
- Periodically re-detects features as points drift out of frame or fail
  to track
- Draws trajectory trails over the video for visualization

## Files
- `lk_optical_flow.py` — the tracker. Usage:
  `python3 lk_optical_flow.py <video_path> [output_path]`
- `make_test_video.py` — generates a synthetic test video (moving
  checkerboard shapes) for testing without needing a real camera recording.
  Usage: `python3 make_test_video.py` (writes `test_input.mp4`)
- `tests/sample_frame.png` — a sample output frame from a test run, showing
  tracked trajectories over the synthetic test video

## Status
Tested and working on synthetic footage (90-frame test sequence, ~99
tracked points/frame average, visually verified correct trajectories).

**Not yet done:**
- Not tested on real camera footage
- Not integrated with `standalone_pipeline/`'s LiDAR-based trajectory data
  (a natural next step: compare optical-flow-estimated motion against the
  existing LiDAR/IMU-based trajectory on NCLT's camera frames, if using
  NCLT data, as a cross-validation check)

## Dependencies
See `requirements.txt`.
