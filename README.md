# Optical Flow Tracking

Sparse optical flow tracking using pyramidal Lucas-Kanade (OpenCV), tested
on both real KITTI data and synthetic sequences.

## Real result: KITTI Stereo/Flow 2012 (scene 000170)

`run_on_kitti_pair.py` run on a real consecutive frame pair
(`000170_10.png` → `000170_11.png`):

- **160 / 160** Shi-Tomasi corner features successfully tracked
- Displacement magnitude: mean **2.2px**, median 1.77px, max 21.4px
- See `results/kitti_000170_flow.png` — tracked points land correctly on
  real scene structure (building window corners, car edges, container
  edges)

```bash
python3 run_on_kitti_pair.py <frame_10.png> <frame_11.png> <output.png>
```

**Honest scope note:** this validates that tracking works correctly and
lands on real, meaningful features — it is not yet a quantitative accuracy
result. KITTI's flow benchmark includes ground-truth optical flow for
training scenes (`flow_noc`/`flow_occ`); computing actual End-Point-Error
(EPE) against that ground truth would be the real next step to turn this
into a measured accuracy number rather than a visual sanity check.

## Synthetic validation suite

Before testing on real data, the tracker was validated against several
synthetic motion patterns (generated in `synthetic_tests/`), to stress-test
different motion types in a controlled way:

| Test | Script | Result |
|---|---|---|
| Moving shapes (translation) | `make_test_video.py` | ~99 pts/frame tracked |
| Camera panning | `make_more_test_videos.py` → `pan_test.mp4` | ~86 pts/frame tracked |
| Rotation | `make_more_test_videos.py` → `rotation_test.mp4` | ~98 pts/frame tracked, correct curved trajectories |
| Occlusion (crossing paths) | `make_more_test_videos.py` → `occlusion_test.mp4` | ~77 pts/frame tracked, recovered after crossing |

Sample output frames for each are in `results/`.

```bash
cd synthetic_tests
python3 make_test_video.py
python3 make_more_test_videos.py
python3 ../lk_optical_flow.py pan_test.mp4 pan_output.mp4
# etc.
```

## Files
- `lk_optical_flow.py` — video-based tracker (for synthetic test videos or
  any real video file)
- `run_on_kitti_pair.py` — two-image tracker, for KITTI-style consecutive
  frame pairs
- `synthetic_tests/` — synthetic test video generators
- `results/` — sample output visualizations, both real (KITTI) and synthetic

## Status
Working on both real (KITTI) and synthetic data. Quantitative accuracy
evaluation against KITTI's ground-truth flow is the clear next step, not
yet done.
