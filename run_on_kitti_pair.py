"""
Runs Lucas-Kanade optical flow on a real consecutive KITTI frame pair
(not a video -- just two images), and saves a visualization showing
tracked feature displacements.
"""
import sys
import cv2
import numpy as np

img1_path, img2_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]

img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

feature_params = dict(maxCorners=200, qualityLevel=0.2, minDistance=7, blockSize=7)
lk_params = dict(winSize=(21, 21), maxLevel=3,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

p0 = cv2.goodFeaturesToTrack(gray1, mask=None, **feature_params)
print(f"Detected {len(p0)} features in frame 10")

p1, status, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, p0, None, **lk_params)

good_new = p1[status == 1]
good_old = p0[status == 1]
print(f"Successfully tracked {len(good_new)} / {len(p0)} features into frame 11")

displacements = good_new - good_old
dists = np.linalg.norm(displacements, axis=1)
print(f"Displacement magnitude (pixels): mean={dists.mean():.2f}, "
      f"median={np.median(dists):.2f}, max={dists.max():.2f}")

# Visualize: draw each tracked point's motion vector over frame 11
vis = img2.copy()
for new, old in zip(good_new, good_old):
    a, b = new.ravel()
    c, d = old.ravel()
    cv2.arrowedLine(vis, (int(c), int(d)), (int(a), int(b)), (0, 255, 0), 1, tipLength=0.3)
    cv2.circle(vis, (int(a), int(b)), 3, (0, 0, 255), -1)

cv2.imwrite(out_path, vis)
print(f"Saved visualization to {out_path}")
