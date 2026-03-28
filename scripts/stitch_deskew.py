#!/usr/bin/env python3
"""Stitch two overlapping scans using anchor-point alignment.

1. Use SIFT anchors to compute affine transform from img2 to img1
2. Warp img2 into img1's coordinate space and blend
3. Detect paper tilt on the stitched result
4. Deskew the entire result once
5. Crop to paper content
"""

import sys
import os
import cv2
import numpy as np
from PIL import Image


def detect_paper_tilt(img):
    """Detect paper tilt angle by fitting lines to left/right paper edges."""
    h, w = img.shape[:2]
    b, g, r = cv2.split(img)
    diff = g.astype(np.int16) - b.astype(np.int16)
    paper_mask = diff > 15

    left_pts = []
    right_pts = []
    for frac in np.linspace(0.05, 0.95, 40):
        y = int(h * frac)
        row = paper_mask[y, :]
        nz = np.where(row)[0]
        if len(nz) > 50:
            left_pts.append((y, nz[0]))
            right_pts.append((y, nz[-1]))

    left_pts = np.array(left_pts, dtype=np.float64)
    right_pts = np.array(right_pts, dtype=np.float64)

    def robust_angle(pts):
        ys, xs = pts[:, 0], pts[:, 1]
        for _ in range(3):
            fit = np.polyfit(ys, xs, 1)
            resid = np.abs(xs - np.polyval(fit, ys))
            thresh = np.median(resid) + 2 * np.std(resid)
            mask = resid < max(thresh, 5)
            ys, xs = ys[mask], xs[mask]
        fit = np.polyfit(ys, xs, 1)
        return np.degrees(np.arctan(fit[0]))

    left_a = robust_angle(left_pts)
    right_a = robust_angle(right_pts)
    return (left_a + right_a) / 2


def deskew(img, angle):
    """Rotate image to correct tilt, expanding canvas, black border."""
    h, w = img.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    return cv2.warpAffine(img, M, (new_w, new_h),
                          flags=cv2.INTER_LANCZOS4,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(0, 0, 0)), M, (new_w, new_h)


# Parse arguments
args = sys.argv[1:]
flags = [a for a in args if a.startswith("--")]
files = [a for a in args if not a.startswith("--")]

file1 = files[0] if len(files) > 0 else "Scan 6.jpeg"
file2 = files[1] if len(files) > 1 else "Scan 7.jpeg"
output = files[2] if len(files) > 2 else "stitch_test/stitched.png"

# Load images
img1 = cv2.imread(file1)
img2 = cv2.imread(file2)
if "--rotate1" in flags:
    img1 = cv2.rotate(img1, cv2.ROTATE_180)
    print(f"Rotated {file1} 180°")
if "--rotate2" in flags:
    img2 = cv2.rotate(img2, cv2.ROTATE_180)
    print(f"Rotated {file2} 180°")

h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]
print(f"{file1}: {w1}x{h1}")
print(f"{file2}: {w2}x{h2}")

# Step 1: Find SIFT anchors
print("\n--- Finding anchor points (SIFT) ---")
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create(nfeatures=10000)
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)
print(f"Features: {len(kp1)} / {len(kp2)}")

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)
good = [m for m, n in matches if m.distance < 0.7 * n.distance]
print(f"Good matches: {len(good)}")

pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

# Step 2: Compute affine transform: img2 -> img1
M, inliers = cv2.estimateAffinePartial2D(pts2, pts1, method=cv2.RANSAC,
                                          ransacReprojThreshold=5.0)
n_inliers = int(inliers.sum())
angle = np.degrees(np.arctan2(M[1, 0], M[0, 0]))
scale = np.sqrt(M[0, 0]**2 + M[1, 0]**2)
tx, ty = M[0, 2], M[1, 2]
print(f"\n--- Affine alignment ---")
print(f"Inliers: {n_inliers} / {len(good)}")
print(f"Rotation: {angle:.4f}°")
print(f"Scale: {scale:.6f}")
print(f"Translation: tx={tx:.1f} ty={ty:.1f}")

# Step 3: Compute canvas
corners2 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
corners2_warped = cv2.transform(corners2, M).reshape(-1, 2)

all_corners = np.vstack([
    [[0, 0], [w1, 0], [w1, h1], [0, h1]],
    corners2_warped
])
x_min = int(np.floor(all_corners[:, 0].min()))
y_min = int(np.floor(all_corners[:, 1].min()))
x_max = int(np.ceil(all_corners[:, 0].max()))
y_max = int(np.ceil(all_corners[:, 1].max()))

canvas_w = x_max - x_min
canvas_h = y_max - y_min

M_adj = M.copy()
M_adj[0, 2] -= x_min
M_adj[1, 2] -= y_min
offset_x = -x_min
offset_y = -y_min

# Step 4: Place and warp
canvas1 = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
canvas1[offset_y:offset_y+h1, offset_x:offset_x+w1] = img1

canvas2 = cv2.warpAffine(img2, M_adj, (canvas_w, canvas_h),
                          flags=cv2.INTER_LANCZOS4,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(0, 0, 0))

mask1 = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
mask1[offset_y:offset_y+h1, offset_x:offset_x+w1] = 255

mask2 = cv2.warpAffine(np.ones((h2, w2), dtype=np.uint8) * 255,
                        M_adj, (canvas_w, canvas_h),
                        flags=cv2.INTER_NEAREST)

# Step 5: Blend overlap
overlap_mask = (mask1 > 0) & (mask2 > 0)
only2_mask = (mask1 == 0) & (mask2 > 0)

overlap_rows = np.any(overlap_mask, axis=1)
blend_top = np.argmax(overlap_rows)
blend_bottom = len(overlap_rows) - np.argmax(overlap_rows[::-1]) - 1
blend_height = blend_bottom - blend_top + 1
print(f"\nBlend zone: rows {blend_top}-{blend_bottom} ({blend_height}px)")

result = canvas1.copy()
result[only2_mask] = canvas2[only2_mask]

for y in range(blend_top, blend_bottom + 1):
    row_overlap = overlap_mask[y]
    if not row_overlap.any():
        continue
    alpha = (y - blend_top) / max(blend_height - 1, 1)
    cols = np.where(row_overlap)[0]
    r1 = canvas1[y, cols].astype(np.float32)
    r2 = canvas2[y, cols].astype(np.float32)
    result[y, cols] = ((1 - alpha) * r1 + alpha * r2).astype(np.uint8)

print(f"Stitched: {canvas_w}x{canvas_h}")

# Step 6: Deskew the entire stitched result
print("\n--- Deskewing final result ---")
tilt = detect_paper_tilt(result)
print(f"Paper tilt: {tilt:.3f}°")

if abs(tilt) > 0.02:
    result, _, _ = deskew(result, -tilt)  # negate: detected tilt is slope, need reverse rotation
    print(f"Deskewed: {result.shape[1]}x{result.shape[0]}")

# Step 7: Crop to paper (use G-B channel diff, avoids HSV issues with black border)
h_r, w_r = result.shape[:2]
b, g, r_ch = cv2.split(result)
diff = g.astype(np.int16) - b.astype(np.int16)
paper = diff > 15

# Also ensure pixel isn't black (from deskew border)
brightness = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
paper = paper & (brightness > 100)

rows = np.any(paper, axis=1)
cols = np.any(paper, axis=0)

if rows.any() and cols.any():
    y_start = np.argmax(rows)
    y_end = len(rows) - np.argmax(rows[::-1])
    x_start = np.argmax(cols)
    x_end = len(cols) - np.argmax(cols[::-1])

    pad = 5
    y_start = max(0, y_start - pad)
    y_end = min(h_r, y_end + pad)
    x_start = max(0, x_start - pad)
    x_end = min(w_r, x_end + pad)

    result = result[y_start:y_end, x_start:x_end]

print(f"Final: {result.shape[1]}x{result.shape[0]}")

# Step 8: Verify edge straightness
h_f, w_f = result.shape[:2]
b, g, r_ch = cv2.split(result)
diff = g.astype(np.int16) - b.astype(np.int16)
paper = (diff > 15) & (cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) > 100)
edges_l, edges_r = [], []
for frac in np.linspace(0.1, 0.9, 9):
    y = int(h_f * frac)
    nz = np.where(paper[y])[0]
    if len(nz) > 50:
        edges_l.append(nz[0])
        edges_r.append(nz[-1])
if edges_l:
    l_drift = max(edges_l) - min(edges_l)
    r_drift = max(edges_r) - min(edges_r)
    print(f"Edge straightness: left drift={l_drift}px, right drift={r_drift}px")

# Step 9: Save with PIL
result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
pil_img = Image.fromarray(result_rgb)

src_pil = Image.open(file1)
dpi = src_pil.info.get("dpi", (600, 600))
icc_profile = src_pil.info.get("icc_profile")
if not icc_profile:
    from PIL import ImageCms
    srgb = ImageCms.createProfile("sRGB")
    icc_profile = ImageCms.ImageCmsProfile(srgb).tobytes()

os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
ext = os.path.splitext(output)[1].lower()
save_kw = {"dpi": dpi, "icc_profile": icc_profile}
if ext in (".jpg", ".jpeg"):
    save_kw["quality"] = 98
    save_kw["subsampling"] = 0

pil_img.save(output, **save_kw)
print(f"\nSaved {output} ({result.shape[1]}x{result.shape[0]}, {float(dpi[0]):.0f} DPI)")
