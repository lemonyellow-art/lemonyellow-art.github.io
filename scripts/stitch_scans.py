#!/usr/bin/env python3
"""Stitch two overlapping scans vertically using SIFT feature matching."""

import sys
import os
import cv2
import numpy as np
from PIL import Image

# Parse arguments
# Usage: script.py file1 file2 output [--rotate1] [--rotate2]
# --rotate1 rotates file1 180°, --rotate2 rotates file2 180°
args = sys.argv[1:]
flags = [a for a in args if a.startswith("--")]
files = [a for a in args if not a.startswith("--")]

file1 = files[0] if len(files) > 0 else "Scan 4.jpeg"
file2 = files[1] if len(files) > 1 else "Scan 5.jpeg"
output = files[2] if len(files) > 2 else "Scan_stitched.jpeg"

# Load images
img1 = cv2.imread(file1)  # top scan
img2 = cv2.imread(file2)  # bottom scan

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

# Detect and crop the paper region (remove scanner background)
def find_paper_bbox(img):
    """Find the bounding box of the yellow paper in the scan."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([15, 20, 180])
    upper = np.array([40, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    lower2 = np.array([10, 10, 220])
    upper2 = np.array([45, 255, 255])
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask, mask2)
    kernel = np.ones((20, 20), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=3)
    mask = cv2.erode(mask, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        pad = 50
        ih, iw = img.shape[:2]
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(iw - x, w + 2 * pad)
        h = min(ih - y, h + 2 * pad)
        return x, y, w, h
    return 0, 0, img.shape[1], img.shape[0]

# Step 1: Find vertical offset using SIFT on raw images
print("\nFinding overlap with SIFT...")
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create(nfeatures=10000)
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)
print(f"Features: img1={len(kp1)} img2={len(kp2)}")

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Lowe's ratio test
good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)
print(f"Good matches: {len(good)}")

# Compute y-offsets and find consensus
y_offsets = np.array([kp1[m.queryIdx].pt[1] - kp2[m.trainIdx].pt[1] for m in good])
x_offsets = np.array([kp1[m.queryIdx].pt[0] - kp2[m.trainIdx].pt[0] for m in good])

# Histogram to find peak
hist, edges = np.histogram(y_offsets, bins=200)
peak_idx = np.argmax(hist)
peak_center = (edges[peak_idx] + edges[peak_idx + 1]) / 2

# Refine with inliers near the peak
inlier_mask = np.abs(y_offsets - peak_center) < 100
y_offset_raw = int(np.median(y_offsets[inlier_mask]))
x_offset_raw = int(np.median(x_offsets[inlier_mask]))
print(f"SIFT offset: y={y_offset_raw} x={x_offset_raw} (inliers: {inlier_mask.sum()})")

# Step 2: Crop to paper and apply offset
bbox1 = find_paper_bbox(img1)
bbox2 = find_paper_bbox(img2)
print(f"Paper bbox1: {bbox1}")
print(f"Paper bbox2: {bbox2}")

x1, y1, pw1, ph1 = bbox1
x2, y2, pw2, ph2 = bbox2
crop1 = img1[y1:y1+ph1, x1:x1+pw1]
crop2 = img2[y2:y2+ph2, x2:x2+pw2]

# Resize crop2 width to match crop1
if crop1.shape[1] != crop2.shape[1]:
    crop2 = cv2.resize(crop2, (crop1.shape[1], crop2.shape[0]))

ch1, cw1 = crop1.shape[:2]
ch2, cw2 = crop2.shape[:2]
print(f"Cropped 1: {cw1}x{ch1}")
print(f"Cropped 2: {cw2}x{ch2}")

# Translate raw offset to cropped coordinates
# crop1 starts at y1, crop2 starts at y2
# In raw: img2_y = img1_y - y_offset_raw
# In crop: crop2_y = (img1_y - y1 + y_offset_raw_adjusted)
# y_offset in crop coords = y_offset_raw - y1 + y2
y_offset = y_offset_raw - y1 + y2
overlap = ch1 - y_offset
print(f"Vertical offset (crop coords): {y_offset}")
print(f"Overlap: {overlap}px ({overlap*100/ch1:.1f}% of crop height)")

if overlap <= 0 or overlap >= ch1 or overlap >= ch2:
    print(f"WARNING: overlap={overlap} seems wrong, trying template matching fallback...")
    # Fallback to template matching
    g1 = cv2.cvtColor(crop1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(crop2, cv2.COLOR_BGR2GRAY)
    margin = 100
    strip = g2[200:500, margin:cw2-margin]
    search = g1[ch1//2:, margin:cw1-margin]
    result = cv2.matchTemplate(search, strip, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    y_offset = ch1 // 2 + max_loc[1] - 200
    overlap = ch1 - y_offset
    print(f"Template fallback: score={max_val:.4f} overlap={overlap}")

# Stitch: crop1 top + blended overlap + crop2 bottom
total_height = y_offset + ch2
result_img = np.zeros((total_height, cw1, 3), dtype=np.uint8)

# Place crop1's non-overlapping top
result_img[0:y_offset] = crop1[0:y_offset]

# Blend overlap region
for row in range(overlap):
    alpha = row / overlap
    result_img[y_offset + row] = ((1 - alpha) * crop1[y_offset + row].astype(np.float32) +
                                   alpha * crop2[row].astype(np.float32)).astype(np.uint8)

# Place crop2's non-overlapping bottom
result_img[ch1:] = crop2[overlap:]

# Save using PIL to preserve DPI and color profile from the original scan
# Convert BGR (OpenCV) to RGB (PIL)
result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
pil_img = Image.fromarray(result_rgb)

# Read DPI and ICC profile from the first source image
src_pil = Image.open(file1)
dpi = src_pil.info.get("dpi", (600, 600))
icc_profile = src_pil.info.get("icc_profile")

# If no embedded ICC profile, create an explicit sRGB profile
if not icc_profile:
    from PIL import ImageCms
    srgb = ImageCms.createProfile("sRGB")
    icc_profile = ImageCms.ImageCmsProfile(srgb).tobytes()

ext = os.path.splitext(output)[1].lower()
save_kwargs = {}
if ext in (".jpg", ".jpeg"):
    save_kwargs["quality"] = 98
    save_kwargs["subsampling"] = 0  # 4:4:4 — no chroma subsampling
if dpi:
    save_kwargs["dpi"] = dpi
if icc_profile:
    save_kwargs["icc_profile"] = icc_profile

pil_img.save(output, **save_kwargs)
print(f"\nSaved {output} ({result_img.shape[1]}x{result_img.shape[0]}, {float(dpi[0]):.0f} DPI)")
