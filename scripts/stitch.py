#!/usr/bin/env python3
"""Stitch two overlapping scans with automatic tilt correction.

Workflow:
  1. Detect SIFT features in both images
  2. Match features → estimate affine transform (handles rotation + translation)
  3. Warp img2 into img1's coordinate space
  4. Laplacian pyramid blend in the overlap zone — smooths exposure/vignetting
     differences while preserving all sharp detail
  5. Save as lossless PNG preserving DPI and ICC profile
"""

import argparse
import os
import sys
import cv2
import numpy as np
from PIL import Image


def find_transform(img1, img2, max_features=10000, ratio_thresh=0.7):
    """Find affine transform from img2 → img1 using SIFT + RANSAC."""
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create(nfeatures=max_features)
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    print(f"  Features: {len(kp1)} / {len(kp2)}")

    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        sys.exit("Not enough features found.")

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < ratio_thresh * n.distance]
    print(f"  Good matches: {len(good)}")

    if len(good) < 10:
        sys.exit("Not enough good matches for alignment.")

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

    M, inliers = cv2.estimateAffinePartial2D(
        pts2, pts1, method=cv2.RANSAC, ransacReprojThreshold=5.0
    )
    n_inliers = int(inliers.sum())
    angle = np.degrees(np.arctan2(M[1, 0], M[0, 0]))
    scale = np.sqrt(M[0, 0] ** 2 + M[1, 0] ** 2)
    print(f"  Inliers: {n_inliers}/{len(good)}")
    print(f"  Rotation: {angle:.4f}°  Scale: {scale:.6f}")
    print(f"  Translation: tx={M[0,2]:.1f} ty={M[1,2]:.1f}")

    return M


def compute_canvas(img1, img2, M):
    """Compute canvas size and adjusted transform for stitching."""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    corners2 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
    corners2_warped = cv2.transform(corners2, M).reshape(-1, 2)

    all_corners = np.vstack([
        [[0, 0], [w1, 0], [w1, h1], [0, h1]],
        corners2_warped,
    ])
    x_min = int(np.floor(all_corners[:, 0].min()))
    y_min = int(np.floor(all_corners[:, 1].min()))
    x_max = int(np.ceil(all_corners[:, 0].max()))
    y_max = int(np.ceil(all_corners[:, 1].max()))

    canvas_w = x_max - x_min
    canvas_h = y_max - y_min
    offset_x = -x_min
    offset_y = -y_min

    M_adj = M.copy()
    M_adj[0, 2] += offset_x
    M_adj[1, 2] += offset_y

    return canvas_w, canvas_h, offset_x, offset_y, M_adj


def row_color_match(canvas1, canvas2, mask1, mask2):
    """Adjust canvas2 to match canvas1 using per-row color correction.

    For each row in the overlap, computes the per-channel mean ratio between
    the two images.  Smooths this into a correction curve and extends it
    beyond the overlap with edge clamping.  This corrects scanner vignetting
    (which varies along the scan/row direction) without introducing column
    artifacts.
    """
    overlap = (mask1 > 0) & (mask2 > 0)
    if not overlap.any():
        return canvas2

    h, w = canvas1.shape[:2]
    rows_any = np.any(overlap, axis=1)
    row_start = np.argmax(rows_any)
    row_end = len(rows_any) - np.argmax(rows_any[::-1])

    # Only use the middle portion of the overlap (skip unreliable edge rows
    # where only a few pixels overlap due to rotation)
    ov_len = row_end - row_start
    margin = max(1, ov_len // 8)
    reliable_start = row_start + margin
    reliable_end = row_end - margin

    # Per-row, per-channel gain (only from reliable rows)
    gain_per_row = np.ones((row_end - row_start, 3), dtype=np.float64)
    for y in range(reliable_start, reliable_end):
        row_mask = overlap[y]
        if row_mask.sum() < 100:
            continue
        c1 = canvas1[y, row_mask].astype(np.float64)
        c2 = canvas2[y, row_mask].astype(np.float64)
        for ch in range(3):
            m1 = c1[:, ch].mean()
            m2 = c2[:, ch].mean()
            if m2 > 5:
                gain_per_row[y - row_start, ch] = np.clip(m1 / m2, 0.85, 1.15)

    # Smooth the gain curve heavily (scanner vignetting is very smooth)
    smooth_size = max(51, (row_end - row_start) // 4 * 2 + 1)
    kernel = np.ones(smooth_size) / smooth_size
    gain_smooth = np.ones_like(gain_per_row)
    for ch in range(3):
        padded = np.pad(gain_per_row[:, ch], smooth_size // 2, mode="edge")
        gain_smooth[:, ch] = np.convolve(padded, kernel, mode="valid")[
            : len(gain_per_row)
        ]

    # Build full-height gain array with edge clamping
    gain_full = np.ones((h, 3), dtype=np.float64)
    gain_full[row_start:row_end] = gain_smooth
    gain_full[:row_start] = gain_smooth[0]
    gain_full[row_end:] = gain_smooth[-1]

    # Apply correction to canvas2
    corrected = canvas2.astype(np.float64)
    has_data = mask2 > 0
    for ch in range(3):
        for y in range(h):
            if not has_data[y].any():
                continue
            cols = has_data[y]
            corrected[y, cols, ch] *= gain_full[y, ch]

    for ch, name in enumerate(["B", "G", "R"]):
        g = gain_smooth[:, ch]
        print(f"    {name}: gain {g.min():.4f}-{g.max():.4f}")

    return np.clip(corrected, 0, 255).astype(np.uint8)


def blend_images(canvas1, canvas2, mask1, mask2):
    """Distance-weighted alpha blend, constrained to the overlap zone.

    Non-overlap pixels are taken directly from their source image (no
    modification).  In the overlap, uses a smoothly varying weight based on
    distance from each exclusive region.
    """
    overlap = (mask1 > 0) & (mask2 > 0)
    only1 = (mask1 > 0) & (mask2 == 0)
    only2 = (mask1 == 0) & (mask2 > 0)

    result = np.zeros_like(canvas1)
    result[only1] = canvas1[only1]
    result[only2] = canvas2[only2]

    if not overlap.any():
        return result

    h, w = canvas1.shape[:2]

    # Distance from each image's exclusive (non-overlap) region.
    # seed pixels mark the "home" zone for each image — the weight ramps
    # from 0 at the other image's home zone to 1 at this one's.
    seed1 = np.zeros((h, w), dtype=np.uint8)
    seed1[only1] = 255
    dist1 = cv2.distanceTransform(255 - seed1, cv2.DIST_L2, 5)

    seed2 = np.zeros((h, w), dtype=np.uint8)
    seed2[only2] = 255
    dist2 = cv2.distanceTransform(255 - seed2, cv2.DIST_L2, 5)

    # Weight for img1: higher when farther from img2-only (closer to img1-only)
    total = dist1 + dist2
    total[total == 0] = 1
    w1 = dist2[overlap] / total[overlap]
    # Smooth the weights with a sigmoid-like curve for gentler transition
    w1 = np.clip(w1, 0, 1)

    result[overlap] = (canvas1[overlap].astype(np.float32) * w1[:, np.newaxis] +
                        canvas2[overlap].astype(np.float32) * (1.0 - w1[:, np.newaxis])
                        ).astype(np.uint8)

    return result


def autocrop(img, border_thresh=10, pad=0):
    """Crop away black borders (from warping). Returns cropped image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = gray > border_thresh
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return img
    y0, y1 = np.argmax(rows), len(rows) - np.argmax(rows[::-1])
    x0, x1 = np.argmax(cols), len(cols) - np.argmax(cols[::-1])
    y0 = max(0, y0 - pad)
    x0 = max(0, x0 - pad)
    y1 = min(img.shape[0], y1 + pad)
    x1 = min(img.shape[1], x1 + pad)
    return img[y0:y1, x0:x1]


def stitch(img1, img2, rotate1=False, rotate2=False):
    """Stitch two overlapping scan images. Returns the stitched result."""
    if rotate1:
        img1 = cv2.rotate(img1, cv2.ROTATE_180)
    if rotate2:
        img2 = cv2.rotate(img2, cv2.ROTATE_180)

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    print(f"  Image 1: {w1}x{h1}")
    print(f"  Image 2: {w2}x{h2}")

    # Find alignment transform
    print("\nStep 1: Finding anchor points...")
    M = find_transform(img1, img2)

    # Compute canvas
    print("\nStep 2: Computing canvas...")
    canvas_w, canvas_h, off_x, off_y, M_adj = compute_canvas(img1, img2, M)
    print(f"  Canvas: {canvas_w}x{canvas_h}")

    # Place img1 and warp img2
    print("\nStep 3: Warping...")
    canvas1 = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas1[off_y:off_y + h1, off_x:off_x + w1] = img1

    canvas2 = cv2.warpAffine(
        img2, M_adj, (canvas_w, canvas_h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    mask1 = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    mask1[off_y:off_y + h1, off_x:off_x + w1] = 255

    mask2 = cv2.warpAffine(
        np.ones((h2, w2), dtype=np.uint8) * 255,
        M_adj, (canvas_w, canvas_h),
        flags=cv2.INTER_NEAREST,
    )

    # Row-based color matching — correct scanner vignetting via overlap
    print("\nStep 4: Color matching...")
    canvas2 = row_color_match(canvas1, canvas2, mask1, mask2)

    # Multi-band blend (Laplacian pyramid) — smooth remaining gradients
    print("\nStep 5: Blending...")
    result = blend_images(canvas1, canvas2, mask1, mask2)
    print(f"  Stitched: {canvas_w}x{canvas_h}")

    # Autocrop black borders
    print("\nStep 6: Cropping...")
    result = autocrop(result)
    print(f"  Final: {result.shape[1]}x{result.shape[0]}")

    return result


def save_lossless(result, output_path, reference_path):
    """Save as lossless PNG, preserving DPI and ICC from the reference image."""
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(result_rgb)

    src_pil = Image.open(reference_path)
    dpi = src_pil.info.get("dpi", (600, 600))
    icc_profile = src_pil.info.get("icc_profile")
    if not icc_profile:
        from PIL import ImageCms
        srgb = ImageCms.createProfile("sRGB")
        icc_profile = ImageCms.ImageCmsProfile(srgb).tobytes()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    save_kw = {"dpi": dpi, "icc_profile": icc_profile}
    ext = os.path.splitext(output_path)[1].lower()
    if ext == ".png":
        save_kw["compress_level"] = 6  # lossless compression
    elif ext == ".tiff" or ext == ".tif":
        save_kw["compression"] = "tiff_lzw"  # lossless
    elif ext in (".jpg", ".jpeg"):
        print("  Warning: JPEG is lossy. Use .png or .tiff for lossless output.")
        save_kw["quality"] = 100
        save_kw["subsampling"] = 0

    pil_img.save(output_path, **save_kw)
    print(f"\nSaved: {output_path} ({result.shape[1]}x{result.shape[0]}, {float(dpi[0]):.0f} DPI)")


def main():
    parser = argparse.ArgumentParser(
        description="Stitch two overlapping scans with automatic tilt correction."
    )
    parser.add_argument("image1", help="First scan (left/top)")
    parser.add_argument("image2", help="Second scan (right/bottom, overlapping)")
    parser.add_argument("-o", "--output", default="stitched.png",
                        help="Output path (default: stitched.png)")
    parser.add_argument("--rotate1", action="store_true",
                        help="Rotate image 1 by 180°")
    parser.add_argument("--rotate2", action="store_true",
                        help="Rotate image 2 by 180°")
    args = parser.parse_args()

    img1 = cv2.imread(args.image1)
    img2 = cv2.imread(args.image2)
    if img1 is None:
        sys.exit(f"Cannot read: {args.image1}")
    if img2 is None:
        sys.exit(f"Cannot read: {args.image2}")

    print(f"Stitching: {args.image1} + {args.image2}")
    result = stitch(img1, img2, rotate1=args.rotate1, rotate2=args.rotate2)
    save_lossless(result, args.output, args.image1)


if __name__ == "__main__":
    main()
