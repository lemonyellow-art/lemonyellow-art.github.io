# Image Scan Stitcher

Stitch two overlapping flatbed scans into a single seamless image. Automatically detects and corrects tilt between scans.

## Requirements

```
pip install opencv-python numpy pillow
```

## Usage

```bash
python3 stitch.py <image1> <image2> [-o output.png] [--rotate1] [--rotate2]
```

| Argument | Description |
|----------|-------------|
| `image1` | First scan |
| `image2` | Second scan (must overlap with image1) |
| `-o`, `--output` | Output path (default: `stitched.png`) |
| `--rotate1` | Rotate image1 by 180° before stitching |
| `--rotate2` | Rotate image2 by 180° before stitching |

## Examples

```bash
# Basic stitch
python3 stitch.py "Scan 1.jpeg" "Scan 2.jpeg" -o stitched.png

# One scan was placed upside down
python3 stitch.py "Scan 1.jpeg" "Scan 2.jpeg" --rotate2 -o stitched.png

# Output as TIFF (also lossless)
python3 stitch.py "Scan 1.jpeg" "Scan 2.jpeg" -o stitched.tiff
```

## How it works

1. **Anchor detection** -- SIFT features are detected in both images and matched to find corresponding points
2. **Alignment** -- RANSAC estimates an affine transform (rotation + translation + scale) from the matched points, correcting any tilt between scans
3. **Color matching** -- per-row brightness correction compensates for scanner vignetting (edges darker than center)
4. **Blending** -- distance-weighted alpha blend across the overlap zone creates a smooth, seamless transition
5. **Save** -- output as lossless PNG (or TIFF) preserving DPI and ICC color profile from the source

## Tips

- The two scans need sufficient overlapping content (at least a few hundred pixels) for feature matching to work. More overlap = better results.
- Order doesn't matter much -- the script auto-detects the spatial relationship. But if results look wrong, try swapping image1 and image2.
- If a scan was placed face-down in the opposite direction, use `--rotate1` or `--rotate2`.
- Use `.png` or `.tiff` output for lossless quality. JPEG output uses quality=100 but is still technically lossy.
