"""
Sensor check from test shots: dust/scratches and hot (dead) pixels.

Which shots to take (RAW .CR2 preferred, JPEG also works):
  * "Flat" frames (dust, scratches): clear sky or a white wall,
    Av mode, f/16–f/22, ISO 100, manual focus set to minimum (background blurred).
    Take 2–3 frames, moving the camera slightly between them.
  * "Dark" frames (hot pixels): lens cap on, M mode,
    ISO 100, exposures of 1 s and 30 s.

Installation (once):
    pip install numpy scipy pillow rawpy

Usage:
    python sensor_check.py FOLDER_WITH_SHOTS

The frame type (flat/dark) is detected automatically.
Output: marked-up images and a report.txt in the sensor_check_result subfolder.
"""

import os
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

RAW_EXT = {".cr2", ".cr3", ".dng", ".nef", ".arw", ".raf", ".orf", ".rw2"}
IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

DUST_THRESHOLD = 0.02   # spot is at least 2 % darker than the background
MIN_SPOT_AREA = 6       # pixels (after downscaling)
HOT_SIGMA = 10          # how much brighter than noise a pixel must be


# ---------- loading ----------

def load(path):
    """Returns (luminance 0..1 for dust analysis, raw data 0..1 for pixels)."""
    ext = os.path.splitext(path)[1].lower()
    if ext in RAW_EXT:
        try:
            import rawpy
        except ImportError:
            sys.exit("rawpy is required for RAW files:  pip install rawpy")
        with rawpy.imread(path) as raw:
            black = float(np.mean(raw.black_level_per_channel))
            white = float(raw.white_level)
            bayer = (raw.raw_image_visible.astype(np.float32) - black) / (white - black)
            rgb = raw.postprocess(half_size=True, gamma=(1, 1), no_auto_bright=True,
                                  output_bps=16, use_camera_wb=True)
        gray = rgb.astype(np.float32).mean(axis=2) / 65535.0
        return gray, np.clip(bayer, 0, None)
    img = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    return img, img


def is_dark(data):
    return float(np.median(data)) < 0.03


# ---------- hot pixels ----------

def check_hot_pixels(data):
    med = float(np.median(data))
    mad = float(np.median(np.abs(data - med))) * 1.4826 + 1e-6
    thr = max(med + HOT_SIGMA * mad, med + 0.02)
    ys, xs = np.nonzero(data > thr)
    vals = data[ys, xs]
    order = np.argsort(vals)[::-1]
    return ys[order], xs[order], vals[order], thr


def draw_hot(data, ys, xs, out_path, max_marks=300):
    h, w = data.shape
    scale = max(1, max(h, w) // 1600)
    # max pooling so single points don't vanish when downscaling
    hh, ww = h // scale, w // scale
    small = data[:hh * scale, :ww * scale].reshape(hh, scale, ww, scale).max(axis=(1, 3))
    vis = np.clip(small / max(np.percentile(small, 99.99), 1e-6), 0, 1) ** 0.5
    img = Image.fromarray((vis * 255).astype(np.uint8)).convert("RGB")
    d = ImageDraw.Draw(img)
    for y, x in zip(ys[:max_marks], xs[:max_marks]):
        cx, cy = x // scale, y // scale
        d.ellipse([cx - 8, cy - 8, cx + 8, cy + 8], outline=(255, 40, 40), width=2)
    img.save(out_path)


# ---------- dust and scratches ----------

def check_dust(gray):
    h, w = gray.shape
    bg = ndimage.gaussian_filter(gray, sigma=max(h, w) / 60)
    ratio = ndimage.gaussian_filter(gray, 1.5) / np.maximum(bg, 1e-4)

    mask = ratio < 1 - DUST_THRESHOLD
    m = int(0.03 * min(h, w))                       # skip frame edges (vignetting)
    mask[:m, :] = mask[-m:, :] = False
    mask[:, :m] = mask[:, -m:] = False

    labels, n = ndimage.label(mask)
    spots = []
    for i, sl in enumerate(ndimage.find_objects(labels), start=1):
        region = labels[sl] == i
        area = int(region.sum())
        if area < MIN_SPOT_AREA:
            continue
        yy, xx = np.nonzero(region)
        yy = yy + sl[0].start
        xx = xx + sl[1].start
        depth = 1 - float(ratio[yy, xx].min())
        # elongation along principal axes
        cov = np.cov(np.vstack([xx, yy])) if area > 2 else np.eye(2)
        ev = np.sort(np.linalg.eigvalsh(cov))
        elong = float(np.sqrt(ev[1] / max(ev[0], 1e-3)))
        length = float(4 * np.sqrt(ev[1]))
        kind = "line (possible scratch)" if elong > 4 and length > 25 else "spot (likely dust)"
        spots.append(dict(x=float(xx.mean()) / w, y=float(yy.mean()) / h, area=area,
                          depth=depth, kind=kind, bbox=(sl[1].start, sl[0].start,
                                                        sl[1].stop, sl[0].stop)))
    spots.sort(key=lambda s: -s["depth"])
    return spots, ratio


def draw_dust(ratio, spots, out_path):
    # contrast heavily boosted so dust is visible to the eye
    vis = np.clip((ratio - 0.9) / 0.15, 0, 1)
    img = Image.fromarray((vis * 255).astype(np.uint8)).convert("RGB")
    d = ImageDraw.Draw(img)
    for s in spots:
        x0, y0, x1, y1 = s["bbox"]
        pad = 12
        color = (255, 170, 0) if "line" in s["kind"] else (255, 40, 40)
        d.ellipse([x0 - pad, y0 - pad, x1 + pad, y1 + pad], outline=color, width=3)
    if max(img.size) > 2000:
        img.thumbnail((2000, 2000))
    img.save(out_path)


# ---------- main ----------

def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python sensor_check.py FOLDER_WITH_SHOTS")
    folder = sys.argv[1]
    files = sorted(f for f in os.listdir(folder)
                   if os.path.splitext(f)[1].lower() in RAW_EXT | IMG_EXT)
    if not files:
        sys.exit("No shots (RAW or JPEG) found in the folder.")

    out_dir = os.path.join(folder, "sensor_check_result")
    os.makedirs(out_dir, exist_ok=True)
    report = []
    flat_spots = []      # (file name, list of spots)

    def say(line=""):
        print(line)
        report.append(line)

    for name in files:
        path = os.path.join(folder, name)
        base = os.path.splitext(name)[0]
        say(f"=== {name}")
        try:
            gray, data = load(path)
        except Exception as e:                      # noqa: BLE001
            say(f"  failed to open: {e}")
            continue

        if is_dark(data):
            ys, xs, vals, thr = check_hot_pixels(data)
            n = len(ys)
            if n == 0:
                verdict = "excellent"
            elif n <= 30:
                verdict = "normal"
            elif n <= 200:
                verdict = "a bit many — try \"Manual sensor cleaning\" (pixel remapping)"
            else:
                verdict = "many — cause for concern"
            say(f"  dark frame: {n} hot/dead pixels → {verdict}")
            for y, x, v in list(zip(ys, xs, vals))[:10]:
                say(f"    x={x:5d}  y={y:5d}  brightness {v:.3f}")
            draw_hot(data, ys, xs, os.path.join(out_dir, base + "_hot.png"))
        else:
            spots, ratio = check_dust(gray)
            lines = [s for s in spots if "line" in s["kind"]]
            say(f"  flat frame: {len(spots) - len(lines)} spots, {len(lines)} lines")
            for s in spots[:15]:
                say(f"    {s['kind']:28s} x={s['x']*100:5.1f}%  y={s['y']*100:5.1f}%  "
                    f"{s['depth']*100:4.1f}% darker than background")
            if len(spots) > 15:
                say(f"    ... and {len(spots) - 15} more")
            draw_dust(ratio, spots, os.path.join(out_dir, base + "_dust.png"))
            flat_spots.append((name, spots))
        say()

    # Whatever repeats across several flat frames sits on the sensor
    if len(flat_spots) >= 2:
        say("=== Flat frame comparison")
        first_name, first = flat_spots[0]
        repeated = []
        for s in first:
            hits = 1
            for _, other in flat_spots[1:]:
                if any(abs(s["x"] - o["x"]) < 0.01 and abs(s["y"] - o["y"]) < 0.01 for o in other):
                    hits += 1
            if hits >= 2:
                repeated.append((s, hits))
        if repeated:
            say(f"  Same location on several frames: {len(repeated)} — these are on the sensor:")
            for s, hits in repeated:
                say(f"    {s['kind']:28s} x={s['x']*100:5.1f}%  y={s['y']*100:5.1f}%  "
                    f"(on {hits} of {len(flat_spots)} frames)")
        else:
            say("  No repeating defects — what was found is probably scene detail.")
        say()

    say("Tips:")
    say("  * Spots blow off with a blower → it's dust, nothing to worry about.")
    say("  * Thin line, repeats on all frames and won't blow off → scratch, grounds for a return.")
    say("  * Marked images: red circles — spots/pixels, orange — lines.")

    with open(os.path.join(out_dir, "report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print(f"\nDone. Results: {out_dir}")


if __name__ == "__main__":
    main()
