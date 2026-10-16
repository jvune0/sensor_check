# sensor_check

Checks a camera sensor from test shots for:

- **Dust and scratches** on flat (bright, uniform) frames
- **Hot / dead pixels** on dark frames

The script detects the frame type automatically, marks defects on images, and writes a text report.

## Installation

```bash
pip install numpy scipy pillow rawpy
```

`rawpy` is only needed for RAW files (`.cr2`, `.cr3`, `.dng`, `.nef`, `.arw`, `.raf`, `.orf`, `.rw2`). JPEG, PNG and TIFF work without it.

## Taking test shots

RAW is preferred; JPEG also works.

**Flat frames** (dust, scratches)
- Clear sky or a white wall
- Av mode, f/16–f/22, ISO 100
- Manual focus set to minimum distance so the background is blurred
- Take 2–3 frames, moving the camera slightly between them

**Dark frames** (hot pixels)
- Lens cap on
- M mode, ISO 100
- Exposures of 1 s and 30 s

## Usage

```bash
python sensor_check.py FOLDER_WITH_SHOTS
```

Results are saved to `FOLDER_WITH_SHOTS/sensor_check_result/`:

- `<name>_dust.png`: contrast-boosted flat frame. Red circles mark spots, orange circles mark lines.
- `<name>_hot.png`: dark frame with hot pixels circled in red
- `report.txt`: the full text report, the same as the console output

## Interpreting results

**Hot pixels per dark frame**

| Count   | Verdict |
|---------|---------|
| 0       | excellent |
| 1–30    | normal |
| 31–200  | a bit many. Try the camera's "Manual sensor cleaning" (pixel remapping). |
| > 200   | many. Cause for concern. |

**Dust and scratches**

- If there are 2 or more flat frames, the script lists defects that appear at the same position on several frames. Those are on the sensor. Anything else is probably scene detail.
- Spots that come off with a blower are dust and nothing to worry about.
- A thin line that repeats on every frame and doesn't blow off is likely a scratch, which is grounds for a return.

## Tuning

Constants at the top of `sensor_check.py`:

| Constant         | Default | Meaning |
|------------------|---------|---------|
| `DUST_THRESHOLD` | `0.02`  | How much darker than the background (2 %) a spot must be |
| `MIN_SPOT_AREA`  | `6`     | Minimum spot size in pixels |
| `HOT_SIGMA`      | `10`    | How far above the noise level (in robust sigmas) a pixel must be to count as hot |
