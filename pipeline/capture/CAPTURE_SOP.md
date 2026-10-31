# Capture SOP: Nothing Phone 3a (and any phone without LiDAR)

This is the procedure for recording a space so stages 2-4 work. The Nothing Phone 3a has
no LiDAR or ToF sensor, but it supports ARCore, including the Depth API (depth estimated
from motion). Check that the phone is still listed at https://developers.google.com/ar/devices.
That gives three capture modes, from best to simplest:

| Mode | App | What you get | Scale |
|---|---|---|---|
| **A. PoseCam RAW** (recommended) | PoseCam (see `BUILD_AND_CAPTURE.md`) | frames + ARCore poses + IMU (+ ARCore depth if enabled) | metric, gravity-aligned |
| **B. Plain video + range sensor** | stock camera + ESP32 logger | video + `range.csv` | metric after fusion |
| **C. Plain video** | stock camera | video | relative only (`metric:false`) |

## 1. Phone settings (all modes)

- Use the **1x main camera**. The ultrawide's distortion breaks the pinhole model.
- Record **1080p at 30 fps**. 4K adds nothing, because frames are downscaled to 640 px.
- Turn **video stabilisation / steady mode OFF**. EIS crops and warps every frame, so the
  intrinsics drift from frame to frame.
- Turn HDR and beauty/AI scene modes off. **Lock exposure and focus** (long-press, AE/AF lock)
  on a mid-brightness area.
- Use bright, even light. Indoors, exposure should be about 1/60 s or faster; slow motion is
  worth more than anything else.
- Clean the lens, and have at least 5 GB free.

## 2. How to move

General rules:
- Move **slowly**: about 0.3 m/s walking and about 30°/s turning. The ingest rejects blurred
  frames, but every rejected frame is a gap.
- **Translate, don't pivot.** Turning on the spot gives no parallax, so there is no depth and
  no splat. Side-step while you turn.
- Hold the phone at chest height (about 1.3-1.5 m), upright and level. Floor alignment
  assumes the phone was roughly upright.
- Keep **textured things in view**: furniture, posters, switchboards. Blank walls, glass and
  mirrors lose tracking. Tilt slightly down so the floor is in the bottom quarter of the frame,
  because stage 2 finds the floor from it.
- Make one continuous take of 1-3 minutes per room. Don't finish exactly where you started in
  PoseCam, because ARCore relocalisation can jump. The ingest keeps the longest jump-free run.

### Room
1. Start in a doorway and give it 3 s pointing at a textured area so ARCore initialises.
2. Walk the **perimeter about 1 m from the walls, facing inward** toward the opposite wall.
3. Do a second, slower pass **through the middle**, facing each wall in turn.
4. For important objects (labels, stage 3), do a short arc around each one at 1-1.5 m.
5. End by looking back at the door, so the room overlaps the hallway scan.

### Hallway
1. Walk **down the centre line facing forward**, with a gentle side-to-side zig-zag (about
   ±0.5 m) so the walls get parallax.
2. At each door, stop and pan into the room for 2-3 s. Those frames tie the room scan to the
   hallway in the building map (stage 4).
3. At the end, turn around slowly while side-stepping, then walk back facing the other way.
   Long featureless corridors need both directions.
4. Split hallways longer than about 25 m into several takes that overlap by about 3 m.

## 3. Optional range sensor (metric scale for modes B and C)

A monocular video has no scale. A single cheap range sensor fixes it. `pipeline/capture/fusion.py`
takes the robust median of `sensor_range / predicted_depth` at the sensor's pixel, rejects
outliers with an iterative MAD test, and writes `meta.json {metric:true, scale, scale_inliers}`.

### Hardware (about ₹600)
- **ESP32 DevKit** plus either:
  - **VL53L1X ToF** (recommended): 4 m long-distance mode, ±1-3 cm, about 27° FoV (set a 4x4
    ROI to narrow it to about 15°). Wire VIN→3V3, GND→GND, SDA→GPIO21, SCL→GPIO22.
  - **HC-SR04 ultrasonic**: 2 cm-4 m, about 15° cone, unreliable on soft or angled
    surfaces. Wire VCC→5V (VIN), GND→GND, TRIG→GPIO5, and ECHO→GPIO18 **through a
    1 kΩ/2 kΩ divider**. ECHO is 5 V and the ESP32 is 3.3 V.
- **Mounting:** tape or clip the sensor directly beside the main camera with its boresight
  **parallel to the optical axis**. Measure the baseline (typically 2-3 cm). At 1 m and fx≈544
  px, that is a parallax of about 15 px, which is negligible. For accuracy, pass the pixel the
  sensor hits as `u,v` per reading, or calibrate it once by aiming at a small dark dot.

### Logging
Example ESP32 sketch (Arduino, Pololu `VL53L1X` library). It prints CSV over USB serial:
```cpp
#include <Wire.h>
#include <VL53L1X.h>
VL53L1X s;
void setup() {
  Serial.begin(921600); Wire.begin(21, 22); Wire.setClock(400000);
  s.init(); s.setDistanceMode(VL53L1X::Long); s.setMeasurementTimingBudget(33000);
  s.startContinuous(33);
  Serial.println("timestamp_ns,range_m");
}
void loop() {
  uint16_t mm = s.read();
  if (!s.timeoutOccurred() && s.ranging_data.range_status == 0)
    Serial.printf("%llu,%.3f\n", (unsigned long long)esp_timer_get_time() * 1000ULL, mm / 1000.0);
}
```
Log it with a phone USB-OTG serial app (for example "Serial USB Terminal", saving to file) or
a laptop in a backpack (`python -m serial.tools.miniterm COM5 921600 > range.csv`).

### Clock sync: the hand tap
The ESP32 clock and the video clock are unrelated, so sync them once per take:
1. Start recording video and the logger.
2. **Cover the sensor and lens with your palm** for about 0.5 s, then pull it away. The sensor
   reads under 0.15 m at the same moment the frame goes dark.
3. After ingest, find the index of the first dark frame in `capture/rgb/` and pass
   `--range-sync-frame <index>`. The first reading under 0.15 m is aligned to that frame.
   Or pass a known offset with `--range-offset-ns`.

```
python -m pipeline.capture.ingest walk.mp4 --out data/captures/roomB \
    --range-csv range.csv --range-sync-frame 3
```
Check `meta.json`. You want `scale_inliers` in the tens or more and `scale_log_spread` below
0.1. A large spread means the sync is off or the sensor is not boresighted.

### ARCore depth (mode A)
If your PoseCam build saves ARCore Depth API images (`depth/<frame>_<ts>.png`, DEPTH16 or mm),
they are used automatically. With `--depth-model none`, they become the capture depth directly
(metric, no torch). With a mono depth model, they set its metric scale.

## 4. Re-capture of a region (iterative refinement)

Use this when part of the twin is bad (a blurry corner, an object that moved, or a newly
furnished area):
1. In the web viewer, note the region's **world-frame box** (min/max x, y, z in metres, floor
   at y = 0).
2. Record a **short take (20-60 s)** with the **same phone and settings**:
   - **start with 3-5 s on an already-scanned, well-textured area next to the region**. These
     frames are what registers the new take onto the old scene, so it needs overlap.
   - cover the region from several angles, translating as you go (section 2 rules);
   - **end on overlap again**.
3. Ingest it like any capture, then refine:
```
python -m pipeline.capture.ingest fix.mp4 --out data/captures/cliproom_fix1
python -m pipeline.recon.refine --scene web/scenes/cliproom --capture data/captures/cliproom_fix1 \
    --region=-0.6,0,-3,0.6,0.6,-1.5 --trainer brush
```
Only the box (plus a feather margin) changes. Everything else in the splat stays
byte-identical, and the previous version is kept as `scene.v<N>.ply`.

## 5. Checklist before you leave the site
- [ ] Each room: a perimeter pass, a centre pass, and the door seen from inside
- [ ] Hallway: both directions, with a pan into every door
- [ ] Range logger (if used): the tap at the start of every take, and the CSV saved
- [ ] PoseCam: copy the **RAW session** (not the MP4 export) off the phone
- [ ] A few photos of signage or room numbers (labels, stage 3)
