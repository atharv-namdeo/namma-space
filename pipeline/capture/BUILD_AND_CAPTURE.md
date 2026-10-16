# PoseCam: build the app, record a scan, feed it to monomap

Goal: get **metric, gravity-aligned** camera poses + RGB (+ IMU) from an Android
phone, so monomap produces a level, real-scale map + clean CAD/nav outputs.

PoseCam has **no APK** — you build it once. The repo is already cloned at
`android/PoseCam`. Versions in this repo (checked): AGP 9.4.0, Gradle 9.7.1,
**compileSdk 37**, minSdk 24, JDK 17, ARCore 1.56.

---

## 0. Prerequisite — is your phone supported?
- Settings → About phone → model, and check it's on Google's ARCore list:
  https://developers.google.com/ar/devices  (if it's not there, PoseCam won't run).
- Android 7.0+ and ~20 GB free (a scan is ~1.5 MB/s raw).

## 1. Install Android Studio (one time)
- Get the **latest stable** Android Studio: https://developer.android.com/studio
  (AGP 9.4 / Gradle 9.7.1 need a current version — an old Studio will reject the sync).
  It bundles the JDK and Gradle.
- First launch → **SDK Manager** (gear icon) → **SDK Platforms** tab → tick
  **Android SDK Platform 37** (enable "Show package details" if you don't see it).
  → **SDK Tools** tab → tick **Android SDK Build-Tools**, **Android SDK Platform-Tools**
  (gives `adb`). Apply.
  - *If Platform 37 isn't available yet on your channel:* open
    `android/PoseCam/app/build.gradle.kts`, change `compileSdk = 37` and
    `targetSdk = 37` to `36`, re-sync. (Safe for this app.)

## 2. Enable the phone for installing
- Settings → About phone → tap **Build number** 7× → enables Developer options.
- Settings → Developer options → turn on **USB debugging**.
- Plug the phone into the PC → accept **Allow USB debugging?** on the phone.

## 3. Build + install (debug APK — no keystore needed)
- Android Studio → **Open** → select the `android/PoseCam` folder → let **Gradle sync**
  finish (first time downloads Gradle 9.7.1 + ARCore; needs internet, a few minutes).
- With the phone connected and picked in the device dropdown, press **Run ▶**.
  It builds the **debug** variant, installs, and launches on the phone.
- CLI alternative (from `android/PoseCam`):
  ```
  gradlew.bat assembleDebug
  adb install -r app\build\outputs\apk\debug\app-debug.apk
  ```
- In the app: allow **camera**, and install **Google Play Services for AR** if prompted.
  Samsung: Settings → Apps → PoseCam → Battery → **Unrestricted**.

## 4. Record a ROOM scan (we are NOT doing the gripper workflow)
Ignore every gripper/jaw instruction in PoseCam's own docs — those are for a robot
training pipeline. For 3D mapping:
- Leave **640×480** and **Focus: auto** as they are.
- Move slowly until the bar says **"Ready to record"** (~3 s of stable tracking).
- **Record** → walk the space slowly and smoothly → **Stop**. One continuous take.
- **Point at textured things** (furniture, posters). Avoid blank walls/ceilings and
  dark areas — tracking is lost there, and ARCore may **jump** on recovery.
- **Don't walk a loop back to the exact start** in one take — returning to start is a
  classic relocalization jump. Prefer a smooth sweep/forward path. (Our ingest keeps
  the longest jump-free segment anyway, but fewer jumps = more usable frames.)
- Good light; move gently (exposure ~20 ms indoors, fast motion blurs).

## 5. Get the RAW session onto the PC  (IMPORTANT: raw, not the MP4 export)
In the app → **Recordings** → tap your recording → **Share raw recording (zip)** or
**Save raw zip to Downloads**. (Do **not** use "Export for pipeline (MP4)" — that
rotates the video and drops intrinsics; monomap reads the raw session.)
- Copy the zip to the PC (USB "File transfer" → phone's `Download/PoseCam/`), and
  unzip into `android/PoseCam/data/` so you get `android/PoseCam/data/capture-XXXX/`
  containing `frames/`, `poses.csv`, `intrinsics.json`, `imu.csv`.

## 6. Run monomap on it
```
cd "C:\Users\Atharv\Desktop\3d mapping\monomap"
venv\Scripts\python src\run_pipeline.py --posecam android\PoseCam\data\capture-XXXX --name scan1
```
Outputs in `output\scan1_v2\`: `mesh.obj/stl/ply`, `map_metric.ply/.pcd` + `poses.tum`,
`planes.json`/`boxes.json`, `floormap.pgm/.yaml/.png` + `occupancy_3d.npy`.

Because the poses are now metric + gravity-aligned, the floor will be level, the map
in real metres, planes classified correctly, and the floorplan room-shaped — unlike
the relative-depth test clips. `imu.csv` is carried into the capture folder for the
rover visual-inertial bridge later.
```
```
