# Capture guide — how to film a room that reconstructs well

The #1 reason a splat comes out empty or broken is a **bad capture**, not the software. 90% of success
is here. (This is exactly why our earlier Scaniverse scans failed.)

## Do
- **Move slowly and smoothly.** Walk like you're carrying a full cup of coffee.
- **Lots of overlap.** Each part of the room should appear in *many* frames from *different* angles.
- **Orbit objects.** Circle a desk, a chair, a shelf — don't just pan across from one spot.
- **Cover several heights.** Low, eye-level, and a bit high. Get the floor and the ceiling edges.
- **Good, even light.** Turn the lights on. Daylight is great. Avoid harsh single-source shadows.
- **Fill the room with texture.** Clutter is your friend — books, posters, objects, patterns.
- **30–90 seconds** of video for a normal room is plenty.

## Avoid (these break reconstruction)
- ❌ **Blank walls, plain floors, white ceilings** — no features to track → empty splat.
- ❌ **Glass, mirrors, shiny/reflective surfaces** — the geometry "moves", confuses matching.
- ❌ **Fast motion / motion blur** — slow down.
- ❌ **Moving things** (people, pets, curtains, screens) — keep the scene static.
- ❌ **Low light / strong backlight** (a bright window behind your subject).
- ❌ **Zooming** while filming — keep zoom fixed.

## A good default recipe
1. Stand in a doorway or corner. Start recording.
2. Slowly walk a **loop** around the room, staying ~1–2 m from surfaces, phone held steady.
3. As you pass objects, **arc around them** instead of walking straight past.
4. Do a second, higher or lower pass if the room is big.
5. Stop. Copy the video into `data/`.

## Frame extraction tips
- `--fps 3` is a good start (3 frames/sec). Raise to 4–5 for fast walks, lower to 2 for slow ones.
- Keep total frames **~150–300**. More frames = slower COLMAP, not always better.
  ```powershell
  python scripts\extract_frames.py data\myroom.mp4 --fps 3 --max 300
  ```

## How to tell it worked
After COLMAP, `output\colmap\sparse\0\` should exist and COLMAP should have registered **most** of
your frames. If it registered only a handful, the capture was too feature-poor — re-film with more
texture and overlap.
