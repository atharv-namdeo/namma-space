"""One command: capture source -> capture/ -> mesh + cloud + planes + nav map.

Source is either a PoseCam export (metric ARCore poses) or a plain video
(relative-scale odometry, for testing). Everything after the source is identical.

Usage:
    # Android / ARCore (metric):
    python src/run_pipeline.py --posecam data/posecam_scan1 --name scan1

    # plain video (relative scale, for testing the output stack):
    python src/run_pipeline.py --video data/clip1.mp4 --name clip1 --stride 2
"""
import argparse
import os

import backbone
import occupancy
import planes


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--posecam", help="PoseCam export folder (metric)")
    src.add_argument("--video", help="plain video (relative scale)")
    ap.add_argument("--name", required=True, help="run name")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--voxel", type=float, default=0.02)
    ap.add_argument("--sdf-trunc", type=float, default=0.08)
    ap.add_argument("--far", type=float, default=8.0)
    ap.add_argument("--target-tris", type=int, default=500_000)
    ap.add_argument("--fuse", choices=["auto", "points", "tsdf"], default="auto",
                    help="auto: tsdf for metric depth, points for relative depth")
    ap.add_argument("--depth-model", choices=["auto", "relative", "indoor", "outdoor"],
                    default="auto", help="auto: indoor(metric) for posecam, relative for video")
    ap.add_argument("--nav-res", type=float, default=0.05)
    ap.add_argument("--skip-capture", action="store_true",
                    help="reuse an existing capture/<name> folder")
    args = ap.parse_args()

    cap_dir = os.path.join("capture", args.name)
    out_dir = os.path.join("output", args.name + "_v2")

    model = args.depth_model
    if model == "auto":
        model = "indoor" if args.posecam else "relative"

    if not args.skip_capture:
        if args.posecam:
            import ingest_posecam
            ingest_posecam.run(args.posecam, cap_dir, args.stride, args.size,
                               args.max_frames, far=args.far, model=model)
        else:
            import make_capture
            make_capture.run(args.video, cap_dir, args.stride, args.size,
                             args.max_frames, far=args.far, model=model)

    # Metric depth (indoor/outdoor model) is scale-consistent -> TSDF gives a clean
    # mesh. Relative depth is per-frame normalised -> point fusion + Poisson instead.
    fuse = args.fuse
    if fuse == "auto":
        fuse = "tsdf" if model in ("indoor", "outdoor") else "points"
    backbone.run(cap_dir, out_dir, args.voxel, args.sdf_trunc, args.far,
                 args.target_tris, fuse=fuse)

    cloud = os.path.join(out_dir, "map_metric.ply")
    for name, fn in (("planes", lambda: planes.run(cloud, out_dir)),
                     ("occupancy", lambda: occupancy.run(cloud, out_dir, res=args.nav_res))):
        try:
            fn()
        except SystemExit as e:
            print(f"[pipeline] {name} skipped: {e}")
        except Exception as e:
            print(f"[pipeline] {name} failed: {e}")

    print("\n[pipeline] ALL DONE ->", out_dir)
    print("  mesh.obj / mesh.stl / mesh.ply      (watertight surface)")
    print("  map_metric.ply / .pcd + poses.tum   (cloud + trajectory, robotics)")
    print("  planes.json / boxes.json            (CAD primitives)")
    print("  floormap.pgm/.yaml/.png + occupancy_3d.npy  (nav map)")


if __name__ == "__main__":
    main()
