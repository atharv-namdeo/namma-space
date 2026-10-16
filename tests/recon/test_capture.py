"""Stage 1 ingest paths that run without torch: PoseCam zip, video frames, splat import."""
import csv
import os
import zipfile

import cv2
import numpy as np

from conftest import SAMPLE, make_splat
from pipeline.capture import common as cm, ingest
from pipeline.recon import plyio


def _posecam_session(root, n=10, with_depth=True):
    os.makedirs(root / "frames")
    if with_depth:
        os.makedirs(root / "depth")
    names = cm.frame_names(SAMPLE)
    P = cm.load_poses(SAMPLE)
    rows = []
    for i in range(n):
        ts = 10_000_000 + i * 33_000_000
        img = cv2.imread(os.path.join(SAMPLE, "rgb", names[i]))
        big = cv2.resize(img, (1280, 720))
        cv2.imwrite(str(root / "frames" / f"{i:06d}_{ts}.jpg"), big)
        T_gl = P[i] @ cm.GL_TO_CV          # CV -> GL is the same flip
        q = cm.mat_to_quat(T_gl[:3, :3])
        state = "TRACKING" if i != 7 else "PAUSED"
        rows.append([i, ts, state, *T_gl[:3, 3], *q])
        if with_depth:
            d = cm.read_depth_m(SAMPLE, names[i])
            small = cv2.resize(d, (160, 90), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(root / "depth" / f"{i:06d}_{ts}.png"), np.round(small * 1000).astype(np.uint16))
    with open(root / "poses.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_index", "timestamp_ns", "tracking_state", "tx", "ty", "tz", "qx", "qy", "qz", "qw"])
        w.writerows(rows)
    (root / "intrinsics.json").write_text('{"fx": 1088, "fy": 1088, "cx": 640, "cy": 360, '
                                          '"width": 1280, "height": 720}')
    (root / "imu.csv").write_text("sensor,timestamp_ns,x,y,z,bias_x,bias_y,bias_z\nacc,1,0,9.8,0,0,0,0\n")


def test_posecam_zip_ingest(tmp_path):
    sess = tmp_path / "capture-0001"
    _posecam_session(sess)
    z = tmp_path / "session.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for dp, _, fs in os.walk(sess):
            for f in fs:
                zf.write(os.path.join(dp, f), os.path.relpath(os.path.join(dp, f), tmp_path))
    out = str(tmp_path / "cap")
    ingest.main([str(z), "--out", out, "--stride", "1"])
    meta = cm.load_meta(out)
    names = cm.frame_names(out)
    # runs: [0] (the sample's 0->1 step is a >3 m/s 'relocalisation' jump), [1..6], [8, 9]
    assert len(names) == 6
    assert meta["metric"] and meta["depth"] == "arcore-depth-api"
    intr = cm.load_intrinsics(out)
    assert intr["width"] == 640 and abs(intr["fx"] - 544) < 1e-6
    assert np.allclose(cm.load_poses(out), cm.load_poses(SAMPLE)[1:7], atol=1e-6)
    d = cm.read_depth_m(out, names[0])
    assert d.shape == (360, 640) and d.max() > 0
    assert os.path.exists(os.path.join(out, "imu.csv"))
    ts = cm.load_timestamps(out)
    assert ts[1] - ts[0] == 33_000_000


def test_video_frames_blur_rejection(tmp_path):
    names = cm.frame_names(SAMPLE)
    vid = str(tmp_path / "walk.avi")
    vw = cv2.VideoWriter(vid, cv2.VideoWriter_fourcc(*"MJPG"), 12, (640, 360))
    blurred = 0
    for i in range(48):
        img = cv2.imread(os.path.join(SAMPLE, "rgb", names[i * 2]))
        if i % 6 == 0:                  # whole 'kept' slots that are blurred
            img = cv2.GaussianBlur(img, (0, 0), 6)
            blurred += 1
        vw.write(img)
    vw.release()
    out = str(tmp_path / "cap")
    ingest.main([vid, "--out", out, "--poses", "none", "--fps", "12", "--oversample", "1",
                 "--blur-rel", "0.5", "--no-ffmpeg"])
    n = len(cm.frame_names(out))
    assert n == 48 - blurred
    assert cm.load_poses(out).shape == (n, 4, 4)
    assert cm.load_intrinsics(out)["fx"] == 0.85 * 640
    assert len(cm.load_timestamps(out)) == n


def test_sharpest_of_group():
    from pipeline.capture.video import sharpness
    img = cv2.imread(os.path.join(SAMPLE, "rgb", "00000.jpg"))
    assert sharpness(img) > 3 * sharpness(cv2.GaussianBlur(img, (0, 0), 4))


def test_ply_import(tmp_path):
    p = make_splat(500)
    src = tmp_path / "scan.ply"
    plyio.write_ply(str(src), p)
    out = str(tmp_path / "cap")
    ingest.main([str(src), "--out", out, "--metric", "--up-hint", "0,1,0"])
    meta = cm.load_meta(out)
    assert meta["scene_only"] and meta["gaussian_splat"] and meta["metric"] and meta["n_gaussians"] == 500
    assert (tmp_path / "cap" / "scene" / "scene.ply").read_bytes() == src.read_bytes()


def test_sample_range_fusion_with_frame_period(tmp_path):
    """The README sample command: synthetic ToF log (true scale 0.6, readings < 6 m) on data/sample/clip1."""
    names = cm.frame_names(SAMPLE)
    rows = ["timestamp_ns,range_m", "500000000,0.05"]       # hand tap, sensor clock
    rng = np.random.default_rng(0)
    for i in range(0, len(names), 5):
        d = cm.read_depth_m(SAMPLE, names[i])
        z = float(np.median(d[178:183, 318:323])) * 0.6 * np.exp(rng.normal(0, 0.01))
        rows.append(f"{500000000 + int(i * 66.7e6)},{z:.4f}")
    (tmp_path / "r.csv").write_text("\n".join(rows))
    out = str(tmp_path / "metric")
    ingest.main([SAMPLE, "--out", out, "--frame-period-ms", "66.7",
                 "--range-csv", str(tmp_path / "r.csv"), "--range-sync-frame", "0"])
    meta = cm.load_meta(out)
    assert abs(meta["scale"] - 0.6) < 0.01 and meta["metric"] is True and meta["scale_inliers"] > 20
    assert np.allclose(cm.load_poses(out)[:, :3, 3], cm.load_poses(SAMPLE)[:, :3, 3] * meta["scale"],
                       atol=1e-5)
    assert not os.path.exists(os.path.join(SAMPLE, "meta.json"))


def test_existing_capture_fusion_is_not_in_place(tmp_path):
    """kind=capture copies before rescaling; the input is untouched."""
    from test_scale import _synthetic_capture
    cap, metric, ts = _synthetic_capture(tmp_path, 2.0)
    before = open(cm.depth_path(cap, "00000.jpg"), "rb").read()
    rows = ["timestamp_ns,range_m"] + [f"{t},{metric[i][24, 32]:.4f}" for i, t in enumerate(ts)]
    (tmp_path / "r.csv").write_text("\n".join(rows))
    out = str(tmp_path / "fused")
    ingest.main([cap, "--out", out, "--range-csv", str(tmp_path / "r.csv")])
    assert open(cm.depth_path(cap, "00000.jpg"), "rb").read() == before
    assert abs(cm.load_meta(out)["scale"] - 2.0) < 0.02
