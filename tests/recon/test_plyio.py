import numpy as np

from conftest import SCENE_PLY, make_splat
from pipeline.recon import plyio


def test_sample_roundtrip_is_byte_exact(tmp_path, sample_ply):
    out = tmp_path / "rt.ply"
    plyio.write_ply(str(out), sample_ply)
    assert out.read_bytes() == open(SCENE_PLY, "rb").read()
    assert len(sample_ply) == 160719 and len(sample_ply.names) == 59


def test_synthetic_roundtrip_and_reread(tmp_path):
    p = make_splat(1000)
    a = tmp_path / "a.ply"
    plyio.write_ply(str(a), p)
    q = plyio.read_ply(str(a))
    assert q.data.tobytes() == p.data.tobytes()
    b = tmp_path / "b.ply"
    plyio.write_ply(str(b), q)
    assert a.read_bytes() == b.read_bytes()


def test_big_endian_and_ascii_read(tmp_path):
    p = make_splat(50)
    be = plyio.Ply(p.data.astype(p.data.dtype.newbyteorder(">")), [], "binary_big_endian", None, b"")
    f = tmp_path / "be.ply"
    plyio.write_ply(str(f), be)
    q = plyio.read_ply(str(f))
    assert np.array_equal(plyio.xyz(q), plyio.xyz(p))
    asc = tmp_path / "a.ply"
    with open(asc, "w") as fh:
        fh.write("ply\nformat ascii 1.0\nelement vertex 2\nproperty float x\nproperty float y\n"
                 "property float z\nproperty uchar red\nend_header\n1 2 3 255\n4 5 6 0\n")
    r = plyio.read_ply(str(asc))
    assert r.data["z"].tolist() == [3, 6] and r.data["red"].tolist() == [255, 0]


def test_subset_keeps_records_exact(sample_ply):
    idx = np.arange(0, len(sample_ply), 97)
    s = sample_ply.subset(idx)
    assert s.data.tobytes() == sample_ply.data[idx].tobytes()


def test_conform_remaps_sh_per_channel():
    lo = make_splat(10, sh_rest=9)   # SH degree 1: 3 coeffs per channel
    hi = make_splat(1, sh_rest=45)   # SH degree 3: 15 per channel
    c = plyio.conform(lo, hi)
    assert c.data.dtype == hi.data.dtype
    for ch in range(3):
        for k in range(3):
            assert np.array_equal(c.data[f"f_rest_{ch * 15 + k}"], lo.data[f"f_rest_{ch * 3 + k}"])
        for k in range(3, 15):
            assert not c.data[f"f_rest_{ch * 15 + k}"].any()
    assert np.array_equal(c.data["opacity"], lo.data["opacity"])
