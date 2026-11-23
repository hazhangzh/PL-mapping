"""Scan-parameter file, main-exciton auto-detect and the reflection-contrast pipeline."""
import numpy as np
import pytest
import yaml

from plmap import HC, cli, config, fitting, lineshapes as ls, synthetic


def test_scan_file_points_and_translation(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "scan.txt").write_text(
        "# copied from the scan program\nstart_1_mm 3.108\nend_1_mm 3.128\npoints_1 41\n"
        "start_2_mm\t-7.9785\nend_2_mm\t-7.9585\npoints_2\t41\nexposure_s 5\noff_sample_1_mm 3.4\n")
    (tmp_path / "w.txt").write_text("700\n")
    (tmp_path / "c.yaml").write_text("data: {folder: data, wavelength_file: w.txt}\n")
    cfg = config.load(str(tmp_path / "c.yaml"))
    g = config.grid(cfg)
    assert (g.nx, g.ny) == (41, 41) and cfg["scan"]["exposure_s"] == 5
    assert g.x_rel_um()[0] == 0 and g.x_rel_um()[-1] == pytest.approx(20.0)
    assert g.y_rel_um()[-1] == pytest.approx(20.0) and g.x_step * g.to_um == pytest.approx(0.5)
    (tmp_path / "c2.yaml").write_text("data: {folder: data, wavelength_file: w.txt}\nscan: {x_start: 1}\n")
    with pytest.raises(SystemExit):                       # values in both places
        config.load(str(tmp_path / "c2.yaml"))
    (tmp_path / "data" / "scan.txt").write_text("start_1 3.1\n")
    with pytest.raises(SystemExit):                       # typo in a name
        config.load(str(tmp_path / "c.yaml"))


def test_date_named_folder(tmp_path):
    (tmp_path / "11042026").mkdir()
    (tmp_path / "w.txt").write_text("700\n")
    (tmp_path / "c.yaml").write_text("data: {folder: 11042026, wavelength_file: w.txt}\n"
                                     "scan: {x_start: 0, x_end: 1, x_points: 3, y_start: 0, y_end: 1, y_steps: 2}\n")
    cfg = config.load(str(tmp_path / "c.yaml"))
    assert cfg["data"]["folder"].endswith("11042026") and cfg["scan"]["x_steps"] == 2


def test_main_exciton_is_highest_energy_strong_peak():
    wl = np.linspace(695, 765, 1800)
    E = HC / wl
    y = np.zeros_like(wl)
    for c, f, h in [(703.0, 0.004, 0.8), (713.6, 0.007, 10), (728.5, 0.015, 13), (736.0, 0.010, 15)]:
        p = [HC / c, 1.0, f]
        y += h * ls.evaluate("lorentzian", E, p) / ls.height("lorentzian", p)
    y += np.random.default_rng(4).normal(0, 0.15, len(wl))
    found = fitting.detect_peaks(wl, y, 10, log=lambda *_: None)
    x0 = fitting.main_exciton(found)
    assert x0["name"] == "X0" and x0["center_nm"] == pytest.approx(713.6, abs=0.3)   # not the weak 703 bump
    lo, hi = fitting.auto_range(x0, wl)
    assert lo < 707 and 717 < hi < 722


@pytest.fixture(scope="module")
def rcmap(tmp_path_factory):
    d = tmp_path_factory.mktemp("rc")
    wl, truth = synthetic.make_rc_map(str(d / "data"))
    return d, wl, truth


@pytest.mark.parametrize("r0", ["auto", "file", "points"])
def test_rc_recovers_resonance(rcmap, r0):
    d, wl, t = rcmap
    rc = {"min_snr": 5}
    if r0 == "file":
        rc.update(r0="data/R0_2D.txt", r0_exposure_s=4)
    elif r0 == "points":
        off = np.flatnonzero(~t["band"] & ~t["lower"])[:6] + 1
        rc["r0"] = [int(k) for k in off]
    else:
        rc["r0"] = "auto"
    cfg = {"data": {"folder": "data", "wavelength_file": "data/wavelength.txt", "output": f"out_{r0}"},
           "spectra": {"plots": "none"}, "rc": rc}
    (d / f"{r0}.yaml").write_text(yaml.safe_dump(cfg))
    cli.main(["rc", str(d / f"{r0}.yaml")])
    out = d / f"out_{r0}" / "rc"
    for f in ("results.csv", "results.npz", "rc_info.yaml", "summary.png", "report.html"):
        assert (out / f).exists(), f
    z = np.load(out / "results.npz")
    band, lower = t["band"], t["lower"]
    assert z["ok"][band].mean() > 0.95
    assert not z["ok"][~band & ~lower].any()             # no false resonances on the background
    assert np.max(np.abs(z["center_meV"][band] - t["E0"][band] * 1e3)) < 0.3
    assert np.max(np.abs(z["fwhm_meV"][band] - t["G"][band] * 1e3)) < 0.6
    # 5 % baseline offset in the RC + 0.85x reflectance at the darker corner don't change A by much
    assert np.median(z["amplitude"][band]) == pytest.approx(t["A"] * 1.05, rel=0.08)
    assert np.median(z["phase_deg"][band]) == pytest.approx(np.degrees(t["phi"]), abs=4)
    info = yaml.safe_load(open(out / "rc_info.yaml"))
    if r0 == "file":
        assert 0.7 < info["r0_scale"] < 0.82               # lamp 30 % brighter in the R0 file
    if r0 == "auto":
        assert not set(info["r0_points"]) & set(np.flatnonzero(band) + 1)
