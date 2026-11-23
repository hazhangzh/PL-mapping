"""Trion lineshape + anchored, shift-tracked multi-peak fits on a synthetic map."""
import copy

import numpy as np
import pytest

from plmap import HC, config, fitting
from plmap import lineshapes as ls

scipy = pytest.importorskip("scipy")


def test_trion_matches_exponnorm_and_is_normalised():
    from scipy.stats import exponnorm
    x = np.linspace(1.60, 1.75, 30001)
    for tail in (1e-5, 1e-3, 5e-3):
        y = ls.trion(x, 1.69, 2.0, 0.004, tail)
        s = 0.004 / (2 * np.sqrt(2 * np.log(2)))
        ref = 2.0 * exponnorm.pdf(-(x - 1.69), tail / s, scale=s)     # mirrored: tail to low energy
        assert np.trapezoid(y, x) == pytest.approx(2.0, rel=1e-5)
        assert np.max(np.abs(y - ref)) < 1e-4 * ref.max()
    # tail -> 0 is a Gaussian; with a tail the maximum sits below x0 and the line broadens
    assert ls.fwhm("trion", [1.69, 1, 0.004, 1e-6]) == pytest.approx(0.004, rel=1e-3)
    assert ls.peak_position("trion", [1.69, 1, 0.004, 3e-3]) < 1.69
    assert ls.fwhm("trion", [1.69, 1, 0.004, 3e-3]) > 0.0045


PEAKS = [  # name, offset (meV), shape, true widths (eV), area
    ("X0", 0.0, "voigt", [0.003, 0.003], 0.15),
    ("XT1", -30.0, "gaussian", [0.006], 0.45),
    ("XT2", -36.5, "gaussian", [0.005], 0.55),
    ("XD", -41.5, "gaussian", [0.0035], 0.08),
    ("P", -50.5, "trion", [0.004, 0.002], 1.2),
]


def _synthetic(n=60, seed=1):
    rng = np.random.default_rng(seed)
    wl = np.linspace(696.6, 773.0, 2048)
    E = HC / wl
    shifts = rng.uniform(-0.006, 0.006, n)            # +-6 meV rigid shifts
    shifts[:3] = 0.0
    Ea = 1.7276
    spc = np.empty((n, len(wl)))
    for i in range(n):
        amp = 1.0 if i < n - 10 else 0.0              # last 10 points: off-flake noise
        y = 2.0 + 5.0 * (E - 1.69)
        for _, off, shape, w, a in PEAKS:
            y = y + amp * ls.evaluate(shape, E, np.r_[Ea + off / 1e3 + shifts[i], a, w])
        spc[i] = y + rng.normal(0, 1.5, len(wl))
    return wl, spc, shifts, Ea


def _cfg(**fit):
    cfg = copy.deepcopy(config.DEFAULTS)
    cfg["fit"].update(range_meV=[-62, 15], anchor="X0", max_shift_meV=10, workers=4, min_snr=4)
    cfg["fit"].update(fit)
    cfg["fit"]["peaks"] = [config.normalise_peak(
        {"name": n, "offset_meV": o + (1.0 if n != "X0" else 0.0),       # starting offsets 1 meV off
         "shape": "gaussian" if s == "trion" and n != "P" else s,
         "window_meV": 2.5 if n == "XD" else 3.0, "fwhm_meV": 5.0, "max_fwhm_meV": 12 if n != "XD" else 6})
        for n, o, s, _, _ in PEAKS]
    return cfg


def test_anchor_found_and_shifts_tracked():
    wl, spc, shifts, Ea = _synthetic()
    present = np.ones(len(spc), bool)
    out = fitting.run(_cfg(combined=COMBINED), wl, spc, present, log=lambda *a: None)
    res = out["res"]
    on = slice(0, len(spc) - 10)
    # X0 is the weakest of the strong peaks, yet the pattern match puts the anchor on it (at the
    # position of the brightest points, which define the template); an alias would be >= 6.5 meV off
    bright = out["mean"]["points"]
    assert out["peaks"][0]["E0"] == pytest.approx(Ea + np.median(shifts[bright]), abs=0.002)
    # the template recovers the true offsets although the configured ones were 1 meV off
    d = {p["name"]: x["center_meV"] for p, x in zip(out["peaks"], out["mean"]["derived"])}
    for name, off, *_ in PEAKS[1:]:
        if name != "P":       # P is a trion: its centre is the zero-momentum edge, fitted with a free tail
            assert d[name] - d["X0"] == pytest.approx(off, abs=0.5)
    # each point's X0 energy follows its own shift (well beyond any fixed +-3 meV window)
    err = res["X0_center_meV"][on] - (Ea + shifts[on]) * 1e3
    assert np.all(res["X0_ok"][on])
    assert np.max(np.abs(err)) < 0.5
    for name, off, *_ in PEAKS[1:3]:
        d = res[f"{name}_center_meV"][on] - res["X0_center_meV"][on]
        assert np.median(np.abs(d - off)) < 0.5
    # the fit range moved with the shift
    lo_E = HC / res["range_hi_nm"][on]
    assert np.allclose(lo_E - (out["peaks"][0]["E0"] - 0.062) - res["shift_meV"][on] / 1e3, 0, atol=2e-4)
    # off-flake points: no trusted shift, no good fits
    assert np.all(res["shift_meV"][-10:] == 0)
    assert not res["X0_ok"][-10:].any()
    # combined maps: trion/X0 height ratio and energy differences, valid where all their peaks are good
    ok = res["X0_ok"] & res["XT1_ok"] & res["XT2_ok"]
    assert np.array_equal(res["XT_X0"+"_ok"], ok) and ok[on].mean() > 0.9
    ratio = (res["XT1_height"] + res["XT2_height"]) / 2 / res["X0_height"]
    assert np.allclose(res["XT_X0"][ok], ratio[ok])
    h = {n: ls.height(s, np.r_[0.0, a, w]) for n, _, s, w, a in PEAKS}          # true heights
    assert np.median(res["XT_X0"][ok]) == pytest.approx((h["XT1"] + h["XT2"]) / 2 / h["X0"], rel=0.05)
    for name, off in (("dE_XT1_X0", -30.0), ("dE_XT2_X0", -36.5)):
        assert np.median(res[name][res[name + "_ok"]]) == pytest.approx(off, abs=0.3)
    assert [c["name"] for c in out["combined"]] == ["XT_X0", "dE_XT1_X0", "dE_XT2_X0"]


COMBINED = [{"name": "XT_X0", "ratio": [["XT1", "XT2"], "X0"], "quantity": "height"},
            {"name": "dE_XT1_X0", "difference": ["XT1", "X0"]},
            {"name": "dE_XT2_X0", "difference": ["XT2", "X0"]}]


def test_combined_spec_and_values():
    peaks = [{"name": n} for n in ("X0", "XT1", "XT2")]
    c = fitting.combined_spec(COMBINED + [{"ratio": ["XT1", "X0"], "quantity": "intensity"}], peaks)
    assert [x["quantity"] for x in c] == ["height", "center_meV", "center_meV", "intensity"]
    assert c[3]["name"] == "XT1_over_X0_intensity"
    val = {("X0", "height"): np.array([10.0, 0.0, 4.0]), ("XT1", "height"): np.array([4.0, 1.0, 2.0]),
           ("XT2", "height"): np.array([6.0, 1.0, 2.0])}
    oks = {"X0": np.array([1, 1, 1], bool), "XT1": np.array([1, 1, 0], bool), "XT2": np.array([1, 1, 1], bool)}
    v, g = fitting.combine(c[0], lambda n, q: val[n, q], lambda n: oks[n])
    assert v[0] == pytest.approx(0.5) and np.isnan(v[1]) and v[2] == pytest.approx(0.5)
    assert g.tolist() == [True, False, False]      # zero denominator; XT1 rejected
    for bad in ({"ratio": ["XT1", "XB"]}, {"ratio": ["XT1"]}, {"name": "X0_height", "difference": ["XT1", "X0"]},
                {"difference": ["XT1", "X0"], "quantity": "area"}, {"ratio": ["XT1", "X0"], "difference": ["XT1", "X0"]}):
        with pytest.raises(SystemExit):
            fitting.combined_spec([bad], peaks)


def test_fixed_windows_without_tracking_lose_shifted_points():
    """Same data without tracking: points shifted by more than the window are not recovered."""
    wl, spc, shifts, Ea = _synthetic()
    present = np.ones(len(spc), bool)
    out = fitting.run(_cfg(track_shift=False), wl, spc, present, log=lambda *a: None)
    res = out["res"]
    on = np.arange(len(spc) - 10)
    t0 = out["peaks"][0]["E0"] - Ea                     # template position relative to the truth
    big = on[np.abs(shifts[on] - t0) > 0.005]           # beyond the +-3 meV centre windows
    assert len(big)
    bad = ~res["X0_ok"][big] | (np.abs(res["XT2_center_meV"][big] - res["X0_center_meV"][big] + 36.5) > 1.0)
    assert bad.mean() > 0.7
