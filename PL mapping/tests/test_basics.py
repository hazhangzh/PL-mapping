import numpy as np
import pytest

from plmap import HC, io, lineshapes as ls, spectra
from plmap.grid import ScanGrid


def test_grid_steps_and_raster():
    g = ScanGrid(0.26, 0.30, 40, 0.25, 0.29, 40, "mm")
    assert (g.nx, g.ny, g.npts) == (41, 41, 1681)          # 40 steps = 41 points
    assert g.rowcol(1) == (0, 0) and g.rowcol(41) == (0, 40) and g.rowcol(42) == (1, 0)
    x, y = g.stage_xy(42)
    assert x == pytest.approx(0.26) and y == pytest.approx(0.251)
    x, y = g.stage_xy(1681)
    assert x == pytest.approx(0.30) and y == pytest.approx(0.29)
    assert g.x_rel_um()[-1] == pytest.approx(40.0)


def test_grid_reversed_axis():
    g = ScanGrid(0.30, 0.26, 4, 0, 0.002, 2, "mm")
    assert g.stage_xy(2)[0] == pytest.approx(0.29)
    assert g.x_rel_um()[1] == pytest.approx(-10.0)          # pure translation: start at 0


def test_load_matrix_crlf_and_dropped_pixel(tmp_path):
    p = tmp_path / "1_loc1.txt"
    p.write_bytes(b"1\t10\r\n2\t--\r\n3\t30\r\n")
    a = io.load_matrix(str(p))
    assert a.shape == (3, 2)
    assert a[1, 1] == pytest.approx(20.0)


def test_find_loc_files_prefix(tmp_path):
    for n in ("1_loc1.txt", "1_loc2.txt", "1_loc10.txt", "notes.txt"):
        (tmp_path / n).write_text("0")
    prefix, files = io.find_loc_files(str(tmp_path))
    assert prefix == "1" and sorted(files) == [1, 2, 10]
    (tmp_path / "2_loc1.txt").write_text("0")
    with pytest.raises(SystemExit):
        io.find_loc_files(str(tmp_path))
    assert sorted(io.find_loc_files(str(tmp_path), 2)[1]) == [1]


def test_cosmic_ray_rejected_but_narrow_spot_kept():
    rng = np.random.default_rng(0)
    n, nrow = 800, 25
    x = np.arange(n)
    peak = 200 * np.exp(-0.5 * ((x - 400) / 30) ** 2)
    rows = np.zeros(nrow)
    rows[10:14] = 1.0                                    # PL lands on only 4 CCD rows
    a = 800 + peak[:, None] * rows[None, :] + rng.normal(0, 4, (n, nrow))
    truth = a.mean(1)
    a[100, 5] += 3000                                    # cosmic ray off the peak
    a[405, 11] += 3000                                   # cosmic ray on the peak
    sp, ncr = spectra.average_rows(a, 6.0)
    assert ncr >= 2
    assert abs(sp[100] - truth[100]) < 5 and abs(sp[405] - truth[405]) < 10
    assert np.max(np.abs(sp - truth)[np.r_[0:95, 110:395, 415:800]]) < 1e-9   # untouched elsewhere


@pytest.mark.parametrize("shape,p", [("lorentzian", [1.7, 2.0, 0.006]), ("gaussian", [1.7, 2.0, 0.006]),
                                     ("voigt", [1.7, 2.0, 0.004, 0.003])])
def test_lineshape_area_and_fwhm(shape, p):
    x = np.linspace(1.0, 2.4, 2_000_001)
    y = ls.evaluate(shape, x, p)
    trapz = getattr(np, "trapezoid", None) or np.trapz      # numpy 2 / numpy 1
    # Lorentzian tails beyond the +-0.7 eV window hold ~0.3 % of the area
    assert trapz(y, x) == pytest.approx(2.0, rel=4e-3 if shape != "gaussian" else 1e-6)
    above = x[y >= y.max() / 2]
    assert above[-1] - above[0] == pytest.approx(ls.fwhm(shape, p), rel=1e-3)
    assert ls.height(shape, p) == pytest.approx(y.max(), rel=1e-6)


def test_voigt_matches_scipy():
    from scipy.special import voigt_profile
    x = np.linspace(1.6, 1.8, 3001)
    for fg, fl in [(0.006, 0.005), (0.0005, 0.02), (0.02, 0.0005)]:
        s = fg / (2 * np.sqrt(2 * np.log(2)))
        ref = voigt_profile(x - 1.7, s, fl / 2)
        assert np.max(np.abs(ls.voigt(x, 1.7, 1.0, fg, fl) - ref)) < 1e-6 * ref.max()


def _edge_map(pl_at_red, n=400, rng=None):
    """Map with drifting CCD offset (sloped across the chip), 20 on-flake points
    whose PL tail optionally reaches into the red edge."""
    rng = rng or np.random.default_rng(1)
    wl = np.linspace(691.6, 768.0, 2048)
    drift = rng.normal(0, 4, n)[:, None]
    sp = 820 + drift - 0.03 * (wl - 700) + rng.normal(0, 0.9, (n, len(wl)))
    peak = 10 * np.exp(-0.5 * ((wl - 735) / (12 if pl_at_red else 5)) ** 2)
    sp[:20] += peak
    return wl, sp, np.ones(n, bool)


@pytest.mark.parametrize("pl_at_red,expect", [(True, "blue"), (False, "red")])
def test_offset_band_avoids_pl_tail(pl_at_red, expect):
    from plmap import background
    wl, sp, present = _edge_map(pl_at_red)
    m = background.choose_offset_band(wl, sp, present, log=lambda *_: None)
    assert (wl[m].mean() < 700) == (expect == "blue")


def test_bright_points_only_on_flake():
    from plmap import background
    rng = np.random.default_rng(2)
    tot = rng.normal(0, 100, 1681)
    tot[:70] += 5000                                     # 70 on-flake points out of 1681
    b = background.bright_points(tot, np.ones(1681, bool))
    assert set(b) == set(range(70))
    flat = rng.normal(5000, 100, 300)                    # whole map on the flake: top 10 %
    assert len(background.bright_points(flat, np.ones(300, bool))) == 30


def test_detect_peaks_widths():
    from plmap import fitting
    wl = np.linspace(700, 760, 1600)
    E = HC / wl
    y = sum(h * ls.evaluate("lorentzian", E, [HC / c, 1.0, f]) / ls.height("lorentzian", [HC / c, 1.0, f])
            for c, f, h in [(713.6, 0.006, 10), (728.5, 0.015, 12), (736.0, 0.012, 14)])
    y = y + np.random.default_rng(3).normal(0, 0.2, len(wl))
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")                   # used to warn "prominence of 0"
        peaks = fitting.detect_peaks(wl, y, 4, log=lambda *_: None)
    assert [round(p["center_nm"]) for p in peaks] == [714, 728, 736]
    # widths only seed the mean-spectrum fit; overlapping peaks read narrow, but
    # they must not collapse to the 1 meV clip as they did before
    assert peaks[0]["fwhm_meV"] == pytest.approx(6, rel=0.25)
    for p in peaks:
        assert p["fwhm_meV"] > 3 and p["window_nm"] > 2.0
