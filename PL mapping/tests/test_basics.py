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
    assert g.x_rel_um()[1] == pytest.approx(10.0)


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
