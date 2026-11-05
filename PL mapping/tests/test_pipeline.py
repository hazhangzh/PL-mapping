"""End-to-end run on a small synthetic map with known peaks."""
import csv
import os

import numpy as np
import yaml

from plmap import cli, lineshapes as ls, synthetic


def test_pipeline_recovers_truth(tmp_path):
    data = tmp_path / "data"
    wl, truth = synthetic.make_map(str(data), x_steps=10, y_steps=8, n_missing=2)
    cfg = {
        "data": {"folder": "data", "wavelength_file": "data/wavelength.txt", "output": "out"},
        "scan": {"x_start": 0.26, "x_end": 0.265, "x_steps": 10, "y_start": 0.25, "y_end": 0.254, "y_steps": 8},
        "spectra": {"plots": [1, 50]},
        "fit": {"range_nm": [700, 760], "workers": 2, "plots": [50], "peaks": [
            {"name": "X0", "center_nm": 710.6, "shape": "lorentzian", "window_nm": 5, "fwhm_meV": 8},
            {"name": "XT", "center_nm": 723.0, "shape": "voigt", "window_nm": 6, "fwhm_meV": 10},
            {"name": "LE", "center_nm": 737.7, "shape": "gaussian", "window_nm": 14, "fwhm_meV": 22}]},
    }
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(cfg))
    cli.main(["run", str(tmp_path / "config.yaml")])

    out = tmp_path / "out" / "fit_X0-L_XT-V_LE-G"
    for f in ("results.csv", "results.npz", "fit_info.yaml", "summary.png", "summary.pdf", "report.html",
              "fit_plots/1_loc50_fit.png"):
        assert (out / f).exists(), f
    assert sorted(os.listdir(tmp_path / "out" / "spectra_plots")) == ["1_loc1.png", "1_loc50.png"]

    rows = list(csv.DictReader(open(out / "results.csv")))
    assert len(rows) == 99 and rows[0]["file"] == "1_loc1" and rows[-1]["present"] == "0"
    assert float(rows[12]["x_mm"]) == np.float64(0.2605) and rows[12]["row"] == "1" and rows[12]["col"] == "1"

    z = np.load(out / "results.npz")
    on = (truth["flake"] > 0.7) & truth["present"]
    assert on.sum() > 5
    for name, tol_e, tol_w in (("X0", 0.15, 0.4), ("XT", 0.3, 0.8), ("LE", 1.5, 3.0)):
        t = truth["peaks"][name]
        e0 = t["E0"] * 1e3
        fw = np.array([ls.fwhm(t["shape"], [0, 1] + [w[i] for w in t["widths"]]) for i in range(len(e0))]) * 1e3
        assert z[f"{name}_ok"][on].all(), name
        assert np.max(np.abs(z[f"{name}_center_meV"][on] - e0[on])) < tol_e, name
        assert np.max(np.abs(z[f"{name}_fwhm_meV"][on] - fw[on])) < tol_w, name
    info = yaml.safe_load(open(out / "fit_info.yaml"))
    assert info["lineshape_check_rel_chi2"]["X0"]["recommended"] == "lorentzian"
    assert info["lineshape_check_rel_chi2"]["XT"]["recommended"] == "voigt"
    assert info["lineshape_check_rel_chi2"]["LE"]["recommended"] == "gaussian"
