"""Writing fit results: CSV (one row per point), NPZ (arrays), YAML (settings used)."""
import csv
import datetime
import os

import numpy as np
import yaml

from .fitting import DERIVED, DERIVED_ERR

COLS = ("center_nm", "center_nm_err", "center_meV", "center_meV_err", "fwhm_meV", "fwhm_meV_err",
        "fwhm_nm", "height", "intensity", "intensity_err", "snr", "tail_meV", "tail_meV_err", "ok")
assert set(COLS) - {"ok"} == set(DERIVED) | set(DERIVED_ERR)
PIXEL_COLS = ("integrated_counts", "r2", "noise", "shift_meV", "shift_signif", "range_lo_nm", "range_hi_nm")


def peak_cols(p):
    """CSV columns of one peak (the recoil tail only for trion peaks)."""
    return [c for c in COLS if p["shape"] == "trion" or not c.startswith("tail")]


def write(outdir, cfg, grid, wl, present, fit, bg, prefix, log=print):
    os.makedirs(outdir, exist_ok=True)
    res, peaks = fit["res"], fit["peaks"]
    ks = np.arange(1, grid.npts + 1)
    rows, cols = grid.rowcol(ks)
    sx, sy = grid.stage_xy(ks)
    u = grid.units
    header = ["file", "loc", "row", "col", f"x_{u}", f"y_{u}", "x_rel_um", "y_rel_um", "present",
              *PIXEL_COLS, "fit_failed", "no_signal"]
    header += [f"{p['name']}_{c}" for p in peaks for c in peak_cols(p)]
    combos = fit.get("combined", [])
    header += [f"{c['name']}{s}" for c in combos for s in ("", "_ok")]
    xr, yr = grid.x_rel_um(), grid.y_rel_um()
    path = os.path.join(outdir, "results.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for i, k in enumerate(ks):
            line = [f"{prefix}_loc{k}", k, rows[i], cols[i], f"{sx[i]:.6f}", f"{sy[i]:.6f}",
                    f"{xr[cols[i]]:.4f}", f"{yr[rows[i]]:.4f}", int(present[i])]
            line += [_f(res[q][i]) for q in PIXEL_COLS] + [int(res["fit_failed"][i]), int(res["no_signal"][i])]
            for p in peaks:
                for c in peak_cols(p):
                    v = res[f"{p['name']}_{c}"][i]
                    line.append(int(v) if c == "ok" else _f(v))
            for c in combos:
                line += [_f(res[c["name"]][i]), int(res[c["name"] + "_ok"][i])]
            w.writerow(line)
    np.savez_compressed(os.path.join(outdir, "results.npz"), wl=wl, fit_mask=fit["fit_mask"], theta=fit["theta"],
                        present=present, nx=grid.nx, ny=grid.ny, x_rel_um=xr, y_rel_um=yr,
                        stage_x=sx, stage_y=sy, substrate=bg["substrate"], offset=bg["offset"],
                        **{k: v for k, v in res.items()})
    info = {
        "written": datetime.datetime.now().isoformat(timespec="seconds"),
        "data_folder": cfg["data"]["folder"], "run_prefix": prefix,
        "grid": {"nx": grid.nx, "ny": grid.ny, "x_step_um": abs(grid.x_step) * grid.to_um,
                 "y_step_um": abs(grid.y_step) * grid.to_um},
        "fit_range_nm": [float(wl[fit["fit_mask"]].min()), float(wl[fit["fit_mask"]].max())],
        "fit_range_moves_with_shift": bool(fit["ctx_range_moves"]) if "ctx_range_moves" in fit else False,
        "anchor": fit.get("anchor"),
        "baseline": cfg["fit"]["baseline"],
        "peaks": [{k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in p.items()} for p in peaks],
        "mean_spectrum_fit": [{k: (bool(v) if isinstance(v, (bool, np.bool_)) else float(v)) for k, v in d.items()}
                              for d in fit["mean"]["derived"]],
        "lineshape_check_rel_chi2": {k: {s: (x if isinstance(x, str) else float(x)) for s, x in v.items()}
                                     for k, v in fit["comparison"].items()},
        "good_fits": {p["name"]: int(res[f"{p['name']}_ok"].sum()) for p in peaks},
        "combined": [{"name": c["name"], "definition": c["label"], "kind": c["kind"], "a": c["a"], "b": c["b"],
                      "quantity": c["quantity"], "mean_spectrum_value": c["template"],
                      "median_over_good_points": c["median"], "good_points": c["good"]} for c in combos],
        "theta_layout": "per peak [E0 (eV), area (counts*eV), width(s) (eV)...], then baseline c0, c1, ... about "
                        "Emid. trion widths = [Gaussian FWHM, tail]; its E0 is the zero-momentum trion energy",
        "Emid_eV": float(fit["model"].Emid),
    }
    with open(os.path.join(outdir, "fit_info.yaml"), "w") as fh:
        yaml.safe_dump(info, fh, sort_keys=False)
    log(f"results: {path}")


def _f(v):
    return "" if not np.isfinite(v) else f"{v:.6g}"


def tag(peaks):
    """Folder name for a fit pass, e.g. 'X0-L_T-G' (L/G/V = Lorentzian/Gaussian/Voigt)."""
    return "_".join(f"{p['name']}-{p['shape'][0].upper()}" for p in peaks)  # T = trion
