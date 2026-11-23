"""Reflection-contrast (RC) maps:  RC = (R - R0) / R0.

Reuses steps 1-2 of the PL pipeline (loc files -> averaged spectra -> map grid),
then per point:

  R   = (spectrum - dark) / exposure                (counts/s per CCD row)
  R0  = background reflectance, from
          * auto:     the median R of featureless map points (no exciton resonance)
          * [locs]:   the median R of points you pick on the map
          * a file:   a measured R0 spectrum, scaled to the map (exposure, or matched
                      to the map's featureless points)
  RC  = R / R0 - 1

The exciton shows up in RC as a dip or a dispersive "heart-beat" feature. It is
fitted at every point with a complex Lorentzian plus a linear baseline,

  RC(E) = A * Re[ exp(i phi) * (G/2) / (E0 - E - i G/2) ] + c0 + c1 (E - Emid),

which covers every mix of absorptive (dip/peak, phi = +-90 deg) and dispersive
(phi = 0/180 deg) shapes produced by thin-film interference. A is the
peak-to-peak size of the feature, E0 the resonance energy and G its FWHM.
For fixed E0 and G the model is linear, so a grid search gives robust start
values for every point before the non-linear refinement.
"""
import datetime
import os

import numpy as np

from . import HC, background, config, plots, spectra

PARAMS = ("E0", "G", "a", "b", "c0", "c1")
OUT = ("center_meV", "center_meV_err", "center_nm", "fwhm_meV", "fwhm_meV_err", "amplitude",
       "amplitude_err", "phase_deg", "baseline", "noise", "snr", "r2")
LABELS = {
    "amplitude": "resonance amplitude A (peak-to-peak RC)", "center_meV": "resonance energy E0 (meV)",
    "center_nm": "resonance wavelength (nm)", "fwhm_meV": "linewidth Γ (FWHM, meV)",
    "phase_deg": "phase φ (deg; 0 = dispersive, -90 = dip)", "baseline": "RC baseline at window centre",
    "snr": "SNR (A / residual noise)", "r2": "fit R²", "feature_p2p": "feature size (model-free p2p)",
    "R_mean": "reflected intensity R (counts/s/row)", "rc_level": "median RC in window",
}


MIN_LAMP = 0.03


def _ranges_nm(wl, m):
    idx = np.flatnonzero(m)
    if not len(idx):
        return "none"
    br = np.flatnonzero(np.diff(idx) > 1)
    starts, ends = np.r_[idx[0], idx[br + 1]], np.r_[idx[br], idx[-1]]
    return ", ".join(f"{wl[a]:.1f}-{wl[b]:.1f} nm" for a, b in zip(starts, ends))


# ------------------------------------------------------------------ R, R0, RC
def dark_level(cfg, wl, S, present, log=print):
    d = cfg["rc"]["dark"]
    if d in ("auto", None):
        m = wl >= wl.max() - 3.0
        v = float(np.nanmedian(S[present][:, m]))
        log(f"dark (CCD offset) = {v:.1f} counts: median of the red-most 3 nm, where the lamp is ~0 "
            "(set rc.dark to a measured value or file to change)")
        return np.full(len(wl), v)
    if config._is_number(d):
        log(f"dark (CCD offset) = {float(d):.1f} counts (rc.dark)")
        return np.full(len(wl), float(d))
    log(f"dark spectrum from {os.path.basename(d)}")
    return background.load_spectrum(d, wl)


def _smooth(a, n=7):
    from scipy.ndimage import uniform_filter1d
    return uniform_filter1d(a, n, axis=-1, mode="nearest")


def highpass(q, width_lines=301):
    """Remove the broad shape of RC (interference, lamp, R0 mismatch), keep resonances."""
    from scipy.ndimage import median_filter
    s = _smooth(np.nan_to_num(q), 7)
    return s - median_filter(s, size=(1, width_lines), mode="nearest")


def auto_window(wl, hp, present, Rref, half_nm=8.0):
    """Window around the wavelength where the high-passed RC varies most across the map
    (only where the lamp is reasonably bright, so the noisy dark end is not chosen)."""
    v = _smooth(np.nanstd(hp[present], 0), 15)
    v[Rref < 0.2 * np.nanmax(Rref)] = 0
    edge = (wl < wl.min() + half_nm / 2) | (wl > wl.max() - half_nm / 2)
    v[edge] = 0
    lam = float(wl[int(np.argmax(v))])
    return [max(float(wl.min()), lam - half_nm), min(float(wl.max()), lam + half_nm)], lam


def featureless_points(amp, present):
    """Points without a resonance: feature size below twice the 10th percentile."""
    a = np.where(present, amp, np.nan)
    a0 = np.nanpercentile(a, 10)
    return present & (a < 2 * a0)


def choose_r0_points(R, Rref, wmask, amp, present):
    """Auto R0 points: featureless, and with the most common reflectance level
    (so a different background region, e.g. another substrate, is not mixed in)."""
    flat = featureless_points(amp, present)
    lvl = np.nanmedian(R[:, wmask] / Rref[wmask], 1)
    med = np.median(lvl[flat])
    mad = 1.4826 * np.median(np.abs(lvl[flat] - med)) or 1e-3
    return flat & (np.abs(lvl - med) < 3 * mad), flat


def get_r0(cfg, wl, R, dark, present, wmask, amp, Rref, exposure, log=print):
    c = cfg["rc"]
    r0 = c["r0"]
    info = {"source": None, "points": np.zeros(len(R), bool), "scale": 1.0}
    flat = featureless_points(amp, present)
    if isinstance(r0, (list, tuple)):
        ks = [int(k) for k in r0]
        bad = [k for k in ks if not (1 <= k <= len(R)) or not present[k - 1]]
        if bad:
            raise SystemExit(f"rc.r0: loc {bad} not in the map or missing")
        info["points"][np.array(ks) - 1] = True
        info["source"] = f"median of {len(ks)} picked points"
        R0 = np.median(R[info["points"]], 0)
    elif r0 in ("auto", None):
        pts, flat = choose_r0_points(R, Rref, wmask, amp, present)
        if pts.sum() < 3:
            raise SystemExit("rc.r0: auto found fewer than 3 featureless points; pick points (r0: [loc, ...]) "
                             "or give an R0 file")
        info["points"] = pts
        info["source"] = f"auto: median of {pts.sum()} featureless points"
        R0 = np.median(R[pts], 0)
    else:
        s0 = background.load_spectrum(r0, wl)
        n = float(c["r0_rows_summed"] or 1)
        e0 = float(c["r0_exposure_s"] or exposure)
        R0 = (s0 - dark) / n / e0          # hardware-binned rows: one offset, signal x n
        info["source"] = f"file {os.path.basename(r0)}"
        mode = c["r0_scale"]
        with np.errstate(all="ignore"):
            k_meas = float(np.nanmedian(R[flat][:, wmask] / R0[wmask]))
        if mode in ("auto", None):
            info["scale"] = k_meas
            log(f"R0 file scaled by {k_meas:.4f} so that RC = 0 on the {flat.sum()} featureless map points "
                f"(exposure ratio alone would give 1)")
        elif mode == "exposure":
            log(f"R0 scaled by exposure only ({e0:g} s file vs {exposure:g} s map); the featureless points "
                f"would need x{k_meas:.4f}")
        else:
            info["scale"] = float(mode)
        R0 = R0 * info["scale"]
        info["points"] = flat
    log(f"R0: {info['source']}")
    return R0, info


def check_r0(wl, R, R0, wm, lit, amp, present, log=print):
    """Warn if R0 itself carries the resonance (e.g. picked points on the flake):
    then every point shows an inverted copy of it. R0 is compared with the median
    of the map's featureless points, whose noise sets the threshold."""
    flat = featureless_points(amp, present)
    if flat.sum() < 3:
        return None
    Rf = np.median(R[flat], 0)
    with np.errstate(all="ignore"):
        q = np.where(lit, R0 / Rf - 1, 0.0)
    f = float(np.ptp(highpass(q[None])[0][wm]))
    thr = 2 * float(np.nanpercentile(np.where(present, amp, np.nan), 10))
    if f > thr:
        log(f"  WARNING: R0 has a resonance-like feature of size {f:.3f} in the window (featureless points: "
            f"< {thr:.3f}). R0 may include points on the flake -- pick points where the report's "
            f"'feature size' map is dark, or use r0: auto")
    return f


# ------------------------------------------------------------------ fitting
def basis(E, E0, G):
    """Re and Im of (G/2) / (E0 - E - i G/2)."""
    d = E0 - E
    h = G / 2
    den = d * d + h * h
    return h * d / den, h * h / den


def model(E, E0, G, a, b, c0, c1, Emid=0.0):
    re, im = basis(E, E0, G)
    return a * re + b * im + c0 + c1 * (E - Emid)


def grid_search(E, Y, e_lo, e_hi, g_lo, g_hi):
    """Best (E0, G) per row of Y by linear least squares on a grid. Returns params [n, 6], rss."""
    Emid = float(E.mean())
    x = E - Emid
    yy = np.sum(Y * Y, 1)
    best = np.full(len(Y), np.inf)
    par = np.zeros((len(Y), 6))
    for E0 in np.arange(e_lo, e_hi + 1e-12, 0.25e-3):
        for G in np.geomspace(g_lo, g_hi, 24):
            re, im = basis(E, E0, G)
            A = np.stack([re, im, np.ones_like(E), x], 1)
            try:
                Ai = np.linalg.inv(A.T @ A)
            except np.linalg.LinAlgError:
                continue
            Q = Y @ A
            coef = Q @ Ai
            rss = yy - np.einsum("ij,ij->i", coef, Q)
            b = rss < best
            best[b] = rss[b]
            par[b, 0], par[b, 1] = E0, G
            par[b, 2:] = coef[b]
    return par, best


def fit_points(E, Y, idx, e_lo, e_hi, g_lo, g_hi, log=print):
    from scipy.optimize import curve_fit
    Emid = float(E.mean())
    par0, _ = grid_search(E, Y[idx], e_lo, e_hi, g_lo, g_hi)
    lo = [e_lo, g_lo, -np.inf, -np.inf, -np.inf, -np.inf]
    hi = [e_hi, g_hi, np.inf, np.inf, np.inf, np.inf]
    f = lambda x, *p: model(x, *p, Emid=Emid)
    out = np.full((len(Y), 6), np.nan)
    cov = np.full((len(Y), 6, 6), np.nan)
    for j, i in enumerate(idx):
        p0 = np.clip(par0[j], lo, hi)
        try:
            p, c = curve_fit(f, E, Y[i], p0=p0, bounds=(lo, hi), method="trf", x_scale="jac", max_nfev=2000)
        except (RuntimeError, ValueError):
            p, c = p0, np.full((6, 6), np.nan)
        out[i], cov[i] = p, c
        if (j + 1) % max(1, len(idx) // 5) == 0:
            log(f"  {j + 1}/{len(idx)}")
    return out, cov, Emid


def derive(E, Y, par, cov, Emid, e_lo, e_hi, g_lo, g_hi, min_snr):
    n = len(Y)
    res = {k: np.full(n, np.nan) for k in OUT}
    res["ok"] = np.zeros(n, bool)
    for i in range(n):
        p = par[i]
        if not np.all(np.isfinite(p)):
            continue
        E0, G, a, b, c0, c1 = p
        r = Y[i] - model(E, *p, Emid=Emid)
        dof = max(len(E) - 6, 1)
        noise = float(np.sqrt(np.sum(r * r) / dof))
        err = np.sqrt(np.clip(np.diag(cov[i]), 0, None)) if np.all(np.isfinite(cov[i])) else np.full(6, np.nan)
        A = float(np.hypot(a, b))
        Aerr = float(np.hypot(a * err[2], b * err[3]) / A) if A > 0 else np.nan
        sst = np.sum((Y[i] - Y[i].mean()) ** 2)
        res["center_meV"][i], res["center_meV_err"][i] = E0 * 1e3, err[0] * 1e3
        res["center_nm"][i] = HC / E0
        res["fwhm_meV"][i], res["fwhm_meV_err"][i] = G * 1e3, err[1] * 1e3
        res["amplitude"][i], res["amplitude_err"][i] = A, Aerr
        res["phase_deg"][i] = np.degrees(np.arctan2(-b, a))
        res["baseline"][i] = c0
        res["noise"][i] = noise
        res["snr"][i] = A / noise if noise > 0 else np.inf
        res["r2"][i] = 1 - np.sum(r * r) / sst if sst > 0 else np.nan
        at_bound = (abs(E0 - e_lo) < 0.3e-3 or abs(E0 - e_hi) < 0.3e-3
                    or np.isclose(G, g_lo, rtol=1e-3) or np.isclose(G, g_hi, rtol=1e-3))
        res["ok"][i] = bool(np.isfinite(err[0]) and res["snr"][i] >= min_snr and not at_bound)
    return res


# ------------------------------------------------------------------ main
def run(cfg, log=print):
    c = cfg["rc"]
    grid = config.grid(cfg)
    log(grid.describe())
    from . import io
    wl = io.load_wavelength(cfg["data"]["wavelength_file"])
    log(f"wavelength axis: {len(wl)} values, {wl.min():.2f}-{wl.max():.2f} nm")
    sp = spectra.build(cfg, grid, wl, log)
    out = cfg["data"]["output"]
    ks = config.selection(cfg["spectra"]["plots"], sorted(k for k in sp["files"] if k <= grid.npts))
    plots.spectrum_plots(os.path.join(out, "spectra_plots"), wl, sp["spectra"], ks, grid, sp["prefix"],
                         sp["files"], cfg["spectra"]["plot_dpi"], log)
    S, present = sp["spectra"], sp["present"]
    exposure = float(cfg["scan"]["exposure_s"] or 1.0)
    if not cfg["scan"]["exposure_s"]:
        log("map exposure unknown (exposure_s in the scan file): R is in counts per exposure")
    dark = dark_level(cfg, wl, S, present, log)
    R = (S - dark) / exposure
    R[~present] = np.nan
    Rref = np.nanmedian(R[present], 0)
    Rs = _smooth(np.nan_to_num(Rref), 31)
    lit = Rs > MIN_LAMP * np.nanmax(Rs)       # where the lamp is too dim, R/R0 is meaningless
    if (~lit).any():
        log(f"RC left blank where the lamp is below {MIN_LAMP:.0%} of its maximum: "
            f"{_ranges_nm(wl, ~lit)}")

    with np.errstate(all="ignore"):
        hp0 = highpass(np.where(lit, R / Rref - 1, 0.0))
    if c["window_nm"]:
        win = [min(c["window_nm"]), max(c["window_nm"])]
        log(f"resonance window {win[0]:.1f}-{win[1]:.1f} nm (rc.window_nm)")
    else:
        win, lam = auto_window(wl, hp0, present, Rref)
        log(f"resonance window {win[0]:.1f}-{win[1]:.1f} nm: RC varies most across the map at {lam:.1f} nm "
            f"(set rc.window_nm to change)")
    wm = (wl >= win[0]) & (wl <= win[1])
    amp0 = np.where(present, np.ptp(hp0[:, wm], 1), np.nan)

    R0, r0info = get_r0(cfg, wl, R, dark, present, wm, amp0, Rref, exposure, log)
    check_r0(wl, R, R0, wm, lit, amp0, present, log)
    with np.errstate(all="ignore"):
        RC = np.where(lit & (R0 > MIN_LAMP * np.nanmax(R0)), R / R0 - 1, np.nan)
        hp = highpass(RC)
    p2p = np.where(present, np.ptp(hp[:, wm], 1), np.nan)
    lvl = np.where(present, np.nanmedian(RC[:, wm], 1), np.nan)
    Rmean = np.where(present, np.nanmean(R, 1), np.nan)
    ref = r0info["points"]
    log(f"RC on the R0/featureless points in the window: median {np.nanmedian(lvl[ref]):+.4f} "
        f"(spread {np.nanpercentile(lvl[ref], 5):+.3f} to {np.nanpercentile(lvl[ref], 95):+.3f}); "
        f"feature size p2p: median {np.nanmedian(p2p):.3f}, max {np.nanmax(p2p):.3f}")

    res = {"feature_p2p": p2p, "rc_level": lvl, "R_mean": Rmean}
    E = HC / wl[wm]
    e_lo, e_hi = float(E.min()), float(E.max())
    g_lo, g_hi = c["min_fwhm_meV"] / 1e3, c["max_fwhm_meV"] / 1e3
    par = np.full((grid.npts, 6), np.nan)
    Emid = float(E.mean())
    if c["fit"]:
        idx = np.flatnonzero(present)
        Y = np.nan_to_num(RC[:, wm])
        log(f"fitting the resonance at {len(idx)} points (complex Lorentzian + linear baseline) ...")
        par, cov, Emid = fit_points(E, Y, idx, e_lo, e_hi, g_lo, g_hi, log)
        res.update(derive(E, Y, par, cov, Emid, e_lo, e_hi, g_lo, g_hi, c["min_snr"]))
        good = res["ok"]
        log(f"  good fits at {good.sum()} of {len(idx)} points (SNR >= {c['min_snr']:g}, not at a bound)")
        if good.any():
            log(f"  E0 {np.median(res['center_meV'][good]):.1f} meV (5-95 %: "
                f"{np.percentile(res['center_meV'][good], 5):.1f}-{np.percentile(res['center_meV'][good], 95):.1f}), "
                f"FWHM {np.median(res['fwhm_meV'][good]):.2f} meV, amplitude up to {np.max(res['amplitude'][good]):.3f}")

    fdir = os.path.join(out, "rc")
    os.makedirs(fdir, exist_ok=True)
    write_results(fdir, cfg, grid, wl, present, res, par, R0, r0info, win, Emid, sp["prefix"], log)
    summary(os.path.join(fdir, "summary"), wl, grid, RC, R, R0, res, r0info, win, par, Emid, cfg)
    log(f"summary figure: {os.path.join(fdir, 'summary.png')}")
    kp = config.selection(c["plots"], [k for k in range(1, grid.npts + 1) if present[k - 1]])
    point_plots(os.path.join(fdir, "rc_plots"), wl, RC, wm, par, Emid, res, kp, sp["prefix"],
                cfg["spectra"]["plot_dpi"], log)
    from . import rc_report
    rc_report.write(os.path.join(fdir, "report.html"), cfg, grid, wl, R, R0, present, res, par, Emid, win,
                    r0info, sp["prefix"], log)
    log(f"\ndone. Outputs in {fdir}")
    return {"RC": RC, "R": R, "R0": R0, "res": res, "par": par, "window": win, "r0": r0info}


# ------------------------------------------------------------------ outputs
def write_results(fdir, cfg, grid, wl, present, res, par, R0, r0info, win, Emid, prefix, log=print):
    import csv
    import yaml
    ks = np.arange(1, grid.npts + 1)
    rows, cols = grid.rowcol(ks)
    sx, sy = grid.stage_xy(ks)
    xr, yr = grid.x_rel_um(), grid.y_rel_um()
    keys = ["R_mean", "rc_level", "feature_p2p"] + [k for k in OUT if k in res] + (["ok"] if "ok" in res else [])
    u = grid.units
    with open(os.path.join(fdir, "results.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "loc", "row", "col", f"x_{u}", f"y_{u}", "x_rel_um", "y_rel_um", "present", "r0_point"]
                   + keys)
        for i, k in enumerate(ks):
            line = [f"{prefix}_loc{k}", k, rows[i], cols[i], f"{sx[i]:.6f}", f"{sy[i]:.6f}", f"{xr[cols[i]]:.4f}",
                    f"{yr[rows[i]]:.4f}", int(present[i]), int(r0info["points"][i])]
            line += [int(res[q][i]) if q == "ok" else ("" if not np.isfinite(res[q][i]) else f"{res[q][i]:.6g}")
                     for q in keys]
            w.writerow(line)
    np.savez_compressed(os.path.join(fdir, "results.npz"), wl=wl, R0=R0, params=par, param_names=np.array(PARAMS),
                        Emid=Emid, window_nm=np.array(win), present=present, r0_points=r0info["points"],
                        nx=grid.nx, ny=grid.ny, x_rel_um=xr, y_rel_um=yr, stage_x=sx, stage_y=sy, **res)
    info = {"written": datetime.datetime.now().isoformat(timespec="seconds"),
            "folder": cfg["data"]["folder"], "scan": {k: v for k, v in cfg["scan"].items() if v is not None},
            "rc": {k: (v if not isinstance(v, tuple) else list(v)) for k, v in cfg["rc"].items()},
            "window_nm": [float(win[0]), float(win[1])], "r0_source": r0info["source"],
            "r0_scale": float(r0info["scale"]), "r0_points": [int(k) for k in np.flatnonzero(r0info["points"]) + 1],
            "good_fits": int(res["ok"].sum()) if "ok" in res else None}
    with open(os.path.join(fdir, "rc_info.yaml"), "w") as fh:
        yaml.safe_dump(info, fh, sort_keys=False)
    log(f"results: {os.path.join(fdir, 'results.csv')}")


def summary(path_base, wl, grid, RC, R, R0, res, r0info, win, par, Emid, cfg):
    plt = plots._mpl()
    from matplotlib.gridspec import GridSpec
    flip = cfg["scan"]["flip_x"]
    nx, ny = grid.nx, grid.ny
    ext = plots._extent(grid)
    ok = res.get("ok", np.ones(len(RC), bool))
    maps = [("R_mean", "gray", None), ("feature_p2p", "magma", None)]
    if "amplitude" in res:
        maps += [("amplitude", "magma", ok), ("center_meV", "coolwarm", ok), ("fwhm_meV", "viridis", ok),
                 ("phase_deg", "twilight", ok)]
    fig = plt.figure(figsize=(3.0 * len(maps), 7.4))
    gs = GridSpec(2, len(maps), figure=fig, height_ratios=[1.15, 1], hspace=0.38, wspace=0.45,
                  left=0.045, right=0.985, top=0.9, bottom=0.08)
    rr, cc = np.divmod(np.flatnonzero(r0info["points"]), nx)
    xs = grid.x_rel_um()
    ys = grid.y_rel_um()
    for j, (key, cmap, mask) in enumerate(maps):
        ax = fig.add_subplot(gs[0, j])
        v = res[key].copy()
        if mask is not None:
            v[~mask] = np.nan
        M = v.reshape(ny, nx)
        fin = M[np.isfinite(M)]
        lo, hi = (np.percentile(fin, 1), np.percentile(fin, 99.5)) if fin.size else (0, 1)
        ax.set_facecolor("0.85")
        im = ax.imshow(M, cmap=cmap, extent=ext, vmin=lo, vmax=hi, interpolation="nearest")
        if key == "R_mean" and len(rr):
            ax.plot(xs[cc], ys[rr], ".", ms=2, color="#2ca02c", alpha=0.7)
        if flip:
            ax.invert_xaxis()
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(labelsize=7)
        ax.set_title(LABELS[key] + ("\n(green: R0 points)" if key == "R_mean" and len(rr) else ""), fontsize=8)
        ax.set_xlabel("x (µm)" + (" ← flipped" if flip else ""), fontsize=7)
        ax.tick_params(labelsize=7)
    E = HC / wl
    wm = (wl >= win[0]) & (wl <= win[1])
    ax = fig.add_subplot(gs[1, : len(maps) // 2 + 1])
    score = np.where(ok & np.isfinite(res.get("amplitude", res["feature_p2p"])),
                     res.get("amplitude", res["feature_p2p"]), -np.inf)
    best = int(np.argmax(score))
    ref = np.flatnonzero(r0info["points"])
    pts = [(best, "#d62728", "strongest resonance")]
    if len(ref):
        pts.append((int(ref[len(ref) // 2]), "#b8860b", "an R0 point"))
    for i, col, lab in pts:
        r, cidx = divmod(i, nx)
        ax.plot(E, RC[i], color=col, lw=0.5, alpha=0.4)
        ax.plot(E, np.where(np.isfinite(RC[i]), _smooth(np.nan_to_num(RC[i]), 7), np.nan), color=col, lw=1.2,
                label=f"{lab}: loc{i + 1} ({xs[cidx]:.1f}, {ys[r]:.1f}) µm")
        if np.all(np.isfinite(par[i])):
            ax.plot(E[wm], model(E[wm], *par[i], Emid=Emid), color="k", lw=1, ls="--")
    ax.axvspan(HC / win[1], HC / win[0], color="0.92", zorder=0)
    ax.axhline(0, color="0.5", lw=0.5, ls=":")
    ax.set_xlim(E.min(), E.max())
    lo, hi = np.nanpercentile(RC[[p[0] for p in pts]], [0.5, 99.5])
    if np.isfinite(lo) and np.isfinite(hi):
        ax.set_ylim(lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo))
    ax.set_xlabel("Energy (eV)", fontsize=8)
    ax.set_ylabel("RC = (R − R0)/R0", fontsize=8)
    ax.legend(fontsize=7, frameon=False)
    ax.tick_params(labelsize=7)
    ax2 = fig.add_subplot(gs[1, len(maps) // 2 + 1:])
    ax2.plot(wl, R0, color="k", lw=1, label=f"R0 ({r0info['source']})")
    ax2.plot(wl, R[best], color="#d62728", lw=0.8, label=f"R at loc{best + 1}")
    ax2.axvspan(win[0], win[1], color="0.92", zorder=0)
    ax2.set_xlim(wl.min(), wl.max())
    ax2.set_xlabel("Wavelength (nm)", fontsize=8)
    ax2.set_ylabel("R (counts/s per CCD row)", fontsize=8)
    ax2.legend(fontsize=7, frameon=False)
    ax2.tick_params(labelsize=7)
    title = cfg["report"]["title"] or os.path.basename(os.path.normpath(cfg["data"]["folder"]))
    fig.suptitle(f"{title} — reflection contrast, {nx}×{ny} points, window {win[0]:.1f}–{win[1]:.1f} nm", fontsize=11)
    fig.text(0.5, 0.005, "Grey = no data or fit rejected. Dashed: fit (complex Lorentzian + linear baseline). "
             "Shaded: resonance window.", ha="center", fontsize=8, color="0.35")
    fig.savefig(path_base + ".png", dpi=150)
    fig.savefig(path_base + ".pdf")
    plt.close(fig)


def point_plots(outdir, wl, RC, wm, par, Emid, res, ks, prefix, dpi=100, log=print):
    if not len(ks):
        return
    plt = plots._mpl()
    os.makedirs(outdir, exist_ok=True)
    E = HC / wl
    for k in ks:
        i = k - 1
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        ax.plot(E, RC[i], color="0.55", lw=0.6, label="RC")
        if np.all(np.isfinite(par[i])):
            ok = bool(res["ok"][i]) if "ok" in res else True
            ax.plot(E[wm], model(E[wm], *par[i], Emid=Emid), color="#d62728" if ok else "#999", lw=1.3,
                    label=f"fit: E0 {res['center_meV'][i]:.1f} meV, Γ {res['fwhm_meV'][i]:.1f} meV, "
                          f"A {res['amplitude'][i]:.3f}" + ("" if ok else " (rejected)"))
        ax.set_xlabel("Energy (eV)", fontsize=9)
        ax.set_ylabel("RC", fontsize=9)
        ax.set_title(f"{prefix}_loc{k}", fontsize=9)
        ax.legend(fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"{prefix}_loc{k}_rc.png"), dpi=dpi)
        plt.close(fig)
    log(f"RC plots: {len(ks)} written -> {outdir}")
