"""Step 3b: multi-peak fits of every pixel.

Fits are done on the energy axis (E = hc/lambda). Every peak has its own
lineshape (Lorentzian by default; change `shape` in the config to gaussian or
voigt and re-run). A constant or linear baseline is fitted together with the
peaks.

Workflow
  1. peaks come from the config, or are detected on the mean spectrum of the
     brightest points (auto_detect)
  2. the mean bright spectrum is fitted first; its parameters seed every pixel
  3. for that mean spectrum each peak is also re-fitted as L / G / V to help
     choose lineshapes (table printed and shown in the HTML)
  4. every pixel is fitted; derived quantities and quality flags are computed
"""
import os

import numpy as np

from . import HC
from . import lineshapes as ls

NBASE = {"none": 0, "constant": 1, "linear": 2}
DERIVED = ("center_nm", "center_meV", "height", "intensity", "fwhm_meV", "fwhm_nm", "snr")
DERIVED_ERR = ("center_nm_err", "center_meV_err", "intensity_err", "fwhm_meV_err")
LABELS = {
    "center_nm": "peak wavelength (nm)", "center_meV": "peak energy (meV)",
    "height": "peak height (counts)", "intensity": "integrated intensity (counts)",
    "fwhm_meV": "FWHM (meV)", "fwhm_nm": "FWHM (nm)", "snr": "SNR (height / noise)",
}


# ------------------------------------------------------------------ model
class Model:
    """Sum of peaks + polynomial baseline on an energy axis."""

    def __init__(self, peaks, E, baseline="linear"):
        self.peaks = peaks
        self.E = E
        self.Emid = float(np.mean(E))
        self.nbase = NBASE[baseline]
        self.baseline = baseline
        self.slices = []
        i = 0
        for p in peaks:
            n = ls.nparams(p["shape"])
            self.slices.append(slice(i, i + n))
            i += n
        self.npk = i
        self.n = i + self.nbase

    def components(self, x, theta):
        return [ls.evaluate(p["shape"], x, theta[s]) for p, s in zip(self.peaks, self.slices)]

    def base(self, x, theta):
        b = theta[self.npk:]
        out = np.zeros_like(x, dtype=float)
        for j, c in enumerate(b):
            out = out + c * (x - self.Emid) ** j
        return out

    def __call__(self, x, *theta):
        theta = np.asarray(theta)
        y = self.base(x, theta)
        for c in self.components(x, theta):
            y = y + c
        return y

    def bounds(self):
        lo, hi = [], []
        for p in self.peaks:
            e_lo, e_hi = HC / (p["center_nm"] + p["window_nm"]), HC / (p["center_nm"] - p["window_nm"])
            w_lo, w_hi = p["min_fwhm_meV"] / 1e3, p["max_fwhm_meV"] / 1e3
            lo += [e_lo, 0.0] + [w_lo] * ls.NWIDTH[p["shape"]]
            hi += [e_hi, np.inf] + [w_hi] * ls.NWIDTH[p["shape"]]
        lo += [-np.inf] * self.nbase
        hi += [np.inf] * self.nbase
        return np.array(lo), np.array(hi)

    def initial(self, y, centers=None, fwhms=None):
        """Initial guess: given centres/widths, solve linearly for areas + baseline."""
        widths = []
        for i, p in enumerate(self.peaks):
            e0 = HC / p["center_nm"] if centers is None else centers[i]
            f = p["fwhm_meV"] / 1e3 if fwhms is None else fwhms[i]
            if p["shape"] == "voigt":
                widths.append((e0, [0.6 * f, 0.6 * f]))
            else:
                widths.append((e0, [f]))
        cols = [ls.evaluate(p["shape"], self.E, np.r_[e0, 1.0, w]) for p, (e0, w) in zip(self.peaks, widths)]
        cols += [(self.E - self.Emid) ** j for j in range(self.nbase)]
        A = np.stack(cols, 1)
        coef = np.linalg.lstsq(A, y, rcond=None)[0]
        theta = []
        for i, (e0, w) in enumerate(widths):
            theta += [e0, max(coef[i], 1e-9)] + list(w)
        theta += list(coef[len(self.peaks):])
        lo, hi = self.bounds()
        return np.clip(np.array(theta, float), lo, hi)

    def peak_params(self, theta, i):
        return theta[self.slices[i]]


def fit_model(model, y, theta0, maxfev=4000):
    from scipy.optimize import curve_fit
    lo, hi = model.bounds()
    theta0 = np.clip(theta0, lo, hi)
    try:
        theta, cov = curve_fit(model, model.E, y, p0=theta0, bounds=(lo, hi), method="trf",
                               x_scale="jac", max_nfev=maxfev)
    except (RuntimeError, ValueError):
        return None, None
    return theta, cov


# ------------------------------------------------------------ derived values
def derive(model, theta, cov, y, wl_step_at, min_snr):
    """Per-peak physical quantities + errors + quality flag, plus fit stats."""
    resid = y - model(model.E, *theta)
    dof = max(len(y) - model.n, 1)
    noise = float(np.sqrt(np.sum(resid ** 2) / dof))
    err = np.sqrt(np.clip(np.diag(cov), 0, None)) if cov is not None and np.all(np.isfinite(cov)) \
        else np.full(model.n, np.nan)
    lo, hi = model.bounds()
    out = []
    for i, p in enumerate(model.peaks):
        s = model.slices[i]
        th, er = theta[s], err[s]
        e0, area = th[0], th[1]
        f = float(ls.fwhm(p["shape"], th))
        h = float(ls.height(p["shape"], th))
        de_line = HC / (HC / e0) ** 2 * wl_step_at(HC / e0)        # eV per spectral line at the peak
        if p["shape"] == "voigt":   # numerical error propagation for the Olivero FWHM
            g = np.array([(ls.voigt_fwhm(th[2] + 1e-6, th[3]) - f) / 1e-6,
                          (ls.voigt_fwhm(th[2], th[3] + 1e-6) - f) / 1e-6])
            ferr = float(np.sqrt(np.sum((g * er[2:4]) ** 2)))
        else:
            ferr = er[2]
        at_bound = (np.isclose(th[0], lo[s][0], rtol=0, atol=2e-5) or np.isclose(th[0], hi[s][0], rtol=0, atol=2e-5)
                    or np.any(np.isclose(th[2:], lo[s][2:], rtol=1e-3)) or np.any(np.isclose(th[2:], hi[s][2:], rtol=1e-3)))
        snr = h / noise if noise > 0 else np.inf
        out.append({
            "center_nm": HC / e0, "center_meV": e0 * 1e3, "height": h, "intensity": area / de_line,
            "fwhm_meV": f * 1e3, "fwhm_nm": HC / (e0 - f / 2) - HC / (e0 + f / 2), "snr": snr,
            "center_nm_err": HC / e0 ** 2 * er[0], "center_meV_err": er[0] * 1e3,
            "intensity_err": er[1] / de_line, "fwhm_meV_err": ferr * 1e3,
            "ok": bool(np.isfinite(er[0]) and snr >= min_snr and not at_bound and area > 0),
        })
    ss_tot = np.sum((y - y.mean()) ** 2)
    stats = {"noise": noise, "r2": 1 - np.sum(resid ** 2) / ss_tot if ss_tot > 0 else np.nan}
    return out, stats


# ----------------------------------------------------------- peak detection
def detect_peaks(wl, y, max_peaks=4, log=print):
    """Find peaks in a (mean) spectrum. Returns list of peak dicts sorted by wavelength."""
    from scipy.signal import find_peaks, peak_widths
    from .spectra import running_median
    n = len(y)
    k = max(3, n // 200)
    ys = np.convolve(np.pad(y, k, mode="edge"), np.ones(2 * k + 1) / (2 * k + 1), "same")[k:-k]
    noise = 1.4826 * np.median(np.abs(np.diff(y))) / np.sqrt(2) / np.sqrt(2 * k + 1)
    base = running_median(ys, max(5, n // 8))
    idx, prop = find_peaks(ys - np.minimum(base, ys), prominence=8 * max(noise, 1e-9), distance=3 * k)
    if len(idx) == 0:
        log("auto-detect: no peaks found in the mean bright spectrum")
        return []
    order = np.argsort(prop["prominences"])[::-1][:max_peaks]
    idx = np.sort(idx[order])
    w = peak_widths(ys, idx, rel_height=0.5)[0]
    peaks = []
    for j, (i, wi) in enumerate(zip(idx, w)):
        dl = abs(np.interp(i + wi / 2, np.arange(n), wl) - np.interp(i - wi / 2, np.arange(n), wl))
        fwhm_meV = HC / wl[i] ** 2 * dl * 1e3
        peaks.append({"name": f"P{j + 1}", "center_nm": round(float(wl[i]), 2), "shape": "lorentzian",
                      "window_nm": round(float(max(2.0, 1.5 * dl)), 1),
                      "fwhm_meV": round(float(np.clip(fwhm_meV, 1.0, 60.0)), 1),
                      "min_fwhm_meV": 0.5, "max_fwhm_meV": round(float(max(80.0, 3 * fwhm_meV)), 0)})
    return peaks


# -------------------------------------------------------------- per pixel
_W = {}


def _init_worker(model, E_mask_y, theta_seed, min_snr, wl):
    _W.update(model=model, Y=E_mask_y, seed=theta_seed, min_snr=min_snr, wl=wl,
              step=np.abs(np.gradient(wl)))


def _wl_step_at(lam):
    return float(np.interp(lam, _W["wl"], _W["step"]))


def _fit_index(i):
    m, y = _W["model"], _W["Y"][i]
    seed = _W["seed"]
    cen = [seed[s][0] for s in m.slices]
    fw = [ls.fwhm(p["shape"], seed[s]) for p, s in zip(m.peaks, m.slices)]
    theta0 = m.initial(y, cen, fw)
    theta, cov = fit_model(m, y, theta0)
    if theta is None:
        return i, None, None, None
    d, st = derive(m, theta, cov, y, _wl_step_at, _W["min_snr"])
    return i, theta, d, st


def run(cfg, wl, spc, present, log=print):
    """Fit all present pixels of background-subtracted spectra `spc` [npts, nwl]."""
    fc = cfg["fit"]
    rng = fc["range_nm"] or [wl.min(), wl.max()]
    fm = (wl >= min(rng)) & (wl <= max(rng))
    E = HC / wl[fm]
    Y = np.nan_to_num(spc[:, fm])

    # mean spectrum of the brightest points: detection + global seed
    tot = np.where(present, np.nansum(spc[:, fm], 1), -np.inf)
    nb = max(3, int(0.1 * present.sum()))
    bright = np.argsort(tot)[::-1][:nb]
    ymean = Y[bright].mean(0)

    peaks = [dict(p) for p in fc["peaks"]]
    if not peaks:
        if not fc["auto_detect"]:
            raise SystemExit("no peaks in fit.peaks and auto_detect is off")
        peaks = detect_peaks(wl[fm], ymean, fc["max_peaks"], log)
        if not peaks:
            raise SystemExit("no peaks to fit")
        log("auto-detected peaks (copy into fit.peaks to edit names/shapes):")
        for p in peaks:
            log(f"  - {{name: {p['name']}, center_nm: {p['center_nm']}, shape: lorentzian, "
                f"window_nm: {p['window_nm']}, fwhm_meV: {p['fwhm_meV']}}}")
    names = [p["name"] for p in peaks]
    if len(set(names)) != len(names):
        raise SystemExit(f"peak names must be unique: {names}")

    model = Model(peaks, E, fc["baseline"])
    _init_worker(model, Y, None, fc["min_snr"], wl)
    theta_g, cov_g = fit_model(model, ymean, model.initial(ymean), maxfev=20000)
    if theta_g is None:
        raise SystemExit("fit of the mean bright spectrum failed -- check fit.peaks / fit.range_nm")
    d_g, st_g = derive(model, theta_g, cov_g, ymean, _wl_step_at, 0)
    log(f"mean of {nb} brightest points: R^2 = {st_g['r2']:.4f}")
    for p, d in zip(peaks, d_g):
        log(f"  {p['name']:>6} ({p['shape']:<10}): {d['center_nm']:.3f} nm = {d['center_meV']:.1f} meV, "
            f"FWHM {d['fwhm_meV']:.2f} meV, height {d['height']:.1f}")

    comparison = compare_shapes(peaks, E, Y[bright[:5]], theta_g, fc["baseline"], log)

    # per-pixel fits
    _init_worker(model, Y, theta_g, fc["min_snr"], wl)
    todo = np.flatnonzero(present)
    npts = len(spc)
    theta_all = np.full((npts, model.n), np.nan)
    res = {f"{p['name']}_{q}": np.full(npts, np.nan) for p in peaks for q in DERIVED + DERIVED_ERR}
    for p in peaks:
        res[f"{p['name']}_ok"] = np.zeros(npts, bool)
    res["r2"] = np.full(npts, np.nan)
    res["noise"] = np.full(npts, np.nan)
    res["fit_failed"] = np.zeros(npts, bool)

    desc = ", ".join("%s=%s" % (p["name"], p["shape"]) for p in peaks)
    log(f"fitting {len(todo)} points ({desc}; baseline {fc['baseline']}) ...")
    workers = fc["workers"]
    workers = min(os.cpu_count() or 1, 8) if workers in ("auto", None) else int(workers)
    workers = max(1, min(workers, len(todo) // 20 or 1))
    if workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork" if hasattr(os, "fork") else "spawn").Pool(
                workers, _init_worker, (model, Y, theta_g, fc["min_snr"], wl)) as pool:
            results = pool.imap_unordered(_fit_index, todo, chunksize=16)
            results = list(_progress(results, len(todo), log))
    else:
        results = list(_progress(map(_fit_index, todo), len(todo), log))
    for i, theta, d, st in results:
        if theta is None:
            res["fit_failed"][i] = True
            continue
        theta_all[i] = theta
        for p, dp in zip(peaks, d):
            for q in DERIVED + DERIVED_ERR:
                res[f"{p['name']}_{q}"][i] = dp[q]
            res[f"{p['name']}_ok"][i] = dp["ok"]
        res["r2"][i] = st["r2"]
        res["noise"][i] = st["noise"]
    res["integrated_counts"] = np.where(present, np.nansum(spc[:, fm], 1), np.nan)
    for p in peaks:
        log(f"  {p['name']}: good fits at {res[p['name'] + '_ok'].sum()} of {len(todo)} points")
    if res["fit_failed"].any():
        log(f"  {res['fit_failed'].sum()} fits failed to converge")
    return {"peaks": peaks, "model": model, "theta": theta_all, "res": res, "fit_mask": fm,
            "mean": {"y": ymean, "theta": theta_g, "derived": d_g, "stats": st_g, "points": bright},
            "comparison": comparison}


def _progress(it, n, log):
    step = max(1, n // 10)
    for j, r in enumerate(it, 1):
        if j % step == 0 or j == n:
            log(f"  {j}/{n}")
        yield r


VOIGT_GAIN = 0.02


def recommend(rel):
    """Simplest adequate shape: Voigt has one more parameter, so it always fits at
    least as well; prefer it only if it is clearly better than both L and G."""
    l, g, v = rel["lorentzian"], rel["gaussian"], rel["voigt"]
    simple = "lorentzian" if (l <= g or not np.isfinite(g)) else "gaussian"
    if np.isfinite(v) and v < (1 - VOIGT_GAIN) * min(l, g):
        return "voigt"
    return simple


def compare_shapes(peaks, E, Ys, seed, baseline, log=print):
    """For each peak, refit the brightest individual spectra `Ys` with that peak
    as L, G and V (other peaks unchanged). Individual points are used rather
    than their average because averaging points with different peak energies
    broadens peaks inhomogeneously and biases the test towards Voigt.
    Returns {peak: {shape: summed chi^2 / best, "recommended": shape}}."""
    base = Model(peaks, E, baseline)
    cen = [seed[s][0] for s in base.slices]
    fw = [ls.fwhm(p["shape"], seed[s]) for p, s in zip(peaks, base.slices)]
    out, rows = {}, []
    for i, p in enumerate(peaks):
        chi = {}
        for shape in ls.SHAPES:
            trial = [dict(q) for q in peaks]
            trial[i]["shape"] = shape
            m = Model(trial, E, baseline)
            tot = 0.0
            for y in Ys:
                th, _ = fit_model(m, y, m.initial(y, cen, fw), maxfev=3000)
                if th is None:
                    tot = np.nan
                    break
                tot += float(np.sum((y - m(E, *th)) ** 2) / max(len(y) - m.n, 1))
            chi[shape] = tot
        best = np.nanmin(list(chi.values()))
        out[p["name"]] = {s: v / best for s, v in chi.items()}
        pick = recommend(out[p["name"]])
        out[p["name"]]["recommended"] = pick
        rows.append(f"  {p['name']:>6}: " + "  ".join(f"{s[:4]} {v / best:6.3f}" for s, v in chi.items())
                    + f"   -> recommended: {pick}")
    log(f"lineshape check on the {len(Ys)} brightest points (summed reduced chi^2 / best; 1.000 = best;\n"
        f"  Voigt is recommended only if it beats both Lorentzian and Gaussian by > {VOIGT_GAIN:.0%}):")
    for r in rows:
        log(r)
    return out
