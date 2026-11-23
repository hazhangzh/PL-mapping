"""Step 3b: multi-peak fits of every pixel.

Fits are done on the energy axis (E = hc/lambda). Every peak has its own
lineshape (lorentzian | gaussian | voigt | trion, see lineshapes.py). A
polynomial baseline is fitted together with the peaks.

Peak positions can be given in two ways:
  * absolute: `center_nm` for every peak (the old behaviour);
  * relative: `offset_meV` for every peak, measured from an anchor peak (the
    neutral exciton X0, offset 0). Offsets between excitonic species are set by
    binding energies and change much less than absolute energies, which move
    with strain, dielectric environment, temperature and sample.

Workflow
  1. anchor: the whole peak pattern (all peaks at their offsets) is slid along
     the mean spectrum of the brightest points, solving for non-negative
     amplitudes at each position (`shift_scan`); the best match gives the
     anchor energy. This works even when the anchor peak itself is weak.
  2. the brightest points are each aligned to the pattern (same scan, within
     +-max_shift_meV), averaged, and the aligned mean is fitted. Its parameters
     are the template and seed for every pixel. Aligning first keeps point-to-
     point shifts from broadening the template peaks.
  3. every pixel: the template is slid again to find that pixel's rigid shift;
     the seeds, the centre windows and (for `range_meV`) the fit range move with
     it; then a full least-squares fit.
  4. for the brightest points each peak is also re-fitted with every lineshape
     to help choose shapes (table printed and shown in the HTML).
"""
import os
import re

import numpy as np

from . import HC
from . import lineshapes as ls

NBASE = {"none": 0, "constant": 1, "linear": 2, "quadratic": 3}
DERIVED = ("center_nm", "center_meV", "height", "intensity", "fwhm_meV", "fwhm_nm", "snr", "tail_meV")
DERIVED_ERR = ("center_nm_err", "center_meV_err", "intensity_err", "fwhm_meV_err", "tail_meV_err")
LABELS = {
    "center_nm": "peak wavelength (nm)", "center_meV": "peak energy (meV)",
    "height": "peak height (counts)", "intensity": "integrated intensity (counts)",
    "fwhm_meV": "FWHM (meV)", "fwhm_nm": "FWHM (nm)", "snr": "SNR (height / noise)",
    "tail_meV": "recoil tail length (meV)",
}
COMBINED_KINDS = ("ratio", "difference")


def combined_spec(entries, peaks):
    """Normalise fit.combined: maps built from the per-peak results of every point.

    ratio:      [A, B] -> mean of quantity over peaks A / mean over peaks B   (default quantity: height)
    difference: [A, B] -> mean over A - mean over B                          (default quantity: center_meV)
    A and B are a peak name or a list of names.
    """
    names = [p["name"] for p in peaks]
    reserved = {f"{n}_{q}" for n in names for q in DERIVED + DERIVED_ERR + ("ok",)}
    reserved |= {"integrated_counts", "r2", "noise", "shift_meV", "shift_signif", "range_lo_nm", "range_hi_nm",
                 "fit_failed", "no_signal"}
    out = []
    for j, e in enumerate(entries or []):
        if not isinstance(e, dict):
            raise SystemExit(f"fit.combined[{j}]: expected a mapping like {{name: ..., ratio: [[XT1, XT2], X0]}}")
        unknown = set(e) - {"name", "label", "quantity", *COMBINED_KINDS}
        kind = [k for k in COMBINED_KINDS if k in e]
        if unknown or len(kind) != 1:
            raise SystemExit(f"fit.combined[{j}]: give exactly one of {COMBINED_KINDS} (+ optional name, label, "
                             f"quantity){'; unknown keys ' + str(sorted(unknown)) if unknown else ''}")
        kind = kind[0]
        sides = e[kind]
        if not isinstance(sides, (list, tuple)) or len(sides) != 2:
            raise SystemExit(f"fit.combined[{j}]: {kind} needs two sides, e.g. {kind}: [[XT1, XT2], X0]")
        a, b = ([str(s)] if isinstance(s, str) else [str(x) for x in s] for s in sides)
        miss = [n for n in a + b if n not in names]
        if miss or not a or not b:
            raise SystemExit(f"fit.combined[{j}]: unknown peak(s) {miss}; peaks are {names}")
        q = e.get("quantity") or ("height" if kind == "ratio" else "center_meV")
        if q not in DERIVED:
            raise SystemExit(f"fit.combined[{j}]: quantity must be one of {DERIVED}, got {q!r}")
        name = str(e.get("name") or f"{'_'.join(a)}_{'over' if kind == 'ratio' else 'minus'}_{'_'.join(b)}_{q}")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name in reserved or name in [c["name"] for c in out]:
            raise SystemExit(f"fit.combined[{j}]: name {name!r} is invalid or already used")
        side = lambda s: s[0] if len(s) == 1 else "mean(" + ", ".join(s) + ")"
        label = e.get("label") or (f"{side(a)} / {side(b)}  ({q})" if kind == "ratio" else
                                   f"{side(a)} \u2212 {side(b)}  ({LABELS[q]})")
        out.append({"name": name, "kind": kind, "a": a, "b": b, "quantity": q, "label": str(label)})
    return out


def combine(c, val, ok):
    """One fit.combined entry from val(peak, quantity) and ok(peak) (arrays or scalars) -> (value, good).

    good = every peak involved passed the quality check (and, for a ratio, the denominator is > 0)."""
    A = np.mean([np.asarray(val(n, c["quantity"]), float) for n in c["a"]], 0)
    B = np.mean([np.asarray(val(n, c["quantity"]), float) for n in c["b"]], 0)
    good = np.all([np.asarray(ok(n), bool) for n in c["a"] + c["b"]], 0)
    if c["kind"] == "ratio":
        with np.errstate(divide="ignore", invalid="ignore"):
            v = np.where(B > 0, A / np.where(B > 0, B, 1.0), np.nan)
    else:
        v = A - B
    return v, good & np.isfinite(v)


SHIFT_COARSE = 1e-3         # eV, coarse grid of the shift scan (peaks are >= ~3 meV wide)
SHIFT_STEP = 2e-4           # eV, fine grid around the coarse best (then refined by a parabola)
SHIFT_MIN_DCHI2 = 500.0     # the pattern must explain this much chi^2 (in noise units) to trust a shift;
                            #   pure-noise points reach ~100-350 on a 41x41 WSe2 map, bright points 1e5-1e6


# ------------------------------------------------------------------ peaks
def prepare_peaks(peaks, anchor_E):
    """Fill in E0 (eV) and window_eV for every peak. Relative peaks are placed at
    anchor_E + offset; center_nm is (re)set so the outputs stay in nm too."""
    out = []
    for p in peaks:
        q = dict(p)
        if q.get("offset_meV") is not None:
            q["E0"] = anchor_E + q["offset_meV"] / 1e3
        else:
            q["E0"] = HC / q["center_nm"]
        q["center_nm"] = HC / q["E0"]
        if q.get("window_meV") is not None:
            q["window_eV"] = q["window_meV"] / 1e3
        else:
            q["window_eV"] = HC / (q["center_nm"] - q["window_nm"]) - q["E0"]
            q["window_meV"] = q["window_eV"] * 1e3
        out.append(q)
    return out


def default_widths(p, f=None):
    """Width parameters (eV) of a peak from a total FWHM f (eV; default the config value)."""
    f = p["fwhm_meV"] / 1e3 if f is None else f
    if p["shape"] == "voigt":
        return [0.6 * f, 0.6 * f]
    if p["shape"] == "trion":
        t = p.get("tail_meV", 1.0) / 1e3
        return [max(f - 0.7 * t, 0.5 * f), t]
    return [f]


def width_bounds(p):
    w_lo, w_hi = p["min_fwhm_meV"] / 1e3, p["max_fwhm_meV"] / 1e3
    if p["shape"] == "trion":
        return [w_lo, p.get("min_tail_meV", 0.05) / 1e3], [w_hi, p.get("max_tail_meV", 8.0) / 1e3]
    n = ls.NWIDTH[p["shape"]]
    return [w_lo] * n, [w_hi] * n


# ------------------------------------------------------------------ model
class Model:
    """Sum of peaks + polynomial baseline on an energy axis.

    `shift` (eV) moves every peak's allowed centre window (the pixel's rigid
    shift). `Emid` is the baseline reference energy; it is kept the same for
    every pixel so theta vectors are comparable and the HTML can evaluate them."""

    def __init__(self, peaks, E, baseline="linear", shift=0.0, Emid=None):
        self.peaks = peaks
        self.E = E
        self.shift = float(shift)
        self.Emid = float(np.mean(E)) if Emid is None else float(Emid)
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
            e0 = p["E0"] + self.shift
            wl_, wh_ = width_bounds(p)
            lo += [e0 - p["window_eV"], 0.0] + wl_
            hi += [e0 + p["window_eV"], np.inf] + wh_
        lo += [-np.inf] * self.nbase
        hi += [np.inf] * self.nbase
        return np.array(lo), np.array(hi)

    def initial(self, y, centers=None, widths=None):
        """Initial guess: given centres and width parameters, solve linearly for areas + baseline."""
        cw = []
        for i, p in enumerate(self.peaks):
            e0 = p["E0"] + self.shift if centers is None else centers[i]
            w = default_widths(p) if widths is None or widths[i] is None else list(widths[i])
            cw.append((e0, w))
        cols = [ls.evaluate(p["shape"], self.E, np.r_[e0, 1.0, w]) for p, (e0, w) in zip(self.peaks, cw)]
        cols += [(self.E - self.Emid) ** j for j in range(self.nbase)]
        A = np.stack(cols, 1)
        coef = np.linalg.lstsq(A, y, rcond=None)[0]
        theta = []
        for i, (e0, w) in enumerate(cw):
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


# ------------------------------------------------------- rigid-shift search
def _noise_var(y):
    d = np.diff(y)
    return max((1.4826 * np.median(np.abs(d - np.median(d)))) ** 2 / 2.0, 1e-12)


def shift_scan(E, y, peaks, centers, widths, region, shifts, nbase, Emid):
    """Slide a peak pattern (peaks at `centers`, fixed `widths`) by each of `shifts` (eV)
    and solve for non-negative peak areas + a free polynomial baseline over the FIXED
    energy `region` (so every trial is judged on the same data points).

    Returns (best shift, significance), where significance = chi^2 explained by the
    peaks at the best shift, in units of the noise variance. Low significance means
    there is no clear pattern (off-flake point) and the shift should not be trusted."""
    from scipy.optimize import nnls
    m = (E >= region[0]) & (E <= region[1])
    x, yy = E[m], y[m]
    if len(x) < 10:
        return 0.0, 0.0
    B = [(x - Emid) ** j for j in range(nbase)]
    B = [c for b in B for c in (b, -b)]                  # +/- columns: free-sign baseline in NNLS
    scale = max(np.abs(yy).max(), 1e-9)

    def rss_at(grid):
        out = np.empty(len(grid))
        for k, s in enumerate(grid):
            cols = [ls.evaluate(p["shape"], x, np.r_[c + s, 1.0, w]) for p, c, w in zip(peaks, centers, widths)]
            cols = [c / max(c.max(), 1e-12) for c in cols] + B
            out[k] = nnls(np.stack(cols, 1), yy / scale, maxiter=200)[1] ** 2 * scale ** 2
        return out

    shifts = np.asarray(shifts, float)
    rss = rss_at(shifts)
    j = int(np.argmin(rss))
    if len(shifts) > 1 and shifts[1] - shifts[0] > 1.5 * SHIFT_STEP:     # coarse -> fine
        d = shifts[1] - shifts[0]
        shifts = np.arange(max(shifts[j] - d, shifts[0]), min(shifts[j] + d, shifts[-1]) + 1e-12, SHIFT_STEP)
        rss = rss_at(shifts)
        j = int(np.argmin(rss))
    best = float(shifts[j])
    if 0 < j < len(shifts) - 1:                          # parabolic refinement
        a, b, c = rss[j - 1], rss[j], rss[j + 1]
        den = a - 2 * b + c
        if den > 0:
            best += 0.5 * (a - c) / den * (shifts[1] - shifts[0])
    if nbase:
        Bm = np.stack([(x - Emid) ** j for j in range(nbase)], 1)
        rss0 = float(np.sum((yy - Bm @ np.linalg.lstsq(Bm, yy, rcond=None)[0]) ** 2))
    else:
        rss0 = float(np.sum(yy ** 2))
    return best, (rss0 - float(rss[j])) / _noise_var(yy)


def _shifts(max_shift):
    """Coarse shift grid (shift_scan refines around its best point)."""
    n = int(np.ceil(max_shift / SHIFT_COARSE))
    return np.clip(np.arange(-n, n + 1) * SHIFT_COARSE, -max_shift, max_shift)


def _align(E, y, s, Egrid):
    """Spectrum y(E) moved by -s, sampled on Egrid (so a feature at E0 + s lands on E0)."""
    o = np.argsort(E)
    return np.interp(Egrid + s, E[o], y[o], left=np.nan, right=np.nan)


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
        if p["shape"] in ("voigt", "trion"):   # numerical error propagation for the FWHM
            d = 1e-6 if p["shape"] == "voigt" else 2e-5
            g = np.array([(ls.fwhm(p["shape"], np.r_[th[:2], th[2] + d, th[3]]) - f) / d,
                          (ls.fwhm(p["shape"], np.r_[th[:3], th[3] + d]) - f) / d])
            ferr = float(np.sqrt(np.sum((g * er[2:4]) ** 2)))
        else:
            ferr = er[2]
        w_lo, w_hi = np.isclose(th[2:], lo[s][2:], rtol=1e-3), np.isclose(th[2:], hi[s][2:], rtol=1e-3)
        # Voigt: one component (G or L) at the width floor just means the peak is effectively
        # pure L or pure G -- a valid fit. Trion: a tail at its floor is just a Gaussian.
        # Reject only if the widths collapse or one runs to the ceiling.
        if p["shape"] == "voigt":
            width_bad = w_hi.any() or w_lo.all()
        elif p["shape"] == "trion":
            width_bad = w_hi.any() or w_lo[0]
        else:
            width_bad = w_hi.any() or w_lo.any()
        at_bound = (np.isclose(th[0], lo[s][0], rtol=0, atol=2e-5) or np.isclose(th[0], hi[s][0], rtol=0, atol=2e-5)
                    or width_bad)
        snr = h / noise if noise > 0 else np.inf
        trion = p["shape"] == "trion"
        out.append({
            "center_nm": HC / e0, "center_meV": e0 * 1e3, "height": h, "intensity": area / de_line,
            "fwhm_meV": f * 1e3, "fwhm_nm": HC / (e0 - f / 2) - HC / (e0 + f / 2), "snr": snr,
            "tail_meV": th[3] * 1e3 if trion else np.nan,
            "center_nm_err": HC / e0 ** 2 * er[0], "center_meV_err": er[0] * 1e3,
            "intensity_err": er[1] / de_line, "fwhm_meV_err": ferr * 1e3,
            "tail_meV_err": er[3] * 1e3 if trion else np.nan,
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
    k = max(2, n // 400)
    ys = np.convolve(np.pad(y, k, mode="edge"), np.ones(2 * k + 1) / (2 * k + 1), "same")[k:-k]
    noise = 1.4826 * np.median(np.abs(np.diff(y))) / np.sqrt(2) / np.sqrt(2 * k + 1)
    base = running_median(ys, max(5, n // 8))
    sig = ys - np.minimum(base, ys)
    idx, prop = find_peaks(sig, prominence=8 * max(noise, 1e-9), distance=3 * k)
    if len(idx) == 0:
        log("auto-detect: no peaks found in the mean bright spectrum")
        return []
    order = np.sort(np.argsort(prop["prominences"])[::-1][:max_peaks])
    idx = idx[order]
    # widths on the same signal the peaks were found in, from their own prominences
    pd = tuple(prop[q][order] for q in ("prominences", "left_bases", "right_bases"))
    w = np.maximum(peak_widths(sig, idx, rel_height=0.5, prominence_data=pd)[0], 2 * k + 1)
    peaks = []
    for j, (i, wi) in enumerate(zip(idx, w)):
        dl = abs(np.interp(i + wi / 2, np.arange(n), wl) - np.interp(i - wi / 2, np.arange(n), wl))
        fwhm_meV = HC / wl[i] ** 2 * dl * 1e3
        peaks.append({"name": f"P{j + 1}", "center_nm": round(float(wl[i]), 2), "shape": "lorentzian",
                      "prominence": float(pd[0][j]),
                      "window_nm": round(float(max(2.0, 1.5 * dl)), 1), "window_meV": None,
                      "offset_meV": None,
                      "fwhm_meV": round(float(np.clip(fwhm_meV, 1.0, 60.0)), 1),
                      "min_fwhm_meV": 0.5, "max_fwhm_meV": round(float(max(80.0, 3 * fwhm_meV)), 0)})
    return peaks


MAIN_MIN_PROMINENCE = 0.3


def main_exciton(peaks):
    """The main (neutral) exciton: the highest-energy peak among the strong ones
    (prominence >= 30 % of the strongest), so weak shoulders or noise on the blue
    side are not picked."""
    if not peaks:
        return None
    pmax = max(p["prominence"] for p in peaks)
    strong = [p for p in peaks if p["prominence"] >= MAIN_MIN_PROMINENCE * pmax]
    return dict(min(strong, key=lambda p: p["center_nm"]), name="X0")


def auto_range(p, wl):
    """Fit range around a single peak: 4 widths (>= 2 nm) to the blue, 2.5 to the red,
    where the lower-energy bands start."""
    w = max(HC / (HC / p["center_nm"] - p["fwhm_meV"] / 2e3) - HC / (HC / p["center_nm"] + p["fwhm_meV"] / 2e3), 2.0)
    return [max(float(wl.min()), p["center_nm"] - 4 * w), min(float(wl.max()), p["center_nm"] + 2.5 * w)]


# -------------------------------------------------------------- per pixel
_W = {}


def _init_worker(ctx):
    _W.clear()
    _W.update(ctx)
    _W["step"] = np.abs(np.gradient(ctx["wl"]))


def _wl_step_at(lam):
    return float(np.interp(lam, _W["wl"], _W["step"]))


def _pixel_range(s):
    """Fit range (eV) for a pixel with rigid shift s."""
    lo, hi = _W["range_E"]
    if _W["range_moves"]:
        lo, hi = lo + s, hi + s
    return lo, hi


def _fit_index(i):
    W = _W
    E, y = W["E"], W["Y"][i]
    peaks, seed, Emid = W["peaks"], W["seed"], W["Emid"]
    cen = [seed[s][0] for s in W["slices"]]
    wid = [seed[s][2:] for s in W["slices"]]
    # pattern match: the point's rigid shift (if tracking) and whether it has any signal at all
    shifts = W["shifts"] if W["track"] else np.zeros(1)
    s, sig = shift_scan(E, y, peaks, cen, wid, W["scan_region"], shifts, W["scan_nbase"], Emid)
    if sig < SHIFT_MIN_DCHI2:
        s = 0.0
    lo, hi = _pixel_range(s)
    fm = (E >= lo) & (E <= hi)
    extra = {"shift_meV": s * 1e3, "shift_signif": sig, "range_lo_nm": HC / hi, "range_hi_nm": HC / lo,
             "integrated_counts": float(np.sum(y[fm]))}
    if W["skip_no_signal"] and sig < SHIFT_MIN_DCHI2:
        return i, "skipped", None, None, extra
    m = Model(peaks, E[fm], W["baseline"], shift=s, Emid=Emid)
    theta0 = m.initial(y[fm], [c + s for c in cen], wid)
    theta, cov = fit_model(m, y[fm], theta0)
    if theta is None:
        return i, None, None, None, extra
    d, st = derive(m, theta, cov, y[fm], _wl_step_at, W["min_snr"])
    return i, theta, d, st, extra


# ------------------------------------------------------------------ setup
def _relative(peaks):
    rel = [p.get("offset_meV") is not None for p in peaks]
    if any(rel) and not all(rel):
        raise SystemExit("fit.peaks: give either offset_meV (relative to the anchor) for every peak, "
                         "or center_nm for every peak, not a mix")
    return all(rel)


def _anchor_index(peaks, name):
    if name is not None:
        for i, p in enumerate(peaks):
            if p["name"] == name:
                return i
        raise SystemExit(f"fit.anchor {name!r} is not one of the peaks {[p['name'] for p in peaks]}")
    if peaks and peaks[0].get("offset_meV") is not None:
        zero = [i for i, p in enumerate(peaks) if abs(p["offset_meV"]) < 1e-9]
        return zero[0] if zero else int(np.argmax([p["offset_meV"] for p in peaks]))
    return int(np.argmin([p["center_nm"] for p in peaks]))     # highest-energy peak


def _template_extent(peaks):
    """Energy span (eV) of a peak set: lowest centre - 3 FWHM - 5 meV ... highest + 3 FWHM + 5 meV."""
    lo = min(p["E0"] - 3 * p["fwhm_meV"] / 1e3 for p in peaks) - 5e-3
    hi = max(p["E0"] + 3 * p["fwhm_meV"] / 1e3 for p in peaks) + 5e-3
    return lo, hi


ANCHOR_POINTS = 10


def find_anchor(E, Ys, peaks, ia, search_E, rel_range, nbase, log=print):
    """Anchor energy: slide the configured pattern (widths from fwhm_meV) over search_E and
    keep the best NNLS match, for each of the spectra Ys (the brightest points); return the median.

    Individual spectra are used, not their mean: averaging points that are shifted by a few meV
    smears peaks a few meV apart (the two trions) into one, and the pattern can then lock one
    peak spacing off."""
    Ea0 = peaks[ia]["E0"]
    cen = [p["E0"] for p in peaks]
    wid = [default_widths(p) for p in peaks]
    shifts = np.arange(search_E[0] - Ea0, search_E[1] - Ea0 + SHIFT_COARSE, SHIFT_COARSE)
    region = (search_E[0] + rel_range[0], search_E[1] + rel_range[1])
    found = []
    for y in Ys:
        s, sig = shift_scan(E, y, peaks, cen, wid, region, shifts, max(nbase, 3), Ea0)
        if sig >= SHIFT_MIN_DCHI2:
            found.append(Ea0 + s)
    if not found:
        raise SystemExit("no match of the peak pattern in the brightest points -- check fit.peaks offsets/shapes, "
                         "or set fit.anchor_search_nm")
    found = np.array(found)
    Ea = float(np.median(found))
    log(f"anchor {peaks[ia]['name']}: pattern match in {HC / search_E[1]:.1f}-{HC / search_E[0]:.1f} nm on the "
        f"{len(found)} brightest points -> {Ea * 1e3:.1f} meV = {HC / Ea:.2f} nm "
        f"(spread {np.percentile(found, 10) * 1e3:.1f}-{np.percentile(found, 90) * 1e3:.1f} meV)")
    odd = np.abs(found - Ea) > 0.6 * min(abs(p["E0"] - q["E0"]) for p in peaks for q in peaks if p is not q)
    if odd.any():
        log(f"  note: {odd.sum()} of {len(found)} points matched far from the median "
            f"(an alias by one peak spacing, or a real shift); the median is used")
    return Ea


def run(cfg, wl, spc, present, log=print):
    """Fit all present pixels of background-subtracted spectra `spc` [npts, nwl]."""
    from .background import bright_points
    fc = cfg["fit"]
    E = HC / wl
    Yall = np.nan_to_num(spc)
    fixed = fc["range_nm"]
    fm0 = (wl >= min(fixed)) & (wl <= max(fixed)) if fixed else np.ones(len(wl), bool)
    bright = bright_points(np.nansum(spc[:, fm0], 1), present)
    nb = len(bright)
    brightest = bright[np.argsort(np.nansum(Yall[bright][:, fm0], 1))[::-1][:ANCHOR_POINTS]]
    ymean_raw = Yall[bright].mean(0)

    peaks = [dict(p) for p in fc["peaks"]]
    if not peaks:
        mode = str(fc["auto_detect"]).lower()
        if mode in ("false", "none", "off", "no"):
            raise SystemExit("no peaks in fit.peaks and auto_detect is off")
        found = detect_peaks(wl[fm0], ymean_raw[fm0], 10 if mode != "all" else fc["max_peaks"], log)
        if not found:
            raise SystemExit("no peaks to fit")
        if mode == "all":
            peaks = found
        else:
            peaks = [main_exciton(found)]
            log(f"auto-detect: {len(found)} peaks in the mean bright spectrum; fitting only the main exciton "
                f"(highest-energy strong peak) at {peaks[0]['center_nm']:.2f} nm")
            if not fixed:
                fixed = auto_range(peaks[0], wl)
                log(f"  fit range {fixed[0]:.1f}-{fixed[1]:.1f} nm (set fit.range_nm to change)")
        for p in peaks:
            p.pop("prominence", None)
        log("auto-detected peaks (copy into fit.peaks to edit names/shapes):")
        for p in peaks:
            log(f"  - {{name: {p['name']}, center_nm: {p['center_nm']}, shape: lorentzian, "
                f"window_nm: {p['window_nm']}, fwhm_meV: {p['fwhm_meV']}}}")
    names = [p["name"] for p in peaks]
    if len(set(names)) != len(names):
        raise SystemExit(f"peak names must be unique: {names}")

    combos = combined_spec(fc.get("combined"), peaks)

    # ---- 1. anchor and global range
    relative = _relative(peaks)
    ia = _anchor_index(peaks, fc.get("anchor"))
    nbase = NBASE[fc["baseline"]]
    Emin, Emax = float(E.min()), float(E.max())
    max_shift = float(fc.get("max_shift_meV") or 0) / 1e3
    track = bool(fc.get("track_shift", True)) and max_shift > 0
    if relative:
        rel_span = tuple(np.array(_template_extent(prepare_peaks(peaks, 1.0))) - 1.0)  # span relative to anchor
        search = fc.get("anchor_search_nm")
        if search:
            search_E = (HC / max(search), HC / min(search))
        else:                                        # anywhere the whole pattern fits in the data
            search_E = (Emin - rel_span[0], Emax - rel_span[1])
            if search_E[0] >= search_E[1]:
                search_E = (Emin, Emax)
        tmpl = prepare_peaks(peaks, 0.5 * (search_E[0] + search_E[1]))
        Ea = find_anchor(E, Yall[brightest], tmpl, ia, search_E, rel_span, nbase, log)
        peaks = prepare_peaks(peaks, Ea)
    else:
        peaks = prepare_peaks(peaks, 0.0)
        Ea = peaks[ia]["E0"]
        if track:
            Ea_new = find_anchor(E, Yall[brightest], peaks, ia, (Ea - max_shift, Ea + max_shift),
                                 tuple(np.array(_template_extent(peaks)) - Ea), nbase, log)
            peaks = [dict(p, E0=p["E0"] + Ea_new - Ea, center_nm=HC / (p["E0"] + Ea_new - Ea)) for p in peaks]
            Ea = Ea_new

    if fixed:
        range_E = (HC / max(fixed), HC / min(fixed))
        range_moves = False
    elif fc.get("range_meV"):
        r = sorted(fc["range_meV"])
        range_E = (Ea + r[0] / 1e3, Ea + r[1] / 1e3)
        range_moves = True
    else:
        range_E = _template_extent(peaks)
        range_moves = True
    range_E = (max(range_E[0], Emin), min(range_E[1], Emax))
    fm = (E >= range_E[0]) & (E <= range_E[1])
    if fm.sum() < 10:
        raise SystemExit(f"fit range {HC / range_E[1]:.2f}-{HC / range_E[0]:.2f} nm does not overlap the measured "
                         f"wavelengths {wl.min():.2f}-{wl.max():.2f} nm")
    outside = [p for p in peaks if not range_E[0] <= p["E0"] <= range_E[1]]
    if outside:
        raise SystemExit("peak centre outside the fit range %.2f-%.2f nm: %s -- check fit.peaks or the fit range"
                         % (HC / range_E[1], HC / range_E[0],
                            ", ".join(f"{p['name']} at {p['center_nm']:.2f} nm" for p in outside)))
    log(f"fit range {HC / range_E[1]:.2f}-{HC / range_E[0]:.2f} nm "
        f"({'moves with each point' if range_moves and track else 'fixed'})"
        + (f"; per-point shift search +-{max_shift * 1e3:.0f} meV" if track else ""))
    for p in peaks:
        log(f"  {p['name']:>6}: start {p['E0'] * 1e3:.1f} meV ({p['center_nm']:.2f} nm), "
            f"centre window +-{p['window_meV']:.1f} meV, {p['shape']}")

    Egrid = E[fm]
    Emid = float(np.mean(Egrid))
    model = Model(peaks, Egrid, fc["baseline"], Emid=Emid)
    scan_region = (range_E[0] - max_shift, range_E[1] + max_shift) if range_moves else range_E
    ctx = dict(E=E, Y=Yall, peaks=peaks, slices=model.slices, baseline=fc["baseline"], Emid=Emid,
               range_E=range_E, range_moves=range_moves, track=track, shifts=_shifts(max_shift),
               scan_region=scan_region, scan_nbase=max(nbase, 2), min_snr=fc["min_snr"], wl=wl, seed=None,
               skip_no_signal=bool(fc.get("skip_no_signal", True)))
    _init_worker(ctx)

    # ---- 2. template: fit the mean of the brightest points, aligned to each other.
    # Pass 0 aligns them with the configured pattern placed at the anchor, so the first fit already
    # refers to the same reference as the anchor (an unaligned mean can sit a few meV away, and the
    # narrow centre windows would then catch the wrong peaks). Passes 1-2 realign with the fitted template.
    bshift = np.zeros(nb)
    ymean = ymean_raw[fm]
    theta_g = model.initial(ymean)
    for it in range(3 if track else 1):
        if track:
            cen = [p["E0"] for p in peaks] if it == 0 else [theta_g[s][0] for s in model.slices]
            wid = [default_widths(p) for p in peaks] if it == 0 else [theta_g[s][2:] for s in model.slices]
            for j, i in enumerate(bright):
                s, sig = shift_scan(E, Yall[i], peaks, cen, wid, scan_region, _shifts(max_shift), max(nbase, 2), Emid)
                bshift[j] = s if sig >= SHIFT_MIN_DCHI2 else 0.0
            ymean = np.nanmean([_align(E, Yall[i], s, Egrid) for i, s in zip(bright, bshift)], 0)
            ymean = np.where(np.isfinite(ymean), ymean, ymean_raw[fm])
        th = fit_model(model, ymean, model.initial(ymean) if it == 0 else theta_g, maxfev=20000)[0]
        if th is None:
            raise SystemExit("fit of the mean bright spectrum failed -- check fit.peaks and the fit range")
        theta_g = th
    if track:
        log(f"brightest {nb} points: rigid shifts {np.percentile(bshift, 5) * 1e3:+.1f} ... "
            f"{np.percentile(bshift, 95) * 1e3:+.1f} meV (5-95 %); template = their aligned mean")
    lo_b, hi_b = model.bounds()
    edge = [p["name"] for p, sl in zip(peaks, model.slices)
            if np.isclose(theta_g[sl][0], lo_b[sl][0], atol=2e-5) or np.isclose(theta_g[sl][0], hi_b[sl][0], atol=2e-5)]
    if edge:
        log(f"  WARNING: in the template fit, {', '.join(edge)} ended at the edge of the centre window -- "
            f"check offset_meV / window_meV for {'it' if len(edge) == 1 else 'them'}")
    cov_g = fit_model(model, ymean, theta_g, maxfev=20000)[1]
    d_g, st_g = derive(model, theta_g, cov_g, ymean, _wl_step_at, 0)
    log(f"mean of {nb} brightest points{' (aligned)' if track else ''}: R^2 = {st_g['r2']:.4f}")
    for p, d in zip(peaks, d_g):
        rel = f"  ({d['center_meV'] - d_g[ia]['center_meV']:+.1f} meV from {peaks[ia]['name']})" if p is not peaks[ia] else ""
        tail = f", tail {d['tail_meV']:.2f} meV" if p["shape"] == "trion" else ""
        log(f"  {p['name']:>6} ({p['shape']:<10}): {d['center_nm']:.3f} nm = {d['center_meV']:.1f} meV{rel}, "
            f"FWHM {d['fwhm_meV']:.2f} meV{tail}, height {d['height']:.1f}")

    top = np.argsort(np.nansum(Yall[bright][:, fm], 1))[::-1][:5]
    comparison = compare_shapes(peaks, E, Yall[bright[top]], bshift[top], theta_g, fc["baseline"], Emid,
                                range_E, range_moves, log)

    # ---- 3. per-pixel fits
    ctx["seed"] = theta_g
    _init_worker(ctx)
    todo = np.flatnonzero(present)
    npts = len(spc)
    theta_all = np.full((npts, model.n), np.nan)
    res = {f"{p['name']}_{q}": np.full(npts, np.nan) for p in peaks for q in DERIVED + DERIVED_ERR}
    for p in peaks:
        res[f"{p['name']}_ok"] = np.zeros(npts, bool)
    for q in ("r2", "noise", "shift_meV", "shift_signif", "range_lo_nm", "range_hi_nm", "integrated_counts"):
        res[q] = np.full(npts, np.nan)
    res["fit_failed"] = np.zeros(npts, bool)
    res["no_signal"] = np.zeros(npts, bool)

    desc = ", ".join("%s=%s" % (p["name"], p["shape"]) for p in peaks)
    log(f"fitting {len(todo)} points ({desc}; baseline {fc['baseline']}) ...")
    workers = fc["workers"]
    workers = min(os.cpu_count() or 1, 8) if workers in ("auto", None) else int(workers)
    workers = max(1, min(workers, len(todo) // 20 or 1))
    if workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork" if hasattr(os, "fork") else "spawn").Pool(workers, _init_worker, (ctx,)) as pool:
            results = pool.imap_unordered(_fit_index, todo, chunksize=16)
            results = list(_progress(results, len(todo), log))
    else:
        results = list(_progress(map(_fit_index, todo), len(todo), log))
    for i, theta, d, st, extra in results:
        for q, v in extra.items():
            res[q][i] = v
        if isinstance(theta, str):          # no peak pattern found: not fitted (counts as rejected)
            res["no_signal"][i] = True
            continue
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
    if res["no_signal"].any():
        log(f"  {res['no_signal'].sum()} points without the peak pattern (off-flake) were not fitted "
            f"(fit.skip_no_signal: false fits them anyway)")
    for p in peaks:
        log(f"  {p['name']}: good fits at {res[p['name'] + '_ok'].sum()} of {len(todo)} points")
    if track:
        tracked = np.isfinite(res["shift_signif"]) & (res["shift_signif"] >= SHIFT_MIN_DCHI2)
        sh = res["shift_meV"][tracked]
        edge = np.sum(np.abs(sh) > 0.97 * max_shift * 1e3)
        log(f"  rigid shift tracked at {tracked.sum()} points: {np.percentile(sh, 5) if sh.size else 0:+.1f} ... "
            f"{np.percentile(sh, 95) if sh.size else 0:+.1f} meV (5-95 %); other points use the template position")
        if edge:
            log(f"  WARNING: {edge} points have a shift at the search limit -- raise fit.max_shift_meV")
    if res["fit_failed"].any():
        log(f"  {res['fit_failed'].sum()} fits failed to converge")
    dg = {p["name"]: d for p, d in zip(peaks, d_g)}
    for c in combos:
        v, g = combine(c, lambda n, q: res[f"{n}_{q}"], lambda n: res[f"{n}_ok"])
        res[c["name"]], res[c["name"] + "_ok"] = v, g
        tv, _ = combine(c, lambda n, q: dg[n][q], lambda n: True)
        c["template"], c["good"] = float(tv), int(g.sum())
        c["median"] = float(np.median(v[g])) if g.any() else float("nan")
        pr = 4 if c["kind"] == "ratio" else 2
        log(f"  {c['name']} = {c['label']}: template {tv:.{pr}f}; median {c['median']:.{pr}f} "
            f"over {g.sum()} points where all its peaks are good")
    return {"peaks": peaks, "model": model, "theta": theta_all, "res": res, "fit_mask": fm,
            "mean": {"y": ymean, "theta": theta_g, "derived": d_g, "stats": st_g, "points": bright},
            "comparison": comparison, "anchor": peaks[ia]["name"], "ctx_range_moves": range_moves and track,
            "combined": combos}


def _progress(it, n, log):
    step = max(1, n // 10)
    for j, r in enumerate(it, 1):
        if j % step == 0 or j == n:
            log(f"  {j}/{n}")
        yield r


SHAPE_GAIN = 0.02


def recommend(rel):
    """Simplest adequate shape: Voigt and trion have one more parameter than L or G, so they
    always fit at least as well; prefer one only if it beats both L and G by > 2 %."""
    l, g = rel["lorentzian"], rel["gaussian"]
    simple = "lorentzian" if (l <= g or not np.isfinite(g)) else "gaussian"
    ref = min(l, g)
    four = {s: rel[s] for s in ("voigt", "trion") if s in rel and np.isfinite(rel[s])}
    if four:
        s = min(four, key=four.get)
        if four[s] < (1 - SHAPE_GAIN) * ref:
            return s
    return simple


def compare_shapes(peaks, E, Ys, shifts, seed, baseline, Emid, range_E, range_moves, log=print):
    """For each peak, refit the brightest individual spectra `Ys` with that peak
    as each lineshape (other peaks unchanged). Individual points are used rather
    than their average because averaging points with different peak energies
    broadens peaks inhomogeneously and biases the test towards Voigt.
    Returns {peak: {shape: summed chi^2 / best, "recommended": shape}}."""
    base = Model(peaks, E[:2], baseline, Emid=Emid)
    cen = [seed[s][0] for s in base.slices]
    wid = [seed[s][2:] for s in base.slices]
    fw = [ls.fwhm(p["shape"], seed[s]) for p, s in zip(peaks, base.slices)]
    out, rows = {}, []
    for i, p in enumerate(peaks):
        chi = {}
        for shape in ls.SHAPES:
            trial = [dict(q) for q in peaks]
            trial[i]["shape"] = shape
            tot = 0.0
            for y, s in zip(Ys, shifts):
                lo, hi = (range_E[0] + s, range_E[1] + s) if range_moves else range_E
                fm = (E >= lo) & (E <= hi)
                m = Model(trial, E[fm], baseline, shift=s, Emid=Emid)
                w = [wid[k] if k != i or shape == p["shape"] else default_widths(trial[i], fw[i])
                     for k in range(len(peaks))]
                th, _ = fit_model(m, y[fm], m.initial(y[fm], [c + s for c in cen], w), maxfev=3000)
                if th is None:
                    tot = np.nan
                    break
                tot += float(np.sum((y[fm] - m(E[fm], *th)) ** 2) / max(fm.sum() - m.n, 1))
            chi[shape] = tot
        best = np.nanmin(list(chi.values()))
        out[p["name"]] = {s: v / best for s, v in chi.items()}
        pick = recommend(out[p["name"]])
        out[p["name"]]["recommended"] = pick
        rows.append(f"  {p['name']:>6}: " + "  ".join(f"{s[:4]} {v / best:6.3f}" for s, v in chi.items())
                    + f"   -> recommended: {pick}")
    log(f"lineshape check on the {len(Ys)} brightest points (summed reduced chi^2 / best; 1.000 = best;\n"
        f"  Voigt or trion is recommended only if it beats both Lorentzian and Gaussian by > {SHAPE_GAIN:.0%}):")
    for r in rows:
        log(r)
    return out
