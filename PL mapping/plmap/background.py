"""Step 3a: background subtraction.

Applied in this order (each optional):
  1. dark spectrum            (background.dark_file)
  2. per-point offset         mean of a signal-free band; removes CCD offset and drift
                              (default: whichever 8-nm spectral edge carries less PL)
  3. substrate background     mean spectrum of the dimmest (off-flake) points, or a file
A constant/linear baseline is additionally fitted per pixel (fit.baseline).
"""
import os

import numpy as np

from . import io
from .spectra import average_rows


def load_spectrum(path, wl):
    """Spectrum file -> 1D array on the map's wavelength grid.
    Accepts a raw loc-style matrix (rows averaged), a single column, or (wavelength, counts)."""
    a = io.load_matrix(path)
    if a.ndim == 1 or a.shape[1] == 1:
        y = a.ravel()
        if len(y) != len(wl):
            raise SystemExit(f"{path}: {len(y)} values but {len(wl)} wavelengths")
        return y
    if a.shape[1] == 2:  # (wavelength nm, counts) if the first column looks like wavelengths
        x, y = a[:, 0], a[:, 1]
        if np.all((x > wl.min() - 50) & (x < wl.max() + 50)) and np.all(np.diff(x) != 0):
            o = np.argsort(x)
            return np.interp(wl, x[o], y[o])
    if a.shape[0] == len(wl):
        return average_rows(a)[0]
    if a.shape[1] == len(wl):
        return average_rows(a.T)[0]
    raise SystemExit(f"Don't understand the format of {path} (shape {a.shape})")


EDGE_NM = 8.0


def bright_points(tot, present, max_frac=0.1, nsig=8.0):
    """Indices of points whose integrated signal `tot` stands clearly above the
    scatter of the map (most points are off-flake), brightest first.
    At least 3 and at most `max_frac` of the present points. If almost nothing
    stands out (e.g. the whole map is on the flake), the top `max_frac` are used."""
    t = np.where(present, tot, np.nan)
    ok = np.flatnonzero(np.isfinite(t))
    order = ok[np.argsort(t[ok])[::-1]]
    nmax = max(3, int(max_frac * len(ok)))
    med = np.median(t[ok])
    sig = 1.4826 * np.median(np.abs(t[ok] - med))
    n = int(np.sum(t[ok] > med + nsig * sig)) if sig > 0 else 0
    if n < 3:
        n = nmax
    return order[:min(max(n, 3), nmax)]


def _edge_masks(wl):
    return {"blue": wl <= wl.min() + EDGE_NM, "red": wl >= wl.max() - EDGE_NM}


def choose_offset_band(wl, spectra, present, log=print):
    """Pick the spectral edge (blue or red, EDGE_NM wide) that carries less PL.

    The CCD offset drifts between points, but it shifts both edges equally,
    so the red-minus-blue difference is drift-free. On off-flake points it is
    the fixed slope of the CCD offset; on bright points PL in one edge band
    moves it up (red has PL) or down (blue has PL)."""
    m = _edge_masks(wl)
    S = spectra[present]
    b, r = S[:, m["blue"]].mean(1), S[:, m["red"]].mean(1)
    mid = S[:, ~(m["blue"] | m["red"])].mean(1) - 0.5 * (b + r)
    order = np.argsort(mid)
    nb = max(3, int(0.05 * len(S)))
    d = r - b
    bright, dim = d[order[-nb:]], d[order[:len(S) // 2]]
    diff = np.median(bright) - np.median(dim)
    err = 1.4826 * np.median(np.abs(dim - np.median(dim))) * 1.25 / np.sqrt(nb)
    if abs(diff) < 3 * err:
        side, why = "red", "no PL detected at either edge"
    else:
        side = "blue" if diff > 0 else "red"
        why = f"{'red' if diff > 0 else 'blue'} edge carries {abs(diff):.2f} counts of PL on the brightest points"
    log(f"offset band: {side} edge ({why}; set background.offset_region_nm to override)")
    return m[side]


def offset_mask(cfg, wl, spectra=None, present=None, log=print):
    reg = cfg["background"]["offset_region_nm"]
    if reg in ("none", False):
        return None
    if reg in (None, "auto"):
        if spectra is None:
            return _edge_masks(wl)["red"]
        return choose_offset_band(wl, spectra, present, log)
    m = (wl >= min(reg)) & (wl <= max(reg))
    if m.sum() < 5:
        raise SystemExit(f"background.offset_region_nm {reg} contains fewer than 5 spectral points")
    return m


def subtract(cfg, wl, spectra, present, fit_mask, log=print):
    bc = cfg["background"]
    sp = spectra.copy()
    info = {"dark": None, "offset": np.zeros(len(sp)), "substrate": np.zeros(len(wl)),
            "substrate_points": np.zeros(len(sp), bool)}
    if bc["dark_file"]:
        info["dark"] = load_spectrum(bc["dark_file"], wl)
        sp -= info["dark"]
        log(f"subtracted dark spectrum {os.path.basename(bc['dark_file'])}")
    om = offset_mask(cfg, wl, sp, present, log)
    if om is not None:
        with np.errstate(all="ignore"):
            info["offset"] = np.where(present, np.nanmean(np.where(present[:, None], sp[:, om], 0.0), 1), np.nan)
        sp -= info["offset"][:, None]
        log(f"per-point offset from {wl[om].min():.1f}-{wl[om].max():.1f} nm: "
            f"median {np.nanmedian(info['offset']):.1f}, spread (5-95 %) "
            f"{np.nanpercentile(info['offset'], 5):.1f}-{np.nanpercentile(info['offset'], 95):.1f} counts")

    sub = bc["substrate"]
    if sub == "auto":
        tot = np.nansum(sp[:, fit_mask], 1)
        tot[~present] = np.nan
        n = max(1, int(round(bc["substrate_fraction"] * present.sum())))
        idx = np.argsort(np.where(present, tot, np.inf))[:n]
        info["substrate_points"][idx] = True
        info["substrate"] = np.nanmean(sp[idx], 0)
        log(f"substrate background = mean of the {n} dimmest points ({bc['substrate_fraction']:.0%}); "
            f"integral in fit range {np.nansum(info['substrate'][fit_mask]):.0f} counts "
            f"(brightest point {np.nanmax(tot):.0f}, median point {np.nanmedian(tot[present]):.0f})")
        if np.nanmax(tot) < 3 * max(abs(np.nanmedian(tot)), 1e-9):
            log("  note: little contrast between points -- if the whole map is on the flake, "
                "set background.substrate: none or give a substrate spectrum file")
    elif sub not in ("none", None):
        s = load_spectrum(sub, wl)
        if info["dark"] is not None:
            s = s - info["dark"]
        if om is not None:
            s = s - np.mean(s[om])
        info["substrate"] = s
        log(f"subtracted substrate spectrum {os.path.basename(sub)}")
    sp -= info["substrate"]
    sp[~present] = np.nan
    return sp, info
