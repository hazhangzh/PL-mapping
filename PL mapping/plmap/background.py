"""Step 3a: background subtraction.

Applied in this order (each optional):
  1. dark spectrum            (background.dark_file)
  2. per-point offset         mean of a signal-free band; removes CCD offset and drift
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


def offset_mask(cfg, wl):
    reg = cfg["background"]["offset_region_nm"]
    if reg in ("none", False):
        return None
    if reg is None:
        reg = [wl.max() - 8.0, wl.max() + 1.0]
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
    om = offset_mask(cfg, wl)
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
