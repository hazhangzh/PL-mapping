"""Step 1: raw CCD matrices -> one averaged spectrum per loc file.

Each raw file has one line per wavelength and one column per CCD row. The
spectrum is the mean over all CCD rows of each line, after rejecting cosmic
rays. Needs only numpy (runs on the lab computer).
"""
import hashlib
import os
import warnings

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from . import io


def running_median(a, k, axis=0):
    """Median over a window of 2k+1 points along `axis`, edges padded."""
    a = np.moveaxis(a, axis, 0)
    pad = np.pad(a, [(k, k)] + [(0, 0)] * (a.ndim - 1), mode="edge")
    out = np.median(sliding_window_view(pad, 2 * k + 1, axis=0), axis=-1)
    return np.moveaxis(out, 0, axis)


def _mad_sigma(r, axis):
    return 1.4826 * np.median(np.abs(r), axis=axis, keepdims=True)


def _robust_sigma(r, axis, floor=0.0):
    """Noise level of residuals `r`: standard deviation after clipping outliers
    at 5x the MAD estimate. Unlike the MAD alone this is not fooled by integer
    counts (MAD of integer data is quantised and can be ~30 % too small)."""
    s0 = np.maximum(_mad_sigma(r, axis), max(floor, 0.5))
    keep = np.abs(r) < 5 * s0
    with np.errstate(all="ignore"):
        s = np.sqrt(np.sum(np.where(keep, r ** 2, 0), axis=axis, keepdims=True) /
                    np.maximum(np.sum(keep, axis=axis, keepdims=True), 1))
    return np.maximum(s, floor)


def average_rows(a, cr_sigma=6.0, k=3, floor=1.0):
    """Mean over CCD rows (columns of `a`), ignoring cosmic-ray pixels.

    Cosmic rays are found per CCD row as sharp outliers above a running
    median along the wavelength axis (window 2k+1 = 7 lines). PL peaks are
    tens to hundreds of lines wide and are not affected, and the test does not
    depend on how many CCD rows the PL spot covers. Flagged pixels and their
    spectral neighbours are excluded from the mean.
    Returns (spectrum, number of rejected pixels).
    """
    if not cr_sigma:
        return a.mean(1), 0
    smooth = running_median(a, k, axis=0)
    r = a - smooth
    hit = r > cr_sigma * _robust_sigma(r, 0, floor)
    cr = hit | np.roll(hit, 1, 0) | np.roll(hit, -1, 0)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # lines with every row rejected -> handled below
        sp = np.where(cr, np.nan, a)
        out = np.nanmean(sp, 1)
    allbad = np.isnan(out)
    out[allbad] = np.median(smooth[allbad], 1)
    return out, int(hit.sum())


def despike(sp, nsig=8.0, k=7, floor=0.3):
    """Replace remaining single-line spikes in averaged spectrum/spectra (last axis)."""
    if not nsig:
        return sp
    sp = np.atleast_2d(sp)
    rm = running_median(sp, k, axis=1)
    r = sp - rm
    s = _robust_sigma(r, 1, floor)
    return np.where(r > nsig * s, rm, sp)


def _fingerprint(files):
    h = hashlib.sha1()
    for k in sorted(files):
        st = os.stat(files[k])
        h.update(f"{k}:{st.st_size}:{st.st_mtime_ns};".encode())
    return h.hexdigest()


def build(cfg, grid, wl, log=print):
    """Load (or reuse cached) averaged spectra. Returns dict with
    spectra [npts, nwl] (NaN rows for missing files), present mask, ks, prefix."""
    out = cfg["data"]["output"]
    prefix, files = io.find_loc_files(cfg["data"]["folder"], cfg["data"]["prefix"])
    sc = cfg["spectra"]
    fp = _fingerprint(files) + f"|{sc['cosmic_ray_sigma']}|{sc['despike_sigma']}|{grid.npts}"
    cache = os.path.join(out, "spectra.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=False)
        if str(z["fingerprint"]) == fp and z["spectra"].shape[1] == len(wl):
            log(f"reusing averaged spectra from {cache}")
            return {"spectra": z["spectra"], "present": z["present"], "prefix": prefix,
                    "ks": np.arange(1, grid.npts + 1), "files": files}

    extra = sorted(k for k in files if k > grid.npts)
    if extra:
        log(f"WARNING: {len(extra)} files have loc index > nx*ny = {grid.npts} "
            f"(e.g. loc{extra[0]}); check scan.x_steps / y_steps. They are ignored.")
    spectra = np.full((grid.npts, len(wl)), np.nan)
    ncr = 0
    for k in range(1, grid.npts + 1):
        if k not in files:
            continue
        a = io.load_matrix(files[k])
        if a.shape[0] != len(wl):
            if a.shape[1] == len(wl):
                a = a.T
            else:
                raise SystemExit(f"{files[k]}: {a.shape[0]} lines but the wavelength file has {len(wl)} values")
        sp, n = average_rows(a, sc["cosmic_ray_sigma"])
        spectra[k - 1] = sp
        ncr += n
        if k % 200 == 0:
            log(f"  read {k}/{grid.npts}")
    present = ~np.isnan(spectra[:, 0])
    spectra[present] = despike(spectra[present], sc["despike_sigma"])
    log(f"loaded {present.sum()} of {grid.npts} points (run prefix '{prefix}'), "
        f"{a.shape[1]} CCD rows averaged, {ncr} cosmic-ray pixels rejected (plus 1 neighbour each side)")
    if (~present).any():
        miss = np.flatnonzero(~present) + 1
        log(f"  missing loc indices: {_ranges(miss)} (shown blank)")
    os.makedirs(out, exist_ok=True)
    np.savez_compressed(cache, spectra=spectra, present=present, wl=wl, fingerprint=fp)
    return {"spectra": spectra, "present": present, "prefix": prefix,
            "ks": np.arange(1, grid.npts + 1), "files": files}


def _ranges(ks):
    ks = list(ks)
    out, start = [], ks[0]
    for a, b in zip(ks, ks[1:] + [None]):
        if b != a + 1:
            out.append(f"{start}" if start == a else f"{start}-{a}")
            if b is not None:
                start = b
    s = ", ".join(out)
    return s if len(s) < 200 else s[:200] + " ..."
