#!/usr/bin/env python3
"""
plot_rc_map.py -- quick-look plotter for reflection-contrast (RC) maps
(Andor spectrometer, Cundiff group). Needs only numpy + matplotlib.

    RC = (R_sample - R0) / R0       (both dark-subtracted, same exposure units)

Inputs
------
  folder     map folder with 1_loc1.txt ... 1_locN.txt (each 2048 x NROWS,
             spectral pixel x CCD row; row-major raster) and x_vector*.txt
  --r0 FILE  substrate reference spectrum R0, taken next to the flake with the
             SAME focus (axis3), exposure, slit and ybin. Accepted formats:
               * 2-column txt (wavelength nm, counts)        [e.g. *_BG.txt]
               * a 2048 x NROWS matrix like the map files    [rows averaged
                 over the same CCD-row window as the map]
             If omitted, R0 = median spectrum of the substrate pixels in the
             map itself (works when the scan contains plenty of bare substrate).
  --dark FILE  dark spectrum (lamp blocked / 0 uW) with the same exposure,
             same formats as --r0. Otherwise a flat CCD offset (--offset,
             default 815 counts) is subtracted.

Exposure: give --exp-map and --exp-r0 (s) if they differ; spectra are
converted to counts/s before taking the ratio. A 2-column R0 that was saved
as a SUM over n CCD rows (Andor ybin) must be put on the map's per-row scale:
use --r0-rows n (divides R0 by n). If unsure, use --normalize-substrate, which
rescales R0 so that RC averages to 0 over the substrate pixels of the map.

Typical use
-----------
  python3 plot_rc_map.py /path/to/RC_map --r0 /path/to/R0_BG.txt --nx 41 --ny 41 \
      --step 0.5 --flip-lr --normalize-substrate

  # plus new scan limits from the old ones (axis1 = fast/columns, mm)
  ... --axis1 0.2730 0.2930 --axis2 0.2565 0.2765 --new-size 10 --new-step 0.25

Output: <folder>/<YYYY.MM.DD>_RC_quicklook/ with RCmap_quicklook.png/.pdf and
rc_cube.npz, plus a printout of the flake extent and a suggested scan window.
"""
import argparse, glob, os, re, sys, datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HC = 1239.84193  # eV nm


def find_xvector(folder):
    c = sorted(glob.glob(os.path.join(folder, "x_vector*.txt")))
    if not c:
        sys.exit("No x_vector*.txt (wavelength axis) found in " + folder)
    return np.loadtxt(c[0])


def load_matrix(path):
    """Load a txt matrix; '--' (dropped pixel) is interpolated."""
    txt = open(path).read()
    if "--" in txt:
        a = np.genfromtxt(path, missing_values="--", filling_values=np.nan)
        a = np.atleast_2d(a)
        for j in range(a.shape[1]):
            col = a[:, j]; bad = np.isnan(col)
            if bad.any() and (~bad).any():
                col[bad] = np.interp(np.where(bad)[0], np.where(~bad)[0], col[~bad])
        return a
    return np.array([r.split() for r in txt.strip().splitlines()], dtype=np.float64)


def load_cube(folder, npts):
    files = {int(m.group(1)): f for f in glob.glob(os.path.join(folder, "1_loc*.txt"))
             if (m := re.search(r"1_loc(\d+)\.txt$", f))}
    if not files:
        sys.exit("No 1_loc*.txt files found in " + folder)
    first = load_matrix(files[min(files)])
    npix, nrow = first.shape
    cube = np.full((npts, nrow, npix), np.nan, np.float32)
    for k in range(1, npts + 1):
        if k in files:
            cube[k - 1] = load_matrix(files[k]).T
    print(f"loaded {len(files)} of {npts} files; each {npix} px x {nrow} CCD rows")
    return cube


def combine_rows(cube, r0, r1, nsig=6.0):
    """Cosmic-ray-robust mean over CCD rows r0..r1 (CR mask from all rows)."""
    med = np.nanmedian(cube, 1, keepdims=True)
    mad = np.nanmedian(np.abs(cube - med), 1, keepdims=True) * 1.4826
    cr = cube > med + nsig * np.maximum(mad, 3.0)
    cr = cr | np.roll(cr, 1, 1) | np.roll(cr, -1, 1) | np.roll(cr, 1, 2) | np.roll(cr, -1, 2)
    with np.errstate(all="ignore"):
        sp = np.nanmean(np.where(cr, np.nan, cube)[:, r0:r1 + 1], 1)
    fill = np.isnan(sp) & ~np.isnan(cube[:, 0, :])
    sp[fill] = np.nanmedian(cube[:, r0:r1 + 1], 1)[fill]
    return sp


def load_spectrum(path, wl, r0, r1):
    """Return a 1D spectrum on the map's wavelength grid, per CCD row."""
    a = load_matrix(path)
    if a.ndim == 2 and a.shape[1] == 2:                       # (wavelength, counts)
        x, y = a[:, 0], a[:, 1]
        o = np.argsort(x)
        return np.interp(wl, x[o], y[o]), "2-column"
    if a.shape[0] == len(wl):                                  # 2048 x NROWS matrix
        cols = a[:, r0:r1 + 1] if a.shape[1] > r1 else a
        return np.median(cols, 1) if cols.shape[1] > 2 else cols.mean(1), f"matrix ({a.shape[1]} rows)"
    if a.shape[1] == len(wl):
        cols = a[r0:r1 + 1] if a.shape[0] > r1 else a
        return np.median(cols, 0) if cols.shape[0] > 2 else cols.mean(0), f"matrix ({a.shape[0]} rows)"
    sys.exit(f"Don't understand the format of {path} (shape {a.shape})")


def smooth(y, n=7):
    k = np.ones(n) / n
    return np.convolve(np.pad(y, n, mode="edge"), k, "same")[n:-n]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--r0", help="substrate reference spectrum (see above); default: substrate pixels of the map")
    ap.add_argument("--dark", help="dark spectrum file (same exposure)")
    ap.add_argument("--offset", type=float, default=815.0, help="flat CCD offset if no --dark (default 815)")
    ap.add_argument("--exp-map", type=float, default=1.0, help="map exposure (s)")
    ap.add_argument("--exp-r0", type=float, default=None, help="R0 exposure (s); default = --exp-map")
    ap.add_argument("--r0-rows", type=int, default=1, help="R0 is a sum over this many CCD rows (divide by it)")
    ap.add_argument("--normalize-substrate", action="store_true",
                    help="rescale R0 so RC = 0 on average over the map's substrate pixels")
    ap.add_argument("--nx", type=int); ap.add_argument("--ny", type=int)
    ap.add_argument("--step", type=float, default=1.0, help="step size in um")
    ap.add_argument("--rows", type=int, nargs=2, metavar=("R0", "R1"),
                    help="CCD rows to use (inclusive). Default: the rows where the lamp image is >50%% of its peak")
    ap.add_argument("--flip-lr", action="store_true", help="display LR-flipped (real space)")
    ap.add_argument("--x0-band", type=float, nargs=2, default=[706.0, 722.0],
                    help="wavelength window (nm) containing the X0 resonance (default 706-722)")
    ap.add_argument("--axis1", type=float, nargs=2, metavar=("START", "END"), help="old fast-axis limits (mm)")
    ap.add_argument("--axis2", type=float, nargs=2, metavar=("START", "END"), help="old slow-axis limits (mm)")
    ap.add_argument("--swap-axes", action="store_true", help="axis2 was the fast axis")
    ap.add_argument("--new-size", type=float, default=20.0); ap.add_argument("--new-step", type=float, default=0.5)
    ap.add_argument("--thresh", type=float, default=0.25,
                    help="flake-extent threshold, fraction of max X0 RC amplitude (default 0.25)")
    a = ap.parse_args()

    folder = os.path.abspath(a.folder)
    wl = find_xvector(folder); E = HC / wl
    nfiles = len(glob.glob(os.path.join(folder, "1_loc*.txt")))
    nx = a.nx or int(round(np.sqrt(nfiles))); ny = a.ny or int(round(nfiles / nx))
    cube = load_cube(folder, nx * ny); nrow = cube.shape[1]
    have = ~np.isnan(cube[:, 0, 0])

    # --- CCD rows: where the reflected lamp image is
    if a.rows:
        r0, r1 = a.rows
    else:
        prof = np.nanmedian(np.nanmean(cube[have], 2), 0) - (a.offset if not a.dark else 0)
        if a.dark is None: prof = prof - np.min(prof)
        on = np.where(prof > 0.5 * prof.max())[0]
        r0, r1 = int(on.min()), int(on.max())
        print("CCD-row lamp profile:", np.round(prof / prof.max(), 2))
    print(f"using CCD rows {r0}-{r1}")

    dark = load_spectrum(a.dark, wl, r0, r1)[0] if a.dark else np.full(len(wl), a.offset)
    exp_r0 = a.exp_r0 or a.exp_map
    R = (combine_rows(cube, r0, r1) - dark) / a.exp_map            # counts/s per CCD row
    Rtot = np.nanmean(R, 1)                                        # reflected intensity map

    # --- R0
    if a.r0:
        R0raw, fmt = load_spectrum(a.r0, wl, r0, r1)
        R0 = (R0raw / a.r0_rows - dark) / exp_r0
        print(f"R0 from {os.path.basename(a.r0)} [{fmt}]")
    else:
        R0 = None; print("R0 from substrate pixels of the map")

    # substrate pixels: X0 resonance feature weakest. Use spectral 'roughness' in the X0 window
    w = (wl > a.x0_band[0]) & (wl < a.x0_band[1])
    ref_guess = R0 if R0 is not None else np.nanmedian(R[have], 0)
    with np.errstate(all="ignore"):
        rc_guess = R / ref_guess - 1
        feat = np.nanmax(smooth_rows(rc_guess[:, w]), 1) - np.nanmin(smooth_rows(rc_guess[:, w]), 1)
        lvl = np.abs(np.nanmedian(rc_guess[:, w], 1) - np.nanmedian(rc_guess[:, w]))
    sub = have & (feat < np.nanpercentile(feat[have], 40)) & (lvl < np.nanpercentile(lvl[have], 60))
    if R0 is None:
        R0 = np.nanmedian(R[sub], 0)
    if a.normalize_substrate:
        with np.errstate(all="ignore"):
            k = np.nanmedian(R[sub] / R0, 0)
        k = smooth(np.nan_to_num(k, nan=1.0), 51)
        print(f"substrate normalisation: R0 scaled by {np.median(k):.3f} (range {k.min():.3f}-{k.max():.3f})")
        R0 = R0 * k
    with np.errstate(all="ignore"):
        RC = R / R0 - 1
    subrc = np.nanmedian(RC[sub][:, w])
    print(f"median RC over substrate pixels in X0 window: {subrc:+.4f} (should be ~0)")
    if abs(subrc) > 0.05:
        print(f"  WARNING: substrate RC is {subrc:+.3f}, not ~0. R0 and the map are on different scales\n"
              f"  (different exposure, ybin/CCD rows, lamp drift or focus). R0 looks off by a factor {1+subrc:.3f}.\n"
              f"  Fix: --exp-map/--exp-r0, --r0-rows, the same --rows as R0's ybin, or simply --normalize-substrate.")

    # --- maps
    RCs = smooth_rows(RC[:, w], 5)
    with np.errstate(all="ignore"):
        amp = np.nanmax(RCs, 1) - np.nanmin(RCs, 1)                # peak-to-peak in X0 window
        jmin = np.nanargmin(np.nan_to_num(RCs, nan=np.inf), 1); Emin = E[w][jmin] * 1e3
        base = np.nanmedian(RC[:, w], 1)
    amp[~have] = np.nan; Emin[~have] = np.nan; base[~have] = np.nan
    Emin[amp < 0.3 * np.nanmax(amp)] = np.nan
    maps = [(Rtot, "Reflected intensity (counts/s/row)", "gray"),
            (amp, f"X$^0$ RC peak-to-peak ({a.x0_band[0]:g}–{a.x0_band[1]:g} nm)", "magma"),
            (Emin, "Energy of RC minimum (meV)", "coolwarm"),
            (base, "Median RC in window (background level)", "RdBu_r")]

    # --- extent and new window
    Am = amp.reshape(ny, nx)
    from numpy.lib.stride_tricks import sliding_window_view
    As = np.nanmedian(sliding_window_view(np.pad(Am, 1, mode="edge"), (3, 3)), axis=(2, 3))
    onm = As > a.thresh * np.nanmax(As)
    rr, cc = np.where(onm)
    r_lo, r_hi, c_lo, c_hi = rr.min(), rr.max(), cc.min(), cc.max()
    rc, cc_ = (r_lo + r_hi) / 2, (c_lo + c_hi) / 2
    print(f"\nX0-RC extent (>{a.thresh:.0%} of max): rows {r_lo}-{r_hi}, cols {c_lo}-{c_hi} (stored indices)"
          f" = {(r_hi-r_lo+1)*a.step:.1f} x {(c_hi-c_lo+1)*a.step:.1f} um (slow x fast)")
    print(f"suggested new-scan centre: row {rc:.1f}, col {cc_:.1f}")
    if onm[0].any() or onm[-1].any() or onm[:, 0].any() or onm[:, -1].any():
        print("  WARNING: RC feature touches the map edge -- flake may extend outside the scan")
    if a.axis1 and a.axis2:
        fast, slow = (a.axis2, a.axis1) if a.swap_axes else (a.axis1, a.axis2)
        f_mm = fast[0] + cc_ / (nx - 1) * (fast[1] - fast[0]); s_mm = slow[0] + rc / (ny - 1) * (slow[1] - slow[0])
        h = a.new_size / 2 / 1000; n = int(round(a.new_size / a.new_step))
        sf = np.sign(fast[1] - fast[0]) or 1; ss = np.sign(slow[1] - slow[0]) or 1
        nf, ns = ("axis2", "axis1") if a.swap_axes else ("axis1", "axis2")
        print(f"old step from limits: fast {abs(fast[1]-fast[0])/(nx-1)*1000:.3f} um, slow {abs(slow[1]-slow[0])/(ny-1)*1000:.3f} um")
        print(f"NEW SCAN ({a.new_size:g} x {a.new_size:g} um, {a.new_step:g} um steps = {n} steps / {n+1} points per axis):")
        print(f"  {nf} (fast): {f_mm - sf*h:.5f} to {f_mm + sf*h:.5f} mm")
        print(f"  {ns} (slow): {s_mm - ss*h:.5f} to {s_mm + ss*h:.5f} mm")

    # --- figure
    ext = [-0.5 * a.step, (nx - 0.5) * a.step, (ny - 0.5) * a.step, -0.5 * a.step]
    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 0.85], hspace=0.38, wspace=0.38,
                          left=0.05, right=0.97, top=0.90, bottom=0.09)
    pB = int(np.nanargmax(np.nan_to_num(amp, nan=-1)))
    pS = int(np.where(sub)[0][len(np.where(sub)[0]) // 2])
    for k, (M, title, cm) in enumerate(maps):
        ax = fig.add_subplot(gs[0, k]); Mm = M.reshape(ny, nx); ax.set_facecolor("0.85")
        if cm == "RdBu_r":
            v = np.nanpercentile(np.abs(Mm), 98); lo, hi = -v, v
        else:
            lo, hi = np.nanpercentile(Mm, 1), np.nanpercentile(Mm, 99.5)
        im = ax.imshow(Mm, cmap=cm, vmin=lo, vmax=hi, extent=ext, interpolation="nearest")
        if a.flip_lr: ax.invert_xaxis()
        lc = "w" if cm in ("magma", "gray") else "k"
        ax.add_patch(Rectangle((cc_ * a.step - a.new_size / 2, rc * a.step - a.new_size / 2), a.new_size, a.new_size,
                               fill=False, ec=lc, ls="--", lw=0.9))
        for p, col in [(pB, "#d62728"), (pS, "#b8860b")]:
            r_, c_ = divmod(p, nx); ax.plot(c_ * a.step, r_ * a.step, "o", mfc="none", mec=col, ms=7, mew=1.5)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.ax.tick_params(labelsize=7)
        ax.set_title(title, fontsize=8.5); ax.tick_params(labelsize=7)
        ax.set_xlabel("fast axis (µm)" + (" ← flipped" if a.flip_lr else ""), fontsize=8)
        if k == 0: ax.set_ylabel("slow axis (µm)", fontsize=8)
    ax = fig.add_subplot(gs[1, :2])
    for p, col, lab in [(pB, "#d62728", "strongest X$^0$ RC"), (pS, "#b8860b", "substrate pixel")]:
        r_, c_ = divmod(p, nx)
        ax.plot(E, RC[p], color=col, lw=0.5, alpha=0.35); ax.plot(E, smooth(RC[p], 7), color=col, lw=1.4,
                                                                  label=f"{lab} (row {r_}, col {c_})")
    ax.axhline(0, color="0.5", lw=0.5, ls=":"); ax.axvspan(HC / a.x0_band[1], HC / a.x0_band[0], color="0.9", zorder=0)
    ax.set_xlim(E.min(), E.max()); ax.set_xlabel("Energy (eV)"); ax.set_ylabel("RC = (R − R$_0$)/R$_0$")
    ax.legend(fontsize=8, frameon=False)
    ax.secondary_xaxis("top", functions=(lambda e: HC / np.clip(e, 1e-3, None), lambda l: HC / np.clip(l, 1e-3, None))).set_xlabel("Wavelength (nm)")
    ax2 = fig.add_subplot(gs[1, 2:])
    ax2.plot(E, R0, color="k", lw=1, label="R$_0$ used"); ax2.plot(E, R[pB], color="#d62728", lw=0.8, label="R at strongest pixel")
    ax2.set_xlim(E.min(), E.max()); ax2.set_xlabel("Energy (eV)"); ax2.set_ylabel("counts/s per CCD row (dark-subtracted)")
    ax2.legend(fontsize=8, frameon=False)
    fig.suptitle(f"{os.path.basename(folder)} — RC map, {nx}×{ny} pts, {a.step:g} µm steps, CCD rows {r0}–{r1}"
                 + (" — display LR-flipped" if a.flip_lr else ""), fontsize=11)
    fig.text(0.5, 0.01, f"Dashed box: suggested {a.new_size:g}×{a.new_size:g} µm scan around the X$^0$ RC feature. "
             "Row/col indices are as stored (file order). Grey shading: X$^0$ window.", ha="center", fontsize=8, color="0.3")
    out = os.path.join(folder, datetime.date.today().strftime("%Y.%m.%d") + "_RC_quicklook")
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, "RCmap_quicklook.png"), dpi=200); fig.savefig(os.path.join(out, "RCmap_quicklook.pdf"))
    np.savez_compressed(os.path.join(out, "rc_cube.npz"), RC=RC.astype(np.float32), R=R.astype(np.float32), R0=R0, wl=wl,
                        nx=nx, ny=ny, step=a.step, rows=np.array([r0, r1]))
    print("\nsaved to", out)


def smooth_rows(M, n=7):
    k = np.ones(n) / n
    P = np.pad(np.nan_to_num(M), ((0, 0), (n, n)), mode="edge")
    out = np.apply_along_axis(lambda r: np.convolve(r, k, "same"), 1, P)[:, n:-n]
    out[np.isnan(M)] = np.nan
    return out


if __name__ == "__main__":
    main()
