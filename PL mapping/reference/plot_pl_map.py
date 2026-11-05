#!/usr/bin/env python3
"""
plot_pl_map.py -- quick-look plotter for Andor PL maps (WSe2, Cundiff group)

Needs only numpy + matplotlib (no scipy).

Data format expected in the map folder:
    1_loc1.txt ... 1_locN.txt    each a 2048 x NROWS text matrix
                                 (spectral pixel x CCD row), raster in
                                 row-major order: loc1..locNX = first row
    x_vector*.txt                 wavelength axis (2048 values, nm)
A "--" in a file (dropped pixel) is handled. Missing files (scan still
running) are shown as blank pixels.

Typical use
-----------
  # quick look, 41x41 map, 1 um steps, display flipped LR to real space
  python3 plot_pl_map.py /path/to/map_folder --nx 41 --ny 41 --step 1.0 --flip-lr

  # also compute new scan limits (20 x 20 um, 0.5 um steps) from the old ones
  python3 plot_pl_map.py /path/to/map_folder --nx 41 --ny 41 --step 1.0 --flip-lr \
      --axis1 0.2600 0.3000 --axis2 0.2500 0.2900 --new-size 20 --new-step 0.5

  axis1 = fast axis (map columns), axis2 = slow axis (map rows), in mm
  (start end), exactly as entered for the scan. Swap with --swap-axes if
  axis2 was the fast one.

Output: <map_folder>/<YYYY.MM.DD>_quicklook/ with
    PLmap_quicklook.png / .pdf   maps + spectra
    pl_cube_rowsXX-YY.npz        processed spectra (for re-plotting)
and a printout of the PL extent and suggested new scan window.
"""
import argparse, glob, os, re, sys, datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HC = 1239.84193  # eV nm


# ----------------------------------------------------------------- loading
def find_xvector(folder):
    c = sorted(glob.glob(os.path.join(folder, "x_vector*.txt")))
    if not c:
        sys.exit("No x_vector*.txt (wavelength axis) found in " + folder)
    return np.loadtxt(c[0])


def load_file(path):
    txt = open(path).read()
    if "--" in txt:                                   # dropped pixel(s)
        a = np.genfromtxt(path, missing_values="--", filling_values=np.nan)
        for j in range(a.shape[1]):                   # interpolate per CCD row
            col = a[:, j]; bad = np.isnan(col)
            if bad.any() and (~bad).any():
                col[bad] = np.interp(np.where(bad)[0], np.where(~bad)[0], col[~bad])
        return a
    rows = [r.split() for r in txt.strip().splitlines()]
    return np.array(rows, dtype=np.float32)


def load_cube(folder, npts):
    files = {int(m.group(1)): f for f in glob.glob(os.path.join(folder, "1_loc*.txt"))
             if (m := re.search(r"1_loc(\d+)\.txt$", f))}
    if not files:
        sys.exit("No 1_loc*.txt files found in " + folder)
    first = load_file(files[min(files)])
    npix, nrow = first.shape
    cube = np.full((npts, nrow, npix), np.nan, np.float32)
    for k in range(1, npts + 1):
        if k in files:
            cube[k - 1] = load_file(files[k]).T
    print(f"loaded {len(files)} of {npts} files; each {npix} px x {nrow} CCD rows")
    return cube


# ------------------------------------------------------------- processing
def combine_rows(cube, r0, r1, nsig=6.0):
    """Cosmic-ray-robust mean over CCD rows r0..r1 (inclusive).
    The CR mask is built from ALL rows (robust even for narrow windows)."""
    med = np.nanmedian(cube, 1, keepdims=True)
    mad = np.nanmedian(np.abs(cube - med), 1, keepdims=True) * 1.4826
    cr = cube > med + nsig * np.maximum(mad, 1.5)
    cr = cr | np.roll(cr, 1, 1) | np.roll(cr, -1, 1) | np.roll(cr, 1, 2) | np.roll(cr, -1, 2)
    sub = np.where(cr, np.nan, cube)[:, r0:r1 + 1]
    with np.errstate(all="ignore"):
        sp = np.nanmean(sub, 1)
    fill = np.isnan(sp) & ~np.isnan(cube[:, 0, :])
    sp[fill] = np.nanmedian(cube[:, r0:r1 + 1], 1)[fill]
    return sp


def despike(sp, k=7, nsig=8.0):
    from numpy.lib.stride_tricks import sliding_window_view
    pad = np.pad(sp, ((0, 0), (k, k)), mode="edge")
    rm = np.median(sliding_window_view(pad, 2 * k + 1, axis=1), axis=2)
    r = sp - rm
    s = 1.4826 * np.nanmedian(np.abs(r), 1, keepdims=True)
    return np.where(r > nsig * np.maximum(s, 0.3), rm, sp)


def lorentz_grid_fit(E, Y, e_lo, e_hi, g_lo=0.0015, g_hi=0.025):
    """Lorentzian + linear background, grid search in (E0, FWHM), linear LSQ
    for amplitude/offset/slope. E in eV. Returns E0, FWHM, amp, snr."""
    E0g = np.arange(e_lo, e_hi, 0.0002); Gg = np.arange(g_lo, g_hi, 0.0005)
    ok = ~np.isnan(Y).any(1)
    Yv = np.nan_to_num(Y); x = E - E.mean(); yy = (Yv ** 2).sum(1)
    best = np.full(len(Y), np.inf); bp = np.zeros((len(Y), 3))
    for e0 in E0g:
        for g in Gg:
            L = (g / 2) ** 2 / ((E - e0) ** 2 + (g / 2) ** 2)
            A = np.stack([L, np.ones_like(E), x], 1)
            Ai = np.linalg.inv(A.T @ A); Q = Yv @ A; cf = Q @ Ai
            rss = yy - np.einsum("ij,ij->i", cf, Q)
            b = rss < best
            best[b] = rss[b]; bp[b] = np.c_[np.full(b.sum(), e0), np.full(b.sum(), g), cf[b, 0]]
    snr = bp[:, 2] / np.sqrt(np.maximum(best, 1e-12) / (len(E) - 5))
    edge = (bp[:, 0] < E0g[1]) | (bp[:, 0] > E0g[-2]) | (bp[:, 1] > Gg[-2]) | (bp[:, 1] < Gg[1])
    good = ok & (snr > 5) & ~edge
    return np.where(good, bp[:, 0], np.nan), np.where(good, bp[:, 1], np.nan), bp[:, 2], snr


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help="folder with 1_loc*.txt and x_vector*.txt")
    ap.add_argument("--nx", type=int, help="points per row (fast axis); default sqrt(N)")
    ap.add_argument("--ny", type=int, help="number of rows (slow axis); default sqrt(N)")
    ap.add_argument("--step", type=float, default=1.0, help="step size in um (default 1.0)")
    ap.add_argument("--rows", type=int, nargs=2, metavar=("R0", "R1"),
                    help="CCD rows to average (inclusive). Default: 15 rows centred on the signal")
    ap.add_argument("--flip-lr", action="store_true", help="display LR-flipped (real space)")
    ap.add_argument("--x0-band", type=float, nargs=2, default=[708.5, 718.5], help="X0 integration band (nm)")
    ap.add_argument("--le-band", type=float, nargs=2, default=[719.5, 750.0], help="low-energy band (nm)")
    ap.add_argument("--no-fit", action="store_true", help="skip X0 Lorentzian fits (faster)")
    ap.add_argument("--axis1", type=float, nargs=2, metavar=("START", "END"), help="old fast-axis limits (mm)")
    ap.add_argument("--axis2", type=float, nargs=2, metavar=("START", "END"), help="old slow-axis limits (mm)")
    ap.add_argument("--swap-axes", action="store_true", help="axis2 was the fast axis (columns)")
    ap.add_argument("--new-size", type=float, default=20.0, help="new scan size in um (default 20)")
    ap.add_argument("--new-step", type=float, default=0.5, help="new scan step in um (default 0.5)")
    ap.add_argument("--thresh", type=float, default=0.10, help="PL-extent threshold, fraction of max of the 3x3-median-filtered map (default 0.10)")
    a = ap.parse_args()

    folder = os.path.abspath(a.folder)
    wl = find_xvector(folder); E = HC / wl
    nfiles = len(glob.glob(os.path.join(folder, "1_loc*.txt")))
    nx = a.nx or int(round(np.sqrt(nfiles))); ny = a.ny or int(round(nfiles / nx))
    if nx * ny < nfiles:
        print(f"WARNING: {nfiles} files but nx*ny = {nx*ny}; set --nx/--ny")
    cube = load_cube(folder, nx * ny)
    nrow = cube.shape[1]

    # --- CCD row window: centre on signal unless given
    off = wl > wl.max() - 8                                # baseline region (red end)
    sig = (wl > a.x0_band[0]) & (wl < a.le_band[1])
    tot = np.nanmean(cube[:, :, sig], (1, 2)) - np.nanmean(cube[:, :, off], (1, 2))
    if a.rows:
        r0, r1 = a.rows
    else:
        bright = np.argsort(np.nan_to_num(tot, nan=-np.inf))[-max(5, nx * ny // 50):]
        dark = np.argsort(np.nan_to_num(tot, nan=np.inf))[: nx * ny // 2]
        prof = (np.nanmean(cube[bright][:, :, sig], (0, 2)) - np.nanmean(cube[bright][:, :, off], (0, 2))) \
             - (np.nanmean(cube[dark][:, :, sig], (0, 2)) - np.nanmean(cube[dark][:, :, off], (0, 2)))
        c = int(np.argmax(np.convolve(prof, np.ones(3) / 3, mode="same")))
        r0, r1 = max(0, c - 7), min(nrow - 1, c + 7)
        print("CCD-row signal profile:", np.round(prof, 1))
    print(f"using CCD rows {r0}-{r1}")

    sp = combine_rows(cube, r0, r1)
    sp = sp - np.nanmean(sp[:, off], 1, keepdims=True)    # CCD offset + drift
    sp = despike(np.nan_to_num(sp)) * np.where(np.isnan(sp[:, :1]), np.nan, 1)
    T = np.nansum(sp[:, sig], 1)
    T[np.isnan(sp[:, 0])] = np.nan
    offflake = T < np.nanpercentile(T, 50)
    bg = np.nanmean(sp[offflake], 0)                       # broad substrate background
    spc = sp - bg

    band = lambda b: (wl > b[0]) & (wl < b[1])
    IX0 = np.nansum(spc[:, band(a.x0_band)], 1); ILE = np.nansum(spc[:, band(a.le_band)], 1)
    IX0[np.isnan(sp[:, 0])] = np.nan; ILE[np.isnan(sp[:, 0])] = np.nan
    maps = [(IX0, f"X$^0$ integrated ({a.x0_band[0]:g}–{a.x0_band[1]:g} nm)", "magma"),
            (ILE, f"Low-energy band ({a.le_band[0]:g}–{a.le_band[1]:g} nm)", "magma")]
    if not a.no_fit:
        fw = (wl > a.x0_band[0] - 3.5) & (wl < a.x0_band[1] + 2.5)
        idx = np.where(fw)[0]; m = (len(idx) // 4) * 4; idx = idx[:m]
        Eb = E[idx].reshape(-1, 4).mean(1); Yb = spc[:, idx].reshape(len(spc), -1, 4).mean(2)
        e_lo, e_hi = HC / a.x0_band[1], HC / a.x0_band[0]
        print("fitting X0 (Lorentzian + linear bkg) ...")
        E0, G, amp, snr = lorentz_grid_fit(Eb, Yb, e_lo, e_hi)
        print(f"  {np.isfinite(E0).sum()} pixels with good fits (SNR>5)")
        maps += [(E0 * 1e3, "X$^0$ peak energy (meV)", "coolwarm"), (G * 1e3, "X$^0$ FWHM (meV)", "viridis")]

    # --- PL extent and new scan window (stored indices)
    Tm = (IX0 + ILE).reshape(ny, nx)
    from numpy.lib.stride_tricks import sliding_window_view
    Ts = np.nanmedian(sliding_window_view(np.pad(Tm, 1, mode="edge"), (3, 3)), axis=(2, 3))  # 3x3 median: drops isolated/noise pixels
    on = Ts > a.thresh * np.nanmax(Ts)
    rr, cc = np.where(on)
    r_lo, r_hi, c_lo, c_hi = rr.min(), rr.max(), cc.min(), cc.max()
    rc, cc_ = (r_lo + r_hi) / 2, (c_lo + c_hi) / 2
    print(f"\nPL extent (>{a.thresh:.0%} of max): rows {r_lo}-{r_hi}, cols {c_lo}-{c_hi} (stored indices)")
    print(f"  = {(r_hi-r_lo+1)*a.step:.1f} um (slow) x {(c_hi-c_lo+1)*a.step:.1f} um (fast)")
    print(f"suggested new-scan centre: row {rc:.1f}, col {cc_:.1f} "
          f"(offset from old centre: slow {(rc-(ny-1)/2)*a.step:+.2f} um, fast {(cc_-(nx-1)/2)*a.step:+.2f} um)")
    if (r_hi - r_lo + 1) * a.step > a.new_size or (c_hi - c_lo + 1) * a.step > a.new_size:
        print(f"  WARNING: PL extent is larger than the {a.new_size:g} um window")
    edge_hit = on[0].any() or on[-1].any() or on[:, 0].any() or on[:, -1].any()
    if edge_hit:
        print("  WARNING: PL touches the edge of this map -- the flake may extend outside the scan")
    if a.axis1 and a.axis2:
        fast, slow = (a.axis2, a.axis1) if a.swap_axes else (a.axis1, a.axis2)
        f_mm = fast[0] + cc_ / (nx - 1) * (fast[1] - fast[0])
        s_mm = slow[0] + rc / (ny - 1) * (slow[1] - slow[0])
        h = a.new_size / 2 / 1000; nsteps = int(round(a.new_size / a.new_step))
        sf = np.sign(fast[1] - fast[0]) or 1; ss = np.sign(slow[1] - slow[0]) or 1
        print(f"\nOld step from limits: fast {abs(fast[1]-fast[0])/(nx-1)*1000:.3f} um, "
              f"slow {abs(slow[1]-slow[0])/(ny-1)*1000:.3f} um  (check vs --step {a.step})")
        nf, ns = ("axis2", "axis1") if a.swap_axes else ("axis1", "axis2")
        print(f"NEW SCAN ({a.new_size:g} x {a.new_size:g} um, {a.new_step:g} um steps = {nsteps} steps / {nsteps+1} points per axis):")
        print(f"  {nf} (fast): {f_mm - sf*h:.5f} to {f_mm + sf*h:.5f} mm")
        print(f"  {ns} (slow): {s_mm - ss*h:.5f} to {s_mm + ss*h:.5f} mm")

    # --- figure
    ext = [-0.5 * a.step, (nx - 0.5) * a.step, (ny - 0.5) * a.step, -0.5 * a.step]
    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, len(maps), height_ratios=[1, 0.85], hspace=0.38, wspace=0.38,
                          left=0.05, right=0.97, top=0.90, bottom=0.09)
    flat = np.nan_to_num(IX0 + ILE, nan=-np.inf)
    pB = int(np.argmax(flat)); pO = int(np.argmin(np.abs(np.nan_to_num(T, nan=1e9) - np.nanmedian(T))))
    for k, (M, title, cm) in enumerate(maps):
        ax = fig.add_subplot(gs[0, k]); Mm = M.reshape(ny, nx)
        lo, hi = (np.nanpercentile(Mm, 1), np.nanpercentile(Mm, 99.5)) if k < 2 else \
                 (np.nanpercentile(Mm, 2), np.nanpercentile(Mm, 98))
        ax.set_facecolor("0.85")
        im = ax.imshow(Mm, cmap=cm, vmin=lo, vmax=hi, extent=ext, interpolation="nearest")
        if a.flip_lr: ax.invert_xaxis()
        ax.add_patch(Rectangle((cc_ * a.step - a.new_size / 2, rc * a.step - a.new_size / 2), a.new_size, a.new_size,
                               fill=False, ec="w" if k < 2 else "k", ls="--", lw=0.9))
        ax.plot(cc_ * a.step, rc * a.step, "+", color="w" if k < 2 else "k")
        for p, col in [(pB, "#d62728"), (pO, "#b8860b")]:
            r_, c_ = divmod(p, nx); ax.plot(c_ * a.step, r_ * a.step, "o", mfc="none", mec=col, ms=7, mew=1.5)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.ax.tick_params(labelsize=7)
        ax.set_title(title, fontsize=9); ax.tick_params(labelsize=7)
        ax.set_xlabel("fast axis (µm)" + (" ← flipped" if a.flip_lr else ""), fontsize=8)
        if k == 0: ax.set_ylabel("slow axis (µm)", fontsize=8)
    ax = fig.add_subplot(gs[1, :])
    for p, col, lab in [(pB, "#d62728", "brightest pixel"), (pO, "#b8860b", "typical off-flake pixel")]:
        r_, c_ = divmod(p, nx)
        ax.plot(E, spc[p], color=col, lw=0.5, alpha=0.35)
        ax.plot(E, np.convolve(spc[p], np.ones(9) / 9, "same"), color=col, lw=1.4, label=f"{lab} (row {r_}, col {c_})")
    ax.axhline(0, color="0.5", lw=0.5, ls=":")
    ax.set_xlim(E.min(), E.max()); ax.set_xlabel("Energy (eV)"); ax.set_ylabel("Counts / CCD row / exposure (bkg-subtracted)")
    top = ax.secondary_xaxis("top", functions=(lambda e: HC / np.clip(e, 1e-3, None), lambda l: HC / np.clip(l, 1e-3, None)))
    top.set_xlabel("Wavelength (nm)")
    ax.legend(fontsize=8, frameon=False)
    fig.suptitle(f"{os.path.basename(folder)} — {nx}×{ny} pts, {a.step:g} µm steps, CCD rows {r0}–{r1}"
                 + (" — display LR-flipped" if a.flip_lr else ""), fontsize=11)
    fig.text(0.5, 0.005, f"Dashed box: suggested {a.new_size:g}×{a.new_size:g} µm scan around all PL (>{a.thresh:.0%} of max). "
             "Grey = no good X$^0$ fit. Row/col indices are as stored (file order).", ha="center", fontsize=8, color="0.3")

    out = os.path.join(folder, datetime.date.today().strftime("%Y.%m.%d") + "_quicklook")
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, "PLmap_quicklook.png"), dpi=200); fig.savefig(os.path.join(out, "PLmap_quicklook.pdf"))
    np.savez_compressed(os.path.join(out, f"pl_cube_rows{r0}-{r1}.npz"), spc=spc.astype(np.float32), bg=bg, wl=wl,
                        nx=nx, ny=ny, step=a.step, IX0=IX0, ILE=ILE)
    print("\nsaved to", out)


if __name__ == "__main__":
    main()
