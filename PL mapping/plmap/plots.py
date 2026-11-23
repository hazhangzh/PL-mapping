"""Static figures: one spectrum plot per loc file, per-file fit plots, summary maps."""
import os

import numpy as np

from . import HC


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _energy_axis(ax):
    sec = ax.secondary_xaxis("top", functions=(lambda l: HC / np.clip(l, 1e-3, None),
                                               lambda e: HC / np.clip(e, 1e-3, None)))
    sec.set_xlabel("Energy (eV)", fontsize=8)
    sec.tick_params(labelsize=7)


def spectrum_plots(outdir, wl, spectra, ks, grid, prefix, files, dpi=100, log=print):
    """Step 1 output: averaged (raw) spectrum of every selected loc file."""
    if not len(ks):
        return
    plt = _mpl()
    os.makedirs(outdir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    fig.subplots_adjust(left=0.13, right=0.97, top=0.76, bottom=0.14)
    line, = ax.plot(wl, wl * 0, lw=0.8, color="#1f4e8c")
    ax.set_xlim(wl.min(), wl.max())
    ax.set_xlabel("Wavelength (nm)", fontsize=9)
    ax.set_ylabel("Counts (mean over CCD rows)", fontsize=9)
    ax.tick_params(labelsize=8)
    _energy_axis(ax)
    title = fig.text(0.5, 0.965, "", ha="center", va="top", fontsize=9)
    n_done = n_skip = 0
    for k in ks:
        if k not in files:
            continue
        png = os.path.join(outdir, f"{prefix}_loc{k}.png")
        if os.path.exists(png) and os.path.getmtime(png) > os.path.getmtime(files[k]):
            n_skip += 1
            continue
        y = spectra[k - 1]
        line.set_ydata(y)
        lo, hi = np.nanmin(y), np.nanmax(y)
        pad = 0.05 * (hi - lo) if hi > lo else 1
        ax.set_ylim(lo - pad, hi + pad)
        r, c = grid.rowcol(k)
        x, yy = grid.stage_xy(k)
        title.set_text(f"{prefix}_loc{k}   row {r}, col {c}   x = {x:.5g}, y = {yy:.5g} {grid.units}")
        fig.savefig(png, dpi=dpi)
        n_done += 1
        if n_done % 200 == 0:
            log(f"  plotted {n_done}")
    plt.close(fig)
    log(f"spectrum plots: {n_done} written, {n_skip} up to date -> {outdir}")


def fit_plots(outdir, wl, spc, fit, ks, grid, prefix, dpi=100, log=print):
    """Optional: data + fitted components for selected loc files."""
    if not len(ks):
        return
    plt = _mpl()
    os.makedirs(outdir, exist_ok=True)
    m = fit["model"]
    res = fit["res"]
    for k in ks:
        i = k - 1
        th = fit["theta"][i]
        if not np.all(np.isfinite(th)):
            continue
        fm = (wl >= res["range_lo_nm"][i] - 1e-9) & (wl <= res["range_hi_nm"][i] + 1e-9)   # this point's range
        E = HC / wl[fm]
        fig, (ax, axr) = plt.subplots(2, 1, figsize=(6.4, 4.4), sharex=True,
                                      gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05})
        y = spc[i, fm]
        ax.plot(wl[fm], y, lw=0.6, color="0.4", label="data (bkg-subtracted)")
        base = m.base(E, th)
        for p, c in zip(m.peaks, m.components(E, th)):
            ax.plot(wl[fm], c + base, lw=1, label=f"{p['name']} ({p['shape']})")
        ax.plot(wl[fm], m(E, *th), lw=1.2, color="k", label="fit")
        axr.plot(wl[fm], y - m(E, *th), lw=0.5, color="0.3")
        axr.axhline(0, color="0.6", lw=0.5)
        ax.legend(fontsize=7, frameon=False)
        r, c = grid.rowcol(k)
        ax.set_title(f"{prefix}_loc{k}  (row {r}, col {c}), shift {res['shift_meV'][i]:+.1f} meV", fontsize=9)
        ax.set_ylabel("Counts", fontsize=8)
        axr.set_ylabel("resid.", fontsize=8)
        axr.set_xlabel("Wavelength (nm)", fontsize=8)
        fig.savefig(os.path.join(outdir, f"{prefix}_loc{k}_fit.png"), dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    log(f"fit plots -> {outdir}")


def summary(path_base, wl, grid, fit, flip_x=False, title=""):
    """PNG + PDF: integrated-PL map, mean spectrum fit, and per-peak maps."""
    plt = _mpl()
    peaks, res = fit["peaks"], fit["res"]
    ny, nx = grid.ny, grid.nx
    ext = _extent(grid)
    combos = fit.get("combined", [])
    nrow = 1 + len(peaks) + (len(combos) + 2) // 3
    fig = plt.figure(figsize=(12, 3.4 * nrow + 0.6), layout="constrained")
    gs = fig.add_gridspec(nrow, 3)

    def show(ax, v, label, cmap, mask=None):
        M = v.copy()
        if mask is not None:
            M[~mask] = np.nan
        M = M.reshape(ny, nx)
        ax.set_facecolor("0.85")
        fin = M[np.isfinite(M)]
        lo, hi = (np.percentile(fin, 1), np.percentile(fin, 99.5)) if fin.size else (0, 1)
        im = ax.imshow(M, cmap=cmap, vmin=lo, vmax=hi, extent=ext, interpolation="nearest", aspect="equal")
        if flip_x:
            ax.invert_xaxis()
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=7)
        ax.set_title(label, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.set_xlabel("x (µm)" + (" ← flipped" if flip_x else ""), fontsize=8)
        ax.set_ylabel("y (µm)", fontsize=8)

    show(fig.add_subplot(gs[0, 0]), res["integrated_counts"], "Integrated PL in fit range (counts)", "magma")
    ax = fig.add_subplot(gs[0, 1:])
    m, fm = fit["model"], fit["fit_mask"]
    E = HC / wl[fm]
    th, y = fit["mean"]["theta"], fit["mean"]["y"]
    ax.plot(wl[fm], y, color="0.5", lw=0.7, label=f"mean of {len(fit['mean']['points'])} brightest points")
    base = m.base(E, th)
    for p, c in zip(m.peaks, m.components(E, th)):
        ax.fill_between(wl[fm], base, c + base, alpha=0.35, label=f"{p['name']} ({p['shape']})")
    ax.plot(wl[fm], m(E, *th), color="k", lw=1.1, label="fit")
    ax.set_xlabel("Wavelength (nm)", fontsize=8)
    ax.set_ylabel("Counts (bkg-subtracted)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, frameon=False)
    _energy_axis(ax)
    for j, p in enumerate(peaks, 1):
        n = p["name"]
        ok = res[f"{n}_ok"]
        show(fig.add_subplot(gs[j, 0]), res[f"{n}_intensity"], f"{n} integrated intensity (counts)", "magma", ok)
        show(fig.add_subplot(gs[j, 1]), res[f"{n}_center_meV"], f"{n} peak energy (meV)", "coolwarm", ok)
        show(fig.add_subplot(gs[j, 2]), res[f"{n}_fwhm_meV"], f"{n} FWHM (meV)", "viridis", ok)
    for j, c in enumerate(combos):
        show(fig.add_subplot(gs[1 + len(peaks) + j // 3, j % 3]), res[c["name"]], c["label"],
             "viridis" if c["kind"] == "ratio" else "coolwarm", res[c["name"] + "_ok"])
    fig.suptitle(title, fontsize=11)
    fig.supxlabel("Grey = no data, or the fit failed the quality check (SNR, parameter at a bound).",
                  fontsize=8, color="0.3")
    fig.savefig(path_base + ".png", dpi=200)
    fig.savefig(path_base + ".pdf")
    plt.close(fig)


def overview(path, wl, spc, grid, flip_x=False, title=""):
    """Step 2 output: integrated PL map (after offset/substrate subtraction) + mean spectrum."""
    plt = _mpl()
    tot = np.nansum(spc, 1)
    tot[np.isnan(spc[:, 0])] = np.nan
    M = tot.reshape(grid.ny, grid.nx)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1, 1.3]})
    fig.subplots_adjust(left=0.07, right=0.97, top=0.80, bottom=0.13, wspace=0.3)
    ax.set_facecolor("0.85")
    fin = M[np.isfinite(M)]
    im = ax.imshow(M, cmap="magma", extent=_extent(grid), interpolation="nearest",
                   vmin=np.percentile(fin, 1) if fin.size else None, vmax=np.percentile(fin, 99.5) if fin.size else None)
    if flip_x:
        ax.invert_xaxis()
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(labelsize=7)
    ax.set_title("Integrated PL, whole spectrum (counts)", fontsize=9)
    ax.set_xlabel("x (µm)" + (" ← flipped" if flip_x else ""), fontsize=8)
    ax.set_ylabel("y (µm)", fontsize=8)
    ax.tick_params(labelsize=7)
    from .background import bright_points
    fin_pts = np.isfinite(tot)
    br = bright_points(tot, fin_pts)
    order = np.argsort(np.where(fin_pts, tot, np.inf))
    dim = order[:max(len(br), int(0.3 * fin_pts.sum()))]
    ax2.plot(wl, np.nanmean(spc[br], 0), color="#d62728", lw=0.9, label=f"mean of {len(br)} brightest points")
    ax2.plot(wl, np.nanmean(spc[dim], 0), color="#b8860b", lw=0.9, label=f"mean of {len(dim)} dimmest points")
    ax2.axhline(0, color="0.6", lw=0.5)
    ax2.set_xlim(wl.min(), wl.max())
    ax2.set_xlabel("Wavelength (nm)", fontsize=8)
    ax2.set_ylabel("Counts (background-subtracted)", fontsize=8)
    ax2.tick_params(labelsize=7)
    ax2.legend(fontsize=7, frameon=False)
    _energy_axis(ax2)
    fig.suptitle(title, fontsize=10)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _extent(grid):
    """imshow extent in µm relative to the scan start (row 0 drawn at the top)."""
    dx, dy = grid.x_step * grid.to_um or 1.0, grid.y_step * grid.to_um or 1.0
    return [-0.5 * dx, (grid.nx - 0.5) * dx, (grid.ny - 0.5) * dy, -0.5 * dy]
