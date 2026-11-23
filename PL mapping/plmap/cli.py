"""Command line interface.

    plmap init   [config.yaml]          write a commented config template (+ scan.txt next to it)
    plmap spectra config.yaml           PL steps 1-2: averaged spectra, per-file plots, overview map
    plmap run     config.yaml           PL steps 1-4: + background, peak fits, results, HTML report
    plmap rc      config.yaml           reflection-contrast map: RC = (R - R0)/R0, resonance fits, report
    plmap demo   [folder]               make a synthetic map + config to try the pipeline

Re-running is cheap: averaged spectra and per-file plots are cached and only
redone when the raw files change. Each combination of peak shapes is written
to its own folder, so Lorentzian and Gaussian/Voigt passes can be compared.
"""
import argparse
import os
import shutil
import sys
import time

import numpy as np

from . import background, config, io, plots, spectra

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "config_template.yaml")
SCAN_TEMPLATE = os.path.join(HERE, "scan_template.txt")


class Log:
    def __init__(self, path=None):
        self.fh = None
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.fh = open(path, "a", encoding="utf-8")
            self.fh.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')}  {' '.join(sys.argv)}\n")

    def __call__(self, msg=""):
        print(msg, flush=True)
        if self.fh:
            self.fh.write(msg + "\n")
            self.fh.flush()


def _steps_1_2(cfg, log):
    grid = config.grid(cfg)
    log(grid.describe())
    wl = io.load_wavelength(cfg["data"]["wavelength_file"])
    log(f"wavelength axis: {len(wl)} values, {wl.min():.2f}-{wl.max():.2f} nm")
    sp = spectra.build(cfg, grid, wl, log)
    out = cfg["data"]["output"]
    ks = config.selection(cfg["spectra"]["plots"], sorted(k for k in sp["files"] if k <= grid.npts))
    plots.spectrum_plots(os.path.join(out, "spectra_plots"), wl, sp["spectra"], ks, grid, sp["prefix"],
                         sp["files"], cfg["spectra"]["plot_dpi"], log)
    return grid, wl, sp


def cmd_spectra(cfg, log):
    grid, wl, sp = _steps_1_2(cfg, log)
    fm = np.ones(len(wl), bool) if not cfg["fit"]["range_nm"] else \
        (wl >= min(cfg["fit"]["range_nm"])) & (wl <= max(cfg["fit"]["range_nm"]))
    spc, _ = background.subtract(cfg, wl, sp["spectra"], sp["present"], fm, log)
    path = os.path.join(cfg["data"]["output"], "overview.png")
    plots.overview(path, wl, spc, grid, cfg["scan"]["flip_x"],
                   f"{os.path.basename(cfg['data']['folder'])}: {grid.nx}×{grid.ny} points")
    log(f"overview map: {path}")
    return grid, wl, sp


def cmd_run(cfg, log):
    from . import fitting, report, results
    grid, wl, sp = _steps_1_2(cfg, log)
    rng = cfg["fit"]["range_nm"] or [wl.min(), wl.max()]
    fm = (wl >= min(rng)) & (wl <= max(rng))
    spc, bg = background.subtract(cfg, wl, sp["spectra"], sp["present"], fm, log)
    out = cfg["data"]["output"]
    plots.overview(os.path.join(out, "overview.png"), wl, spc, grid, cfg["scan"]["flip_x"],
                   f"{os.path.basename(cfg['data']['folder'])}: {grid.nx}×{grid.ny} points")
    fit = fitting.run(cfg, wl, spc, sp["present"], log)
    fdir = os.path.join(out, "fit_" + results.tag(fit["peaks"]))
    results.write(fdir, cfg, grid, wl, sp["present"], fit, bg, sp["prefix"], log)
    title = (cfg["report"]["title"] or os.path.basename(os.path.normpath(cfg["data"]["folder"]))) + \
        "  —  " + ", ".join(f"{p['name']} {p['shape']}" for p in fit["peaks"])
    plots.summary(os.path.join(fdir, "summary"), wl, grid, fit, cfg["scan"]["flip_x"], title)
    log(f"summary figure: {os.path.join(fdir, 'summary.png')}")
    ks = config.selection(cfg["fit"]["plots"], [k for k in range(1, grid.npts + 1) if sp["present"][k - 1]])
    plots.fit_plots(os.path.join(fdir, "fit_plots"), wl, spc, fit, ks, grid, sp["prefix"],
                    cfg["spectra"]["plot_dpi"], log)
    report.write(os.path.join(fdir, "report.html"), cfg, grid, wl, sp["spectra"], sp["present"], bg, fit,
                 sp["prefix"], log)
    log(f"\ndone. Outputs in {fdir}")
    log("To refit some peaks as Gaussian or Voigt: set `shape:` for those peaks under fit.peaks "
        "in the config and run `plmap run` again (a new fit_* folder is written).")


def cmd_init(path):
    if os.path.exists(path):
        sys.exit(f"{path} already exists")
    shutil.copy(TEMPLATE, path)
    print(f"wrote {path} -- edit data.folder and data.wavelength_file")
    scan = os.path.join(os.path.dirname(os.path.abspath(path)), "scan.txt")
    if not os.path.exists(scan):
        shutil.copy(SCAN_TEMPLATE, scan)
        print(f"wrote {scan} -- copy the scan parameters into it and move it into the map folder "
              "(or point scan.file in the config to it)")


def cmd_rc(cfg, log):
    from . import rc
    rc.run(cfg, log)


def cmd_demo(folder):
    from .synthetic import make_demo
    cfg_path = make_demo(folder)
    print(f"synthetic map written to {folder}\nnow run:  plmap run {cfg_path}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="plmap", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init", help="write a config template")
    p.add_argument("config", nargs="?", default="config.yaml")
    for name, h in (("spectra", "PL steps 1-2 (needs only numpy + matplotlib)"), ("run", "PL steps 1-4"),
                    ("rc", "reflection-contrast map")):
        p = sub.add_parser(name, help=h)
        p.add_argument("config")
    p = sub.add_parser("demo", help="make a synthetic dataset to try the pipeline")
    p.add_argument("folder", nargs="?", default="plmap_demo")
    a = ap.parse_args(argv)
    if a.cmd == "init":
        return cmd_init(a.config)
    if a.cmd == "demo":
        return cmd_demo(a.folder)
    cfg = config.load(a.config)
    log = Log(os.path.join(cfg["data"]["output"], "plmap_log.txt"))
    t0 = time.time()
    {"spectra": cmd_spectra, "run": cmd_run, "rc": cmd_rc}[a.cmd](cfg, log)
    log(f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
