"""Synthetic PL map in the Andor file format, for testing and trying the pipeline.

Mimics the real files: 2048 lines x 25 CCD rows, tab-separated integers with
CRLF line endings, ~828-count offset sloping across the chip, ~4.4 counts
read noise per pixel. A WSe2-like flake has a Lorentzian X0, a Voigt trion
and a Gaussian low-energy band whose energies and widths vary in space.
Includes cosmic rays, one dropped "--" pixel and missing files at the end.
"""
import os

import numpy as np

from . import HC
from . import lineshapes as ls

TRUTH_PEAKS = [  # name, shape, E0 (eV), fwhm(s) (eV), peak height (counts/row at brightest)
    ("X0", "lorentzian", 1.7450, (0.0060,), 90.0),
    ("XT", "voigt", 1.7150, (0.0060, 0.0050), 45.0),
    ("LE", "gaussian", 1.6800, (0.0300,), 15.0),
]


def wavelength_axis(n=2048):
    i = np.arange(n)
    return 691.585397 + 3.77074459e-02 * i - 1.90668962e-07 * i ** 2


def make_map(folder, x_steps=20, y_steps=16, nrow=25, n_missing=3, seed=1):
    """Write <folder>/1_loc*.txt + wavelength.txt. Returns (wl, truth dict)."""
    rng = np.random.default_rng(seed)
    os.makedirs(folder, exist_ok=True)
    wl = wavelength_axis()
    E = HC / wl
    nx, ny = x_steps + 1, y_steps + 1
    with open(os.path.join(folder, "wavelength.txt"), "w", newline="") as fh:
        fh.write("".join(f"{v:.6g}\r\n" for v in wl))
    yy, xx = np.mgrid[0:ny, 0:nx]
    # flake: tilted band with soft edges; brightness and strain gradient inside it
    u = (xx - 0.45 * nx) * np.cos(0.5) + (yy - 0.5 * ny) * np.sin(0.5)
    v = -(xx - 0.45 * nx) * np.sin(0.5) + (yy - 0.5 * ny) * np.cos(0.5)
    flake = 1 / (1 + np.exp((np.abs(u) - 0.3 * nx) / 0.6)) / (1 + np.exp((np.abs(v) - 0.18 * ny) / 0.6))
    shift = 0.004 * (u / (0.3 * nx))            # +-4 meV strain gradient along the flake
    broad = 1 + 0.4 * (v / (0.18 * ny)) ** 2    # broader towards the flake edges
    rowprof = np.exp(-0.5 * ((np.arange(nrow) - 12) / 5.0) ** 2)
    rowprof /= rowprof.mean()
    dark = 891.42 - 0.0866 * wl
    truth = {"nx": nx, "ny": ny, "flake": flake.ravel(), "peaks": {}}
    for name, shape, e0, w, h in TRUTH_PEAKS:
        truth["peaks"][name] = {"shape": shape, "E0": (e0 + shift * (1 if name != "LE" else 0.5)).ravel(),
                                "widths": [np.asarray(wi * broad).ravel() for wi in w], "h": (h * flake).ravel()}
    npts = nx * ny
    for i in range(npts - n_missing):
        sig = np.zeros_like(E)
        for name, shape, *_ in TRUTH_PEAKS:
            t = truth["peaks"][name]
            prof = ls.evaluate(shape, E, [t["E0"][i], 1.0] + [w[i] for w in t["widths"]])
            sig += t["h"][i] * prof / ls.evaluate(shape, np.array([t["E0"][i]]),
                                                   [t["E0"][i], 1.0] + [w[i] for w in t["widths"]])[0]
        sig += 4.0 * (1 - flake.ravel()[i]) * np.exp(-0.5 * ((wl - 730) / 25) ** 2)   # weak substrate band
        frame = dark[:, None] + sig[:, None] * rowprof[None, :] + rng.normal(0, 4.4, (len(wl), nrow))
        frame += rng.integers(-2, 3, nrow)[None, :]
        for _ in range(rng.poisson(1.5)):                 # cosmic rays: 1-3 lines, 1-2 rows
            j, r = rng.integers(0, len(wl) - 3), rng.integers(0, nrow - 2)
            frame[j:j + rng.integers(1, 4), r:r + rng.integers(1, 3)] += rng.uniform(200, 3000)
        txt = np.round(frame).astype(int).astype(str)
        if i == 7:
            txt[1000, 3] = "--"
        with open(os.path.join(folder, f"1_loc{i + 1}.txt"), "w", newline="") as fh:
            fh.write("".join("\t".join(row) + "\r\n" for row in txt))
    truth["present"] = np.arange(npts) < npts - n_missing
    return wl, truth


DEMO_CONFIG = """\
# synthetic demo map -- see plmap/config_template.yaml for every option
data:
  folder: data
  wavelength_file: data/wavelength.txt
  output: output
scan:
  x_start: 0.2600
  x_end: 0.2700
  x_steps: {xs}
  y_start: 0.2500
  y_end: 0.2580
  y_steps: {ys}
  units: mm
spectra:
  plots: all
fit:
  range_nm: [700, 760]
  baseline: linear
  peaks: []        # empty -> auto-detect; then copy the printed list here and edit shapes
"""


def make_demo(folder, x_steps=20, y_steps=16):
    make_map(os.path.join(folder, "data"), x_steps, y_steps)
    path = os.path.join(folder, "config.yaml")
    with open(path, "w") as fh:
        fh.write(DEMO_CONFIG.format(xs=x_steps, ys=y_steps))
    return path
