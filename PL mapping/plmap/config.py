"""Config file handling. All user inputs live in one YAML file (see config_example.yaml)."""
import copy
import os
import sys

import yaml

from .grid import ScanGrid
from .lineshapes import SHAPES

DEFAULTS = {
    "data": {
        "folder": None,             # folder with <run>_loc<k>.txt
        "wavelength_file": None,    # one wavelength (nm) per line
        "prefix": None,             # run prefix (the "1" in 1_loc5.txt); auto if only one run
        "output": None,             # default: <folder>/plmap_output
    },
    "scan": {
        "x_start": None, "x_end": None, "x_steps": None,   # fast axis
        "y_start": None, "y_end": None, "y_steps": None,   # slow axis
        "units": "mm",
        "flip_x": False,            # display mirrored left-right (real space)
    },
    "spectra": {
        "cosmic_ray_sigma": 6.0,    # reject outliers across CCD rows before averaging; null = off
        "despike_sigma": 8.0,       # remove single-pixel spikes along the spectrum; null = off
        "plots": "all",             # all | none | list of loc indices
        "plot_dpi": 100,
    },
    "background": {
        "dark_file": None,          # optional dark spectrum (same format as a loc file, or 1 column)
        "offset_region_nm": None,   # signal-free band for the per-point offset; null = red-most 8 nm
        "substrate": "auto",        # auto (dimmest pixels) | none | path to a spectrum file
        "substrate_fraction": 0.3,  # fraction of dimmest pixels averaged as substrate background
    },
    "fit": {
        "range_nm": None,           # [min, max] wavelength range to fit; null = whole spectrum
        "baseline": "linear",       # none | constant | linear (fitted per pixel, on top of the above)
        "peaks": [],                # list of {name, center_nm, shape, window_nm, fwhm_meV, max_fwhm_meV}
        "auto_detect": True,        # detect peaks on the mean spectrum if `peaks` is empty
        "max_peaks": 4,
        "min_snr": 5.0,             # peak height / residual noise needed for a "good" fit
        "workers": "auto",          # parallel processes for per-pixel fits (auto = number of CPUs, max 8)
        "plots": "none",            # per-file fit plots: all | none | list of loc indices
    },
    "report": {
        "title": None,
        "spectral_binning": 1,      # bin spectra in the HTML to shrink the file
    },
}

PEAK_DEFAULTS = {"shape": "lorentzian", "window_nm": 3.0, "fwhm_meV": 10.0,
                 "min_fwhm_meV": 0.5, "max_fwhm_meV": 80.0}


def _merge(base, new):
    out = copy.deepcopy(base)
    for k, v in (new or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load(path):
    with open(path) as fh:
        user = yaml.safe_load(fh) or {}
    unknown = set(user) - set(DEFAULTS)
    if unknown:
        sys.exit(f"Unknown config section(s): {sorted(unknown)}")
    cfg = _merge(DEFAULTS, user)
    base = os.path.dirname(os.path.abspath(path))

    def resolve(p):
        return p if p is None or os.path.isabs(p) else os.path.normpath(os.path.join(base, p))

    d = cfg["data"]
    for key in ("folder", "wavelength_file", "output"):
        d[key] = resolve(d[key])
    for key in ("dark_file",):
        cfg["background"][key] = resolve(cfg["background"][key])
    if cfg["background"]["substrate"] not in ("auto", "none", None):
        cfg["background"]["substrate"] = resolve(cfg["background"]["substrate"])
    if not d["folder"] or not d["wavelength_file"]:
        sys.exit("data.folder and data.wavelength_file are required")
    if d["output"] is None:
        d["output"] = os.path.join(d["folder"], "plmap_output")
    missing = [k for k in ("x_start", "x_end", "x_steps", "y_start", "y_end", "y_steps")
               if cfg["scan"][k] is None]
    if missing:
        sys.exit(f"scan.{', scan.'.join(missing)} must be set")
    cfg["fit"]["peaks"] = [normalise_peak(p, i) for i, p in enumerate(cfg["fit"]["peaks"] or [])]
    return cfg


def normalise_peak(p, i=0):
    p = {**PEAK_DEFAULTS, **p}
    p.setdefault("name", f"P{i + 1}")
    if p.get("center_nm") is None:
        sys.exit(f"peak {p['name']}: center_nm is required")
    p["shape"] = str(p["shape"]).lower()
    if p["shape"] not in SHAPES:
        sys.exit(f"peak {p['name']}: shape must be one of {SHAPES}, got {p['shape']!r}")
    return p


def grid(cfg):
    s = cfg["scan"]
    return ScanGrid(float(s["x_start"]), float(s["x_end"]), int(s["x_steps"]),
                    float(s["y_start"]), float(s["y_end"]), int(s["y_steps"]), s["units"])


def selection(spec, ks):
    """Turn a plots setting (all | none | list) into a list of loc indices."""
    if spec in (None, "none", False):
        return []
    if spec in ("all", True):
        return list(ks)
    want = {int(k) for k in (spec if isinstance(spec, (list, tuple)) else [spec])}
    return [k for k in ks if k in want]
