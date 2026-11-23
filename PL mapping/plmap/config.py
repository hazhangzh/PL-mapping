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
        "file": None,               # scan-parameter file (see scan_template.txt); default <folder>/scan.txt
        "x_start": None, "x_end": None, "x_steps": None,   # fast axis (or x_points = x_steps + 1)
        "y_start": None, "y_end": None, "y_steps": None,   # slow axis (or y_points)
        "x_points": None, "y_points": None,
        "exposure_s": None,         # exposure per point (s), from the scan file; used by RC
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
        "offset_region_nm": None,   # signal-free band for the per-point offset; null = auto (edge with less PL)
        "substrate": "auto",        # auto (dimmest pixels) | none | path to a spectrum file
        "substrate_fraction": 0.3,  # fraction of dimmest pixels averaged as substrate background
    },
    "fit": {
        "range_nm": None,           # [min, max] fixed wavelength range to fit; overrides range_meV
        "range_meV": None,          # [lo, hi] fit range relative to the anchor peak, e.g. [-62, 15];
                                    #   moves with each point's shift. null + range_nm null = around the peaks
        "baseline": "linear",       # none | constant | linear | quadratic (fitted per pixel)
        "peaks": [],                # list of {name, center_nm | offset_meV, shape, window_nm | window_meV,
                                    #   fwhm_meV, min/max_fwhm_meV, tail_meV, max_tail_meV}
        "combined": [],             # extra maps from the fitted peaks, e.g.
                                    #   {name: XT_X0_ratio, ratio: [[XT1, XT2], X0], quantity: height}
                                    #   {name: dE_XT1_X0, difference: [XT1, X0]}   (peak energies, meV)
        "anchor": None,             # peak the offsets refer to (default: the one with offset_meV 0)
        "anchor_search_nm": None,   # [min, max] where the anchor may be; null = anywhere in the data
        "track_shift": True,        # per point: find the rigid shift of the whole peak pattern first
        "max_shift_meV": 10.0,      # largest point-to-point shift searched (from the bright-point mean)
        "skip_no_signal": True,     # do not fit points where the peak pattern is not found (off-flake)
        "auto_detect": "main",      # if `peaks` is empty: main = the main (highest-energy) exciton only,
                                    # all = up to max_peaks peaks, false = off
        "max_peaks": 4,
        "min_snr": 4.0,             # peak height / residual noise needed for a "good" fit
        "workers": "auto",          # parallel processes for per-pixel fits (auto = number of CPUs, max 8)
        "plots": "none",            # per-file fit plots: all | none | list of loc indices
    },
    "rc": {                         # reflection contrast (plmap rc): RC = (R - R0) / R0
        "dark": "auto",             # CCD offset: auto (red edge, where the lamp is ~0) | number | file
        "r0": "auto",               # auto (featureless points of the map) | [loc, loc, ...] | path to a spectrum file
        "r0_exposure_s": None,      # exposure of an R0 file (s); default = map exposure
        "r0_rows_summed": 1,        # an R0 file saved as a sum over n CCD rows is divided by n
        "r0_scale": "auto",         # R0 file only: auto = match featureless map points | exposure | a number
        "window_nm": None,          # [min, max] around the resonance; null = auto (where RC varies most)
        "fit": True,                # fit the resonance at every point (complex Lorentzian + linear baseline)
        "min_fwhm_meV": 1.0, "max_fwhm_meV": 40.0,
        "min_snr": 5.0,             # resonance amplitude / residual noise needed for a "good" fit
        "plots": "none",            # per-file RC plots: all | none | list of loc indices
    },
    "report": {
        "title": None,
        "spectral_binning": 1,      # bin spectra in the HTML to shrink the file
    },
}

PEAK_DEFAULTS = {"shape": "lorentzian", "window_nm": 3.0, "window_meV": None, "fwhm_meV": 10.0,
                 "min_fwhm_meV": 0.5, "max_fwhm_meV": 80.0, "center_nm": None, "offset_meV": None,
                 "tail_meV": 1.0, "min_tail_meV": 0.05, "max_tail_meV": 8.0}


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
        if p is None:
            return None
        p = os.path.expanduser(str(p))  # YAML reads date-named folders (11042026) as int
        return p if os.path.isabs(p) else os.path.normpath(os.path.join(base, p))

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
    _scan_values(cfg["scan"], base, d["folder"], resolve)
    if cfg["rc"]["dark"] not in ("auto", None) and not _is_number(cfg["rc"]["dark"]):
        cfg["rc"]["dark"] = resolve(cfg["rc"]["dark"])
    if isinstance(cfg["rc"]["r0"], str) and cfg["rc"]["r0"] != "auto":
        cfg["rc"]["r0"] = resolve(cfg["rc"]["r0"])
    cfg["fit"]["peaks"] = [normalise_peak(p, i) for i, p in enumerate(cfg["fit"]["peaks"] or [])]
    return cfg


SCAN_KEYS = {  # scan-file name -> config key (names follow the scan program: axis 1 = x, axis 2 = y)
    "start_1_mm": "x_start", "end_1_mm": "x_end", "points_1": "x_points", "steps_1": "x_points",
    "start_2_mm": "y_start", "end_2_mm": "y_end", "points_2": "y_points", "steps_2": "y_points",
    "x_start": "x_start", "x_end": "x_end", "x_points": "x_points", "x_steps": "x_steps",
    "y_start": "y_start", "y_end": "y_end", "y_points": "y_points", "y_steps": "y_steps",
    "exposure_s": "exposure_s", "units": "units",
}
SCAN_INFO = ("off_sample_1_mm", "off_sample_2_mm", "averages", "date", "sample", "notes")


def _is_number(v):
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def read_scan_file(path):
    """Two-column scan-parameter file: `name  value` per line ('#' starts a comment).
    Returns {config key: value}. Unknown names are an error, so typos are caught."""
    out = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for n, line in enumerate(fh, 1):
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.replace(",", " ").replace("=", " ").split()
            if len(parts) < 2:
                sys.exit(f"{path}, line {n}: expected `name value`, got {line!r}")
            name, val = parts[0].lower(), parts[1]
            if name in SCAN_INFO:
                continue
            if name not in SCAN_KEYS:
                sys.exit(f"{path}, line {n}: unknown name {parts[0]!r}; expected start_1_mm, end_1_mm, points_1, "
                         f"start_2_mm, end_2_mm, points_2, exposure_s (see `plmap init`)")
            out[SCAN_KEYS[name]] = val if name == "units" else float(val)
    return out


def _scan_values(s, base, folder, resolve):
    """Fill the scan section from the scan file (if any); points -> steps; check completeness."""
    path = resolve(s["file"]) if s["file"] else os.path.join(folder, "scan.txt")
    explicit = [k for k in ("x_start", "x_end", "x_steps", "y_start", "y_end", "y_steps", "x_points", "y_points")
                if s[k] is not None]
    if os.path.exists(path) and (s["file"] or not explicit):
        if explicit:
            sys.exit(f"scan values are given both in {path} and in the config ({', '.join(explicit)}); "
                     "keep them in one place")
        s.update(read_scan_file(path))
        s["file"] = path
    elif s["file"]:
        sys.exit(f"scan file {path} not found")
    for ax in ("x", "y"):
        if s[f"{ax}_steps"] is None and s[f"{ax}_points"] is not None:
            s[f"{ax}_steps"] = int(round(float(s[f"{ax}_points"]))) - 1
    missing = [k for k in ("x_start", "x_end", "x_steps", "y_start", "y_end", "y_steps") if s[k] is None]
    if missing:
        sys.exit(f"scan.{', scan.'.join(missing)} must be set (in the config or in a scan file: "
                 f"`plmap init` writes a scan.txt template)")


def normalise_peak(p, i=0):
    p = {**PEAK_DEFAULTS, **p}
    p.setdefault("name", f"P{i + 1}")
    if (p["center_nm"] is None) == (p["offset_meV"] is None):
        sys.exit(f"peak {p['name']}: give exactly one of center_nm (absolute) or offset_meV (relative to the anchor)")
    for k in ("center_nm", "offset_meV", "window_meV", "fwhm_meV", "min_fwhm_meV", "max_fwhm_meV",
              "tail_meV", "min_tail_meV", "max_tail_meV", "window_nm"):
        if p[k] is not None:
            p[k] = float(p[k])
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
