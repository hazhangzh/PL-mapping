"""Reading raw Andor files: per-point CCD matrices and the wavelength axis."""
import glob
import os
import re
import sys

import numpy as np

LOC_RE = re.compile(r"^(?P<prefix>.*?)_loc(?P<k>\d+)\.txt$", re.IGNORECASE)


def load_wavelength(path):
    """One wavelength (nm) per line; blank lines ignored. Line i <-> spectral line i."""
    wl = np.loadtxt(path, ndmin=1)
    if wl.ndim != 1:  # tolerate a 2-column (index, wavelength) file
        wl = wl[:, -1]
    return wl.astype(np.float64)


def load_matrix(path):
    """Load one raw file: rows = spectral lines, columns = CCD rows.

    Tabs/spaces and CRLF are handled. A "--" (dropped pixel) is replaced by
    linear interpolation along the spectral direction for that CCD row.
    """
    with open(path, "r", errors="replace") as fh:
        txt = fh.read()
    if "--" not in txt:
        return np.array([ln.split() for ln in txt.strip().splitlines()], dtype=np.float64)
    a = np.genfromtxt(path, missing_values="--", filling_values=np.nan)
    a = np.atleast_2d(a)
    for j in range(a.shape[1]):
        col = a[:, j]
        bad = np.isnan(col)
        if bad.any() and (~bad).any():
            col[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), col[~bad])
    return a


def find_loc_files(folder, prefix=None):
    """Return {k: path} for files named <prefix>_loc<k>.txt.

    If several prefixes (runs) are present, `prefix` must say which one.
    """
    found = {}
    for path in glob.glob(os.path.join(folder, "*_loc*.txt")):
        m = LOC_RE.match(os.path.basename(path))
        if m:
            found.setdefault(m["prefix"], {})[int(m["k"])] = path
    if not found:
        sys.exit(f"No <run>_loc<k>.txt files found in {folder}")
    if prefix is None:
        if len(found) > 1:
            sys.exit(f"Several runs found in {folder}: {sorted(found)}. Set data.prefix in the config.")
        prefix = next(iter(found))
    if str(prefix) not in found:
        sys.exit(f"No files with prefix '{prefix}' in {folder}; found {sorted(found)}")
    return str(prefix), found[str(prefix)]
