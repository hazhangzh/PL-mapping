"""plmap -- processing of hyperspectral PL maps from the Andor spectrometer.

Pipeline: raw loc files -> averaged spectra -> spatial map -> background
subtraction -> per-pixel peak fits -> CSV/NPZ results + interactive HTML.
"""
import os

# The fits use many small matrices: multithreaded BLAS only adds overhead and,
# with several worker processes, makes them fight for the CPU (15x slower).
# Must be set before numpy is imported.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

__version__ = "0.1.0"

HC = 1239.84193  # eV nm
