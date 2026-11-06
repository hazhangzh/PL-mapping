# Development notes / handoff

Status as of 2026-11-06 (v0.1.0). Read with `README.md`.

## Where things stand

- Steps 1–4 are implemented: `plmap spectra` and `plmap run`, plus `plmap init` and `plmap demo`.
- **Tested on real data:** only one off-flake file (`1_loc1.txt`, 2048 × 25, tab-separated,
  CRLF) and the 2048-line `wavelength.txt`. Loading and row-averaging are verified on them
  (identical to a plain row mean). The file had no PL, so the peak fitting has **not** been
  checked on real spectra yet.
- **Tested on synthetic data:** `plmap/synthetic.py` writes maps in the same format and
  noise level (offset ~828 counts sloping across the chip, 4.4-count read noise per pixel),
  with three WSe2-like peaks:
  - X0, Lorentzian, 1.745 eV
  - XT, Voigt, 1.715 eV
  - LE, Gaussian, 1.680 eV

  The X0 peak energy is recovered within 0.03 meV, and the lineshape check picks the right
  shapes. The full 41×41 map runs in about 150 s on 4 cores; about 85 s of that is writing
  the PNGs, on the first run only.
- `pytest` (10 tests) passes.

## Decisions made with the user

- "N steps" means N + 1 points. The raster is always one-directional, row-major, x fast.
- Spectrum = mean over **all** CCD rows of each line.
- Per-file spectrum plots are made for **all** files by default.
- scipy and plotly are fine on the analysis desktop. The lab PC may lack them, so steps 1–2
  only need numpy, matplotlib and pyyaml, and scipy/plotly are imported only when needed.
- Reflection contrast (RC) comes later and should reuse `io`, `grid`, `spectra`,
  `plots` and `report`.

## Implementation choices worth knowing

- **Cosmic rays** are detected per CCD row along wavelength (a 7-line running median), not
  across rows. The reference scripts compared across rows, which would remove real PL when
  the spot covers only a few rows. The noise level is a clipped std rather than the MAD,
  because the MAD of integer counts is quantised (it read 3.0 when the true noise was 4.3)
  and gave false hits.
- **Fits** are done in energy. Peaks are area-normalised.
  - Voigt uses Weideman's Faddeeva approximation with N = 16 in Python (5× faster than
    scipy, error ~1e-7) and N = 32 in the HTML's JavaScript.
  - Each pixel is seeded by the fit of the mean bright spectrum; a linear solve for areas
    and baseline gives the starting values.
- **Lineshape check** runs on the 5 brightest individual points, not on the mean
  spectrum. Averaging points with different peak energies makes every peak look Voigt-like.
  Voigt is recommended only if it beats both Lorentzian and Gaussian by more than 2 %.
- **Speed:** BLAS is forced to one thread (`plmap/__init__.py`). Without that, the parallel
  workers were 15× slower. The `__main__` guard is there for Windows (spawn), which was
  tested here by simulation only.
- The **HTML** embeds plotly.js and float32 spectra as base64, and recomputes fit curves in
  JavaScript from the parameters. A 41×41 map gives about 24 MB; `report.spectral_binning`
  makes it smaller.
- **Maps** put y = 0 at the top (file order), as the reference scripts do. `scan.flip_x`
  mirrors left-right.

## Next steps (need real on-flake maps, run locally)

1. Run `plmap spectra` and then `plmap run` with `peaks: []` on a real map. Check the
   auto-detected peaks, then fix the names, windows and default FWHM in the config for WSe2
   (X0, trion, low-energy band).
2. Check the background on real data:
   - whether `substrate: auto` (dimmest 30 %) is right, or whether a measured substrate or
     dark spectrum works better
   - whether the default offset band (red-most 8 nm) is free of signal
3. Check the cosmic-ray rejection count and the `min_snr` threshold on real spectra.
4. Check that the map orientation and `flip_x` match real space, for example against a
   microscope image.
5. Run on Windows for real: parallel workers and paths with spaces.
6. Add the RC pipeline (see `reference/plot_rc_map.py`):
   - RC = (R − R0)/R0
   - R0 from a file or from substrate pixels, with exposure scaling
