# PL mapping

Python tools for processing hyperspectral photoluminescence (PL) maps taken with
the Andor spectrometer: per-point spectra → spatial map → background
subtraction and peak fitting → interactive HTML report.

> Status: **planning**. The plan below is a draft for review; no processing code yet.
> `reference/` holds the earlier quick-look scripts (`plot_pl_map.py`,
> `plot_rc_map.py`) and an example output figure that this package builds on.

## Planned workflow

| Step | What it does | Output |
|---|---|---|
| 1. Spectra | Read each `<run>_loc<k>.txt` (2048 spectral lines × N CCD rows), average each line over CCD rows (with cosmic-ray rejection), attach wavelengths from the user's wavelength file | one spectrum per file + line plot per file |
| 2. Map | Convert loc index `k` to a stage coordinate (x fast, y slow) from the user's start/end/steps | spectral cube `[ny, nx, λ]` with real x/y axes |
| 3. Background + fits | Subtract CCD offset and substrate background; fit all peaks as Lorentzians; user switches chosen peaks to Gaussian/Voigt and re-fits | per-pixel peak wavelength, intensity, FWHM (+ errors) as CSV/NPZ and maps |
| 4. Report | Self-contained interactive HTML: maps of any fitted quantity, click a pixel to see its spectrum and fit | `report.html` (+ static PNG/PDF summary) |

See the plan in the session for details and open questions.
