# PL mapping (`plmap`)

Processing of hyperspectral photoluminescence (PL) maps from the Andor spectrometer:

1. **Spectra**: each `<run>_loc<k>.txt` file (one line per wavelength, one column per CCD row)
   becomes one spectrum by averaging every CCD row of each line, after cosmic-ray rejection.
   A line plot is written for every file.
2. **Map**: loc index `k` becomes a stage coordinate, using the scan start/end and step counts
   (one-directional raster, x fast, y slow).
3. **Background and peak fits**: dark/offset/substrate background is subtracted, then all
   peaks are fitted at every point. Peaks start as Lorentzians; you then switch selected
   peaks to Gaussian or Voigt and re-run. The output is peak wavelength/energy, intensity and FWHM for each point.
4. **Interactive HTML report**: one self-contained file that works offline.

`reference/` holds the earlier quick-look scripts this package grew from.
Reflection-contrast (RC) maps are planned and will reuse steps 1, 2 and 4.

## Install

```bash
cd "PL mapping"
pip install -e ".[fit]"          # numpy, matplotlib, pyyaml + scipy, plotly
```

Steps 1-2 (`plmap spectra`) need only numpy, matplotlib and pyyaml, so they also run on the
lab computer. Fitting and the HTML report (`plmap run`) also need scipy and plotly.
If the package is not installed, use `python -m plmap ...` from inside `PL mapping/`.

## Quick start

```bash
plmap demo plmap_demo                 # synthetic map in the real file format, to try things out
plmap run plmap_demo/config.yaml

plmap init my_map.yaml                # commented config template for real data
plmap spectra my_map.yaml             # steps 1-2 only (quick look while the scan is running)
plmap run my_map.yaml                 # steps 1-4
```

## The config file

Everything you would type for a map goes in one YAML file (`plmap init` writes a commented
template, `plmap/config_template.yaml`). The parts you must set:

```yaml
data:
  folder: D:/data/2026-11-05/map3          # holds 1_loc1.txt ... 1_loc1681.txt
  wavelength_file: D:/data/wavelength.txt  # 2048 lines: wavelength (nm) of each spectral line
scan:                                      # x = fast axis, y = slow axis, as entered for the scan
  x_start: 0.2600
  x_end: 0.3000
  x_steps: 40                              # 40 steps = 41 points
  y_start: 0.2500
  y_end: 0.2900
  y_steps: 40
  units: mm
fit:
  range_nm: [700, 760]
  peaks:                                   # leave empty ([]) to auto-detect on the first run
    - {name: X0, center_nm: 710.5, shape: lorentzian, window_nm: 3, fwhm_meV: 6}
    - {name: XT, center_nm: 723.0, shape: lorentzian, window_nm: 3, fwhm_meV: 8}
```

`loc k` is at row `(k-1) // nx`, column `(k-1) % nx`, with `nx = x_steps + 1`. Files that are
missing (the scan is still running) show as blank pixels.

## Choosing peak shapes

1. Run with every peak set to `shape: lorentzian`, or with `peaks: []` to auto-detect them. The
   console output and the HTML report include a **lineshape check**: each peak is refitted as
   Lorentzian, Gaussian and Voigt on the 5 brightest points and their χ² values are compared. Voigt is only
   recommended if it beats both simpler shapes by more than 2 %, because its extra parameter
   always lets it fit at least as well.
2. Set `shape: gaussian` or `shape: voigt` for the peaks you choose, and run `plmap run` again.

Each shape combination is written to its own folder (`fit_X0-L_XT-V/`, ...), so passes can be
compared side by side.

## Outputs (in `data.output`, default `<folder>/plmap_output/`)

| File | Contents |
|---|---|
| `spectra_plots/<run>_loc<k>.png` | step 1: averaged spectrum of every file |
| `spectra.npz` | cached averaged spectra (re-read only when raw files change) |
| `overview.png` | step 2: integrated-PL map + mean bright/dim spectra |
| `fit_<shapes>/results.csv` | one row per point: file, loc, row, col, stage x/y, and for each peak its centre (nm, meV) ± error, FWHM (meV, nm) ± error, height, integrated intensity ± error, SNR, good-fit flag |
| `fit_<shapes>/results.npz` | the same as arrays, plus the raw fit parameters |
| `fit_<shapes>/fit_info.yaml` | settings used, mean-spectrum fit, lineshape check |
| `fit_<shapes>/summary.png/.pdf` | maps of intensity, peak energy and FWHM for every peak |
| `fit_<shapes>/report.html` | interactive report (below) |
| `fit_<shapes>/fit_plots/` | per-file fit plots if `fit.plots` is set |
| `plmap_log.txt` | everything printed to the console |

**HTML report:**
- Choose which map to show: integrated PL, any peak's intensity, height, wavelength, energy,
  FWHM or SNR, fit quality, or the integral over a wavelength band you type in.
- Set the colour range and colour scale, flip x, and hide rejected fits.
- Click a pixel or use the arrow keys to see its raw and background-subtracted spectrum,
  fitted components and residual, in wavelength or energy, with a table of fitted values.
- The page also shows the mean-spectrum fit and the lineshape-check table.

## Method notes

- **Cosmic rays** are found in each CCD row as spikes above a 7-line running median along
  wavelength, and excluded from the row average. Unlike a comparison across CCD rows, this
  does not remove real PL when the spot covers only a few rows. A second pass removes
  leftover single-line spikes from the averaged spectrum. A `--` in a file (dropped pixel)
  is interpolated.
- **Background**, applied in order and each step optional:
  - a dark spectrum (`dark_file`)
  - a per-point offset, the mean of a signal-free band (default: the red-most 8 nm)
  - a substrate spectrum: the mean of the 30 % dimmest points, or a file you give
  - a linear baseline fitted together with the peaks
- **Fits** use the energy axis, where Lorentzian, Gaussian and Voigt widths have physical
  meaning. All peaks plus the baseline are fitted together at each point, starting from the
  fit of the mean bright spectrum. The Voigt profile uses Weideman's Faddeeva approximation:
  it agrees with scipy to 1e-7 and is 5× faster.
- **Reported values**:
  - intensity = fitted peak area in counts, summed over spectral lines
  - height = value of the fitted peak at its centre (counts)
  - FWHM: for a Voigt peak, from the Olivero–Longbothum formula
  - errors come from the fit covariance
- **Quality flag**: a peak counts as good if SNR (height / residual noise) is at least `min_snr`, its
  centre and widths are not stuck at their limits, and its area is above 0. Rejected points are grey in
  the maps; their values remain in the CSV.
- **Speed**: points are fitted in parallel (`fit.workers: auto`). A 41×41 map takes a few
  minutes, and plotting about 1,700 per-file PNGs adds roughly 1-2 minutes on the first run only.

## Tests

```bash
pytest          # unit tests + an end-to-end run on a synthetic map with known peaks
```
