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

5. **Reflection contrast (RC)**: `plmap rc` turns a reflection map into RC = (R − R0)/R0 and fits the
   exciton resonance at every point (see [Reflection contrast](#reflection-contrast-plmap-rc)).

`reference/` holds the earlier quick-look scripts this package grew from.

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

plmap init my_map.yaml                # commented config template + scan.txt template
plmap spectra my_map.yaml             # steps 1-2 only (quick look while the scan is running)
plmap run my_map.yaml                 # steps 1-4
plmap rc my_rc_map.yaml               # reflection-contrast map
```

## Scan parameters: `scan.txt`

For every map, copy the values from the scan program into a two-column `scan.txt` in the map
folder (`plmap init` writes a template). plmap reads it automatically:

```
start_1_mm   3.108        # axis 1 = x = fast axis
end_1_mm     3.128
points_1     41           # the scan program's "Steps" box = number of points (41 points = 40 steps)
start_2_mm   -7.9785      # axis 2 = y = slow axis
end_2_mm     -7.9585
points_2     41
exposure_s   5            # per point; used to put R and R0 on the same scale
```

The step is (end − start)/(points − 1), here 0.5 µm. Maps are drawn in µm with the scan start at
(0, 0): x = stage x − start, y = stage y − start (a pure translation). The CSV keeps the stage
coordinates as well. You can also give `x_start`, `x_end`, `x_points` (or `x_steps`), … in the
config's `scan:` section instead, but not in both places.

## The config file

Everything else for a map goes in one YAML file (`plmap init` writes a commented
template, `plmap/config_template.yaml`). The parts you must set:

```yaml
data:
  folder: D:/data/2026-11-04/11042026      # holds 1_loc1.txt ... 1_loc1681.txt and scan.txt
  wavelength_file: D:/data/wavelength.txt  # 2048 lines: wavelength (nm) of each spectral line
scan:
  flip_x: true                             # draw mirrored left-right (real-space view)
fit:
  range_nm: [700, 721]
  peaks:                                   # leave empty ([]) to auto-detect the main exciton
    - {name: X0, center_nm: 713.6, shape: lorentzian, window_nm: 4.5, fwhm_meV: 6, min_fwhm_meV: 2}
```

`loc k` is at row `(k-1) // nx`, column `(k-1) % nx`, with `nx` = points along x. Files that are
missing (the scan is still running) show as blank pixels.

**Which peaks are fitted.** By default only the main exciton X0 (the highest-energy strong peak)
is fitted. The low-energy band of WSe2 changes too much from point to point to fit with fixed
peaks; map it with the report's band integral instead. `fit.auto_detect: all` detects up to
`max_peaks` peaks if you want them.

## Multi-peak fits that follow the peaks (`offset_meV`, `range_meV`)

Peak energies move between samples (strain, dielectric environment, temperature) and from point
to point on one flake (11/16 WSe2: X0 1719-1737 meV). The energy *differences* between excitonic
species (trion and dark-exciton binding energies) change much less. So instead of fixed wavelengths,
give each peak an `offset_meV` from an anchor peak (X0, offset 0) and the fit range as `range_meV`
relative to it (with `range_nm: null`). Then:

1. **Anchor.** The whole pattern (all peaks at their offsets, non-negative amplitudes, free baseline)
   is slid along each of the 10 brightest spectra; the median best match is X0's energy. This works
   even when X0 is weak. Individual spectra are used because their mean is smeared by the shifts.
2. **Template.** The brightest points are each aligned to the pattern, averaged and fitted (twice, the
   second time with the fitted template). This fit sets the start values for every point.
3. **Per point.** The template is slid again (±`max_shift_meV`) to find the point's rigid shift. The
   start values, the ±`window_meV` centre windows and the fit range all move with it, then the full fit
   runs. Points where the pattern explains < 500 noise units of χ² (off-flake) keep shift 0 and, with
   `skip_no_signal: true`, are not fitted (17 s instead of 5 min for a 41×41 map).

Outputs: `shift_meV` (map of the rigid shift, a strain/environment map), `shift_signif`, and the fit
range of every point (`range_lo_nm`, `range_hi_nm`); the HTML shows each point's own range.
`track_shift: false` turns tracking off, and `range_nm` still gives a fixed range.

**Lineshapes (literature, low-T hBN-encapsulated WSe2):**
- **X0:** `voigt`. The Lorentzian part is homogeneous (radiative decay and dephasing); the Gaussian
  part is inhomogeneous (disorder). Near the radiative limit it is close to Lorentzian.
- **Trions XT1 (intervalley), XT2 (intravalley):** `trion` = Gaussian ⊗ one-sided exponential on the
  low-energy side. Electron recoil: E = E_T(0) − (M_X/m_e)·E_kin, with a thermal tail
  τ = (M_X/m_e)·k_B·T ≈ 2 k_B T. Its centre is the zero-momentum trion energy (the high-energy edge),
  not the maximum. At ~4 K, τ (< 1 meV) is below the disorder width, and on the 11/16 map the tail went
  to zero, so `gaussian` is used there; use `trion` for warm or high-power data.
- **Dark exciton XD:** `gaussian`. Its radiative width is µeV-scale, so the observed width is
  inhomogeneous. Keep `max_fwhm_meV` small so it cannot absorb the neighbouring trion tail.
- A strong neighbour inside the range (the 1.674 eV line) must be fitted too, or its tail biases XD.

### Combined maps (`fit.combined`)

Maps computed from the fitted peaks at every point, e.g. the trion/X0 ratio and the trion binding energies:

```yaml
fit:
  combined:
    - {name: XT_X0_height_ratio, ratio: [[XT1, XT2], X0], quantity: height}  # mean(XT1, XT2) / X0
    - {name: dE_XT1_X0, difference: [XT1, X0]}     # E(XT1) - E(X0), meV
    - {name: dE_XT2_X0, difference: [XT2, X0]}
```

- `ratio: [A, B]` = mean of `quantity` over the peaks in A / mean over B (default `quantity: height`).
- `difference: [A, B]` = mean over A − mean over B (default `quantity: center_meV`).
- A and B are one peak name or a list of names.
- `quantity` is any per-peak result: `center_meV`, `center_nm`, `height`, `intensity`, `fwhm_meV`, `fwhm_nm`,
  `snr`, `tail_meV`. Use `intensity` for a ratio of integrated areas.
- A value counts as good (`<name>_ok`) only where every peak involved passed the quality check (and, for
  a ratio, the denominator is > 0).
- Each map is written to `results.csv`/`results.npz` (`<name>`, `<name>_ok`), as an extra row in
  `summary.png`, and under "Combined" in the report. Its value on the mean spectrum and its median are
  written to the log and to `fit_info.yaml`.

## Choosing peak shapes

1. Run with every peak set to `shape: lorentzian`, or with `peaks: []` to auto-detect X0. The
   console output and the HTML report include a **lineshape check**: each peak is refitted as
   Lorentzian, Gaussian, Voigt and trion on the 5 brightest points and their χ² values are compared.
   Voigt or trion is only recommended if it beats both simpler shapes by more than 2 %, because the
   extra parameter always lets it fit at least as well.
2. Set `shape: gaussian` or `shape: voigt` for the peaks you choose, and run `plmap run` again.

Each shape combination is written to its own folder (`fit_X0-L_XT-V/`, ...), so passes can be
compared side by side.

## Outputs (in `data.output`, default `<folder>/plmap_output/`)

| File | Contents |
|---|---|
| `spectra_plots/<run>_loc<k>.png` | step 1: averaged spectrum of every file |
| `spectra.npz` | cached averaged spectra (re-read only when raw files change) |
| `overview.png` | step 2: integrated-PL map + mean bright/dim spectra |
| `rc/…` | reflection contrast (`plmap rc`), see above |
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

## Reflection contrast (`plmap rc`)

Same folder layout, `scan.txt` and config file as PL (the `rc:` section; `background:` and `fit:`
are not used). Per point:

1. **R** = (spectrum − dark) / exposure, in counts/s per CCD row. `rc.dark: auto` takes the CCD
   offset from the red-most 3 nm, where the lamp is ~0; or give a number or a dark file.
2. **R0**, the background reflectance (`rc.r0`):
   - `auto` (default): the median R of the map's featureless points: those without a resonance,
     and with the most common reflectance level. They are drawn as green dots on the R map.
   - `[loc, loc, ...]`: points you choose. In the report, click featureless pixels and press
     **add this point**; it writes the `r0:` line to paste into the config.
   - a file (`R0_2D_spectrum.txt`, 2-column, or loc-style matrix): put on the map's scale by a
     single factor matched to the featureless points (`r0_scale: auto`), by the exposure ratio
     (`exposure`, with `r0_exposure_s`), or by a number.

   plmap warns if R0 itself carries a resonance (for example, picked points on the flake): every
   point would then show an inverted copy of it.
3. **RC** = R/R0 − 1, left blank where the lamp is below 3 % of its maximum.
4. **Resonance window**: where RC varies most across the map, ±8 nm (or `rc.window_nm`).
5. **Fit**: the exciton gives a dip or a dispersive "heart-beat" feature, fitted as
   RC(E) = A·Re[e^{iφ}·(Γ/2)/(E0 − E − iΓ/2)] + c0 + c1·(E − Emid). φ = 0 is purely dispersive,
   φ = −90° a dip. A is the peak-to-peak size of the feature. A grid search over (E0, Γ) gives start
   values (the model is linear in the rest), then a least-squares fit refines them. A fit is good
   if A/noise ≥ `rc.min_snr` (default 5) and E0 and Γ are not at their limits.

Outputs in `<output>/rc/`: `results.csv` (per point: stage and relative coordinates, whether it is
an R0 point, R, RC level, model-free feature size, E0 ± err, λ0, Γ ± err, A ± err, φ, SNR, good-fit
flag), `results.npz`, `rc_info.yaml`, `summary.png/.pdf` and `report.html` (maps of every quantity;
click a pixel for its RC with the fit and residual, and its R with R0; R0 picker).

## Method notes

- **Cosmic rays** are found in each CCD row as spikes above a 7-line running median along
  wavelength, and excluded from the row average. Unlike a comparison across CCD rows, this
  does not remove real PL when the spot covers only a few rows. A second pass removes
  leftover single-line spikes from the averaged spectrum. A `--` in a file (dropped pixel)
  is interpolated.
- **Background**, applied in order and each step optional:
  - a dark spectrum (`dark_file`)
  - a per-point offset, the mean of a signal-free band. By default plmap picks whichever 8-nm
    spectral edge (blue or red) carries less PL on the brightest points, and logs its choice
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
- **Quality flag**: a peak counts as good if SNR (height / residual noise) is at least `min_snr` (default 4), its
  centre and widths are not stuck at their limits, and its area is above 0. Rejected points are grey in
  the maps; their values remain in the CSV.
- **Speed**: points are fitted in parallel (`fit.workers: auto`). A 41×41 map takes a few
  minutes, and plotting about 1,700 per-file PNGs adds roughly 1-2 minutes on the first run only.

## Tests

```bash
pytest          # unit tests + an end-to-end run on a synthetic map with known peaks
```
