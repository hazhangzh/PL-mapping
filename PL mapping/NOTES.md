# Development notes / handoff

Status as of 2026-11-06 (v0.1.0 + real-data tuning + RC pipeline). Read with `README.md`.

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

## Decisions made with the user (2026-11-06, second round)

- **Scan coordinates** come from a per-map two-column `scan.txt` (start/end/points per axis, named
  as in the scan program, plus exposure_s). The program's "Steps" box is the number of points.
  The 11/04 PL and 11/05 RC maps share the same scan: x 3.108→3.128 mm, y −7.9785→−7.9585 mm,
  41 × 41 points, 0.5 µm steps. The earlier 1 µm/step assumption was wrong.
- **Maps are drawn in µm with the scan start at (0, 0)** (pure translation; `x_rel_um` is now
  signed, so a scan towards smaller stage values gives negative x).
- **PL: fit only the main exciton (X0)** for now; the low-energy band varies too much over the
  flake. `auto_detect: main` (default) picks the highest-energy peak with ≥ 30 % of the largest
  prominence and sets the fit range to 4 widths blue / 2.5 widths red of it.
- **RC**: R0 may come from a file, from points the user picks on the map, or automatically from
  featureless points. The sample is found by its resonance (dip / dispersive "heart-beat").

## Decisions made with the user (first round)

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

## First real on-flake map (2026-11-04, `11042026_Zihan_first_PLmap`)

41×41 points, 25 CCD rows, 691.6–768.0 nm, gate grounded, about 30 s per point. About 75 points are on the flake.
The full run takes about 90 s for the spectra step (first time, with PNGs) and about 35 s for the fits.

- **Config bug fixed:** YAML reads a date-named folder (`folder: 11042026`) as an int, and that crashed `config.load`.
  Paths are now converted with `str()`.
- **Offset band:** the red-most 8 nm (the old default) has a PL tail on the flake (+0.6–0.9 counts), so it pushed every
  point down by about 1 count and counted 268 points as "bright" when about 80 are. The CCD offset also drifts
  813–828 counts over the scan, as large as the PL (peaks of 10–100 counts). New default `offset_region_nm: null`
  = auto: it uses whichever 8-nm edge carries less PL, judged from the red-minus-blue difference (which drift does
  not affect) on bright against dim points (`background.choose_offset_band`). It picked blue here.
- **The first 5 points (loc1–5)** have a steeper CCD offset slope (blue−red about 5 counts, against 2.3 elsewhere): the
  camera had not settled at scan start. They looked like a faint signal strip in the overview. Suggest waiting before starting a scan.
- **Bright points** (the mean spectrum for auto-detect, the seed and the overview) were the top 10 % = 168 points, so about half were off-flake.
  Now: points more than 8 MAD above the map median, capped at 10 % (`background.bright_points`).
- **Auto-detect** gave `fwhm_meV: 1.0`, `window_nm: 2.0` for all but the first peak (scipy "prominence of 0"),
  because widths were measured on a different curve from the one the peaks were found on. Fixed. Widths of overlapping peaks still read narrow,
  but they only seed the mean fit.
- **min_snr 5 → 4:** off-flake noise fits reach SNR ≤ 2.1 (about 2000 peak fits). X0 at SNR 3–5 has a centre error
  of 0.12–0.17 meV and the same FWHM as the bright points, so 5 rejected real flake-edge points.
- **Spectrum:** X0 at 1734–1738 meV (gradient along the flake), FWHM about 4–8 meV; the lineshape check prefers
  Gaussian (L 1.03 against G 1.00). Below it, 1.66–1.71 eV is a cluster of narrow (2–4 meV) lines whose energies
  change from point to point, fitted for now as envelopes LE1 (about 1703 meV), LE2 (about 1685) and LE3 (about 1670, poorly
  constrained, good at only about 17 points). A small separate piece at y ≈ 9 µm has X0 at about 1730 meV and FWHM about 20 meV.
- **A cosmic-ray cluster** (loc262, CCD rows 19–21, about 13 lines wide) gets through the 7-line per-row filter and
  was fitted as a 1.2 meV "X0" at SNR 4.1. Fixed in the config with `min_fwhm_meV: 2` on X0. A cross-row
  check for multi-line clusters is a possible improvement.
- Scan coordinates were not recorded with the data. The config assumes 1 µm/step.

## RC map (2026-11-05, `11052026_Zihan_RCmap`)

41×41 points, 4 CCD rows (Y 265–268), 5 s per point, dark ~809 counts (red edge). Runs in about 20 s
(6 s reading, about 10 s fitting).
- The X0 resonance traces a diagonal band about 5 µm wide across the whole map, larger than the
  PL-bright part. E0 = 1735–1740 meV with a gradient, Γ ≈ 5–8 meV, A up to about 1.8, φ ≈ 0–40°
  (mostly dispersive). A region at the bottom has E0 ≈ 1722 meV and a different phase; it is
  worth checking against the micrograph.
- `r0: auto`, the R0 file (scale 0.483 needed; exposure unknown) and 9 picked off-flake points
  all give E0 = 1736.3 meV, Γ = 6.4–6.6 meV, 677–712 good fits.
- The featureless region is not uniform: RC on it spreads from −0.25 to +0.1, with darker corners
  at the bottom left and top right, so different background materials are probably present.
  Picking R0 points next to the area of interest is better than `auto` for quantitative A.
- The first points and the row near y ≈ 10.5 µm show lamp/camera level steps (also in PL).
- R0 files: `R0_1D` (binned) and `R0_2D` (4 rows × 2048, transposed) differ by ×1.6, and neither
  matches the map by exposure alone, so exposures should be recorded for R0 too.

## Multi-peak fit with shift tracking (2026-11-18, `11162026_wse2 reconnected`)

Config `plmap_11162026_multipeak.yaml`: X0 (voigt), XT1 and XT2 (gaussian), XD (gaussian),
P1674 (voigt), offsets from X0, `range_meV: [-62, 15]`. 94 points carry the pattern; X0 is 1719-1737
meV over the flake (gradient along it), but the offsets are constant: XT1 −30.1, XT2 −37.1,
XD −41.5, P1674 −50.5 meV (5-95 % spread ≤ ±2 meV). Good fits: X0 77, XT1 79, XT2 76, XD 38,
P1674 82. Run time 17 s.
- Trion recoil tail fitted at its floor (< 0.1 meV), so the trions are Gaussian here; the lineshape check agrees.
- P1674 as `trion` (it does have a low-energy tail) makes XD run to its width limit: model
  degenerate in the 1.680-1.686 eV region, so P1674 stays Voigt.
- Pitfalls found and fixed: (1) an anchor from the *mean* of shifted points can lock one trion
  spacing off, so it now uses the median over individual bright spectra; (2) the first template fit
  must use the *aligned* mean, or the narrow windows catch the wrong peaks; (3) per-trial χ² over a
  moving window favours empty windows, so the shift scan scores every trial on the same region.
- loc1-11 (start of the scan, top row) show the full flake spectrum: the spot was probably still on
  the flake before the stage moved to the scan start.
- 2 weak edge points (loc688, loc729; only X0 good) have shifts at the +12 meV limit.
- 2026-11-20: added `fit.combined` (ratio / difference maps). On this map, over the ~68 points
  where X0, XT1 and XT2 are all good: mean(XT1, XT2)/X0 height ratio has median 3.68 (5-95 %: 1.27-4.57).
  XT1−X0 has median −30.1 meV (−32.4…−29.3); XT2−X0 has median −37.1 meV (−39.1…−35.3).

## Next steps

0b. RC: optionally normalise each point's R by a featureless band (lamp drift), and compare the
    RC and PL maps point by point (same scan grid).
0c. The Voigt option in the lineshape check uses `min_fwhm_meV` for both of its widths. With
    `min_fwhm_meV: 2` it cannot reduce to a pure Gaussian, so it can read worse than Gaussian
    (seen for X0: G 1.000, V 1.052).
0d. With only X0 fitted, the tail of an unfitted neighbour (trion) inside the fit range biases the
    lineshape check: the synthetic demo (Lorentzian X0, Voigt trion 30 meV below) recommends Voigt
    (L 1.06 against V 1.00). On the real 11/04 map the preference is much stronger (L 1.36 against G 1.00).
    Options: a quadratic baseline, or a narrower red side of the auto range.

0. Low-energy band: decided to fit X0 only for now (see above). Revisit with a per-point
   narrow-line finder once the gate series is in.

1. ~~Run on a real map with `peaks: []`~~ done (above).
2. Background on real data: offset band done (auto). Still open:
   - whether `substrate: auto` (dimmest 30 %) is right, or whether a measured substrate or
     dark spectrum works better
   - whether the default offset band (red-most 8 nm) is free of signal
3. Check the cosmic-ray rejection count and the `min_snr` threshold on real spectra.
4. Check that the map orientation and `flip_x` match real space, for example against a
   microscope image.
5. Run on Windows for real: parallel workers and paths with spaces.
6. ~~Add the RC pipeline~~ done (`plmap rc`, `plmap/rc.py`, `plmap/rc_report.py`).
