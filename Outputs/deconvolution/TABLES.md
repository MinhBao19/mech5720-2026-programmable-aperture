# Best-performing results, 5 Nov captures

Made by `make_best_results.py`. One folder per variation, each with:
- `01_psfs.png`: the PSFs, with the one used for the star outlined.
- `02_star_candidates.png`: every candidate's score.
- `03_star.png`: the captured and restored star.
- `04_K_sweep.png`: the K sweep with the chosen PSF.
- `05_mtf.png`: the MTF before and after.
- `06_scene_*.png`: the natural scenes at each distance.
- `images/`: every captured and restored image on its own, full frame.
- `summary.json`: the numbers below.

Scores are ring fidelity on the 1300 mm Siemens star: the correlation with an ideal 36-spoke star, from 0 (no spokes) to 1. Paper runs at full resolution on the green channel. LCD runs at half resolution on the Bayer planes. Frequencies are in cycles per full-resolution pixel.

## 1. Chosen configuration

| Variation | Method | Frames | PSF for star | PSFs for scenes (1000 / 1150 / 1300 mm) | Parameter | Note |
|---|---|---|---|---|---|---|
| paper Levin | Wiener | 10-frame mean | 1300 mm | 1000 / 1150 / 1300 | K = 9.8e-04 | – |
| paper Nayar | Wiener | 10-frame mean | 1300 mm | 1000 / 1150 / 1300 | K = 3.7e-04 | – |
| paper Circle | Wiener | 10-frame mean | 1150 mm | 1000 / 1150 / 1150 | K = 7.1e-04 | – |
| LCD Levin | M4 + boundary | 10-frame mean | 1300 mm | 1000 / 1150 / 1300 | K = 1.4e-05 | M5 TV scores 0.186 vs M4 0.175 (within 0.02); M4 kept (TV adds line artefacts on scenes) |
| LCD Nayar | M4 + boundary | 10-frame mean | 1300 mm | 1000 / 1150 / 1300 | K = 1.0e-06 | – |
| LCD Circle | M4 + boundary | 10-frame mean | 1300 mm | 1300 / 1300 / 1300 | K = 6.5e-05 | – |

## 2. Star results (1300 mm)

| Variation | Fidelity: captured | Fidelity: restored | Gain | MTF50: captured → restored | MTF10: captured → restored |
|---|---|---|---|---|---|
| paper Levin | 0.279 | **0.628** | 0.35 | – → 0.066 | 0.022 → 0.109 |
| paper Nayar | 0.197 | **0.626** | 0.43 | 0.019 → 0.094 | 0.033 → 0.142 |
| paper Circle | 0.390 | **0.547** | 0.16 | 0.019 → 0.065 | 0.045 → 0.080 |
| LCD Levin | 0.006 | **0.175** | 0.17 | – → 0.033 | – → 0.119 |
| LCD Nayar | -0.005 | **0.208** | 0.21 | – → 0.044 | – → 0.346 |
| LCD Circle | 0.033 | **0.302** | 0.27 | – → 0.025 | – → 0.071 |

MTF is the 36th angular harmonic of the star divided by that of the in-focus no-aperture star. MTF50 / MTF10 are the lowest frequency where it falls below 0.5 / 0.1. "–" means it never reaches that level, which for LCD captures means the contrast is at the noise floor at every frequency. A restored MTF can exceed the captured one at frequencies where the restoration also amplifies noise, so read it alongside fidelity.

## 3. Every candidate (1300 mm star fidelity, best parameter each)

**paper**

| Candidate | Levin | Nayar | Circle |
|---|---|---|---|
| Wiener, 1 frame, PSF 1000 mm | 0.418 | 0.263 | 0.436 |
| Wiener, 1 frame, PSF 1150 mm | 0.487 | 0.287 | 0.508 |
| Wiener, 1 frame, PSF 1300 mm | 0.545 | 0.590 | 0.183 |
| Wiener, 10-frame mean, PSF 1000 mm | 0.456 | 0.266 | 0.451 |
| Wiener, 10-frame mean, PSF 1150 mm | 0.545 | 0.294 | **0.547** |
| Wiener, 10-frame mean, PSF 1300 mm | **0.628** | **0.626** | 0.184 |

Captured: Levin 0.279, Nayar 0.197, Circle 0.390. Bold = chosen.

**LCD**

| Candidate | Levin | Nayar | Circle |
|---|---|---|---|
| M1 G, 1 frame, PSF 1000 mm | 0.070 | 0.010 | 0.111 |
| M2 G, 10-frame mean, PSF 1000 mm | 0.080 | 0.016 | 0.131 |
| M3 RGB joint Wiener, PSF 1000 mm | 0.077 | 0.029 | 0.141 |
| M4 + boundary, PSF 1000 mm | 0.113 | 0.063 | 0.204 |
| M1 G, 1 frame, PSF 1150 mm | 0.081 | 0.021 | 0.058 |
| M2 G, 10-frame mean, PSF 1150 mm | 0.086 | 0.015 | 0.058 |
| M3 RGB joint Wiener, PSF 1150 mm | 0.089 | 0.014 | 0.091 |
| M4 + boundary, PSF 1150 mm | 0.139 | 0.035 | 0.174 |
| M1 G, 1 frame, PSF 1300 mm | 0.083 | 0.079 | 0.182 |
| M2 G, 10-frame mean, PSF 1300 mm | 0.086 | 0.082 | 0.199 |
| M3 RGB joint Wiener, PSF 1300 mm | 0.121 | 0.127 | 0.217 |
| M4 + boundary, PSF 1300 mm | **0.175** | **0.208** | **0.302** |
| M5 RGB TV + boundary, PSF 1300 mm | 0.186 | 0.132 | 0.295 |

Captured: Levin 0.006, Nayar -0.005, Circle 0.033. Bold = chosen.

## 4. PSFs

| Variation | Design scale (px/cell) at 1000 / 1150 / 1300 mm | Design-fit correlation at 1000 / 1150 / 1300 | Light in straight-through copy / other diffraction copies / halo (star PSF) |
|---|---|---|---|
| paper Levin | 1.47 / 2.81 / 3.80 | 0.98 / 0.93 / 0.89 | – |
| paper Nayar | 1.43 / 1.58 / 2.37 | 0.98 / 0.98 / 0.96 | – |
| paper Circle | 2.10 / 2.94 / 5.55 | 0.99 / 0.99 / 0.98 | – |
| LCD Levin | 2.39 / 3.82 / 5.99 | 0.80 / 0.76 / 0.77 | 8% / 24% / 68% |
| LCD Nayar | 3.58 / 12.17 / 21.32 | 0.72 / 0.76 / 0.86 | 6% / 41% / 54% |
| LCD Circle | 3.30 / 3.21 / 4.27 | 0.84 / 0.72 / 0.73 | 10% / 18% / 72% |

LCD design scales come from the model with the diffraction copies (Part 6). Nayar 1150 / 1300 mm are unreliable there. The LCD light split is from `analyse_lcd_design_vs_grid.py`; the halo may include some room light (see the ghosting discussion).

## 5. Raw star contrast on the three Part 7 circles (Michelson, captured)

| Variation | Centre circle | Middle circle | Edge circle | Star centre (row, col), R (full-res px) | Centre from |
|---|---|---|---|---|---|
| paper Levin | 0.034 | 0.047 | 0.193 | (570.8, 741.5), 414 | picked |
| paper Nayar | 0.031 | 0.179 | 0.509 | (651.6, 760.6), 412 | picked |
| paper Circle | 0.027 | 0.224 | 0.507 | (520.2, 754.5), 403 | picked |
| LCD Levin | 0.007 | 0.010 | 0.013 | (596.5, 840.9), 404 | automatic |
| LCD Nayar | 0.011 | 0.008 | 0.014 | (629.5, 879.5), 394 | picked |
| LCD Circle | 0.016 | 0.016 | 0.021 | (591.3, 837.5), 406 | picked |
