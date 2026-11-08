# Coded aperture imaging: paper and LCD apertures

This code processes the Lab 3 captures for two kinds of coded aperture:

- laser-cut **paper** masks
- the same patterns shown on an **LCD**

Each kind comes in three designs: Levin, Nayar and a plain circle.

The code works from the raw DNGs. It covers:

- the lab steps (Parts 0–8, Bonus 2 and Part 9)
- the best reconstruction for each aperture
- the PSF and MTF of each aperture
- a breakdown of how much of each LCD PSF comes from the displayed design and how much from the LCD pixel grid

## Setup

You need Python 3.11 or later.

```
pip install -r requirements.txt
```

The code expects the capture folder `Captures` to sit next to the repository folder:

```
<parent>/
├── Captures/
│   ├── PaperAperture/PaperAperture/      Paper-Part0 (dark frames), Paper-Part1, Levin - Paper, Nayar - Paper, Circle - Paper
│   ├── LCDApertures/LCDApertures/        LCD-Part1, LCDLevin, LCDNayar, LCDCircle
│   └── Part9_extract/Part9/              Part 9 depth scenes (extracted from Part9.zip)
└── coded_aperture_lab3/                  this repository
```

If the data is somewhere else, set `LAB3_DATA` to the path of `Captures`. In PowerShell:

```
$env:LAB3_DATA = "D:\path\to\Captures"
```

Each capture folder holds a `dng/` subfolder with 10 frames. The code reads only those DNG files.

## Running

Run each script from this folder, in this order. Each script reads what the scripts before it saved.

| Step | Command | What it does | Output |
|---|---|---|---|
| 1 | `py lab3_paper_aperture.py` | Paper lab, Parts 0–8 and Bonus 2 | `output_lab3_paper/` |
| 2 | `py lab3_paper_aperture.py --lcd` | LCD lab, same steps | `output_lab3_lcd/` |
| 3 | `py compare_lcd_paper.py` | Paper vs LCD comparison figures | `output_lab3_compare/` |
| 4 | `py analyse_lcd_design_vs_grid.py lcd`<br>`py analyse_lcd_design_vs_grid.py paper` | How much of each LCD PSF comes from the design and how much from the LCD grid | `output_lcd_design_vs_grid/` |
| 5 | `py make_best_results.py` | Best method and PSF for each aperture, with figures and tables | `output_best/` |
| 6 | `py make_psf_mtf.py` | PSF and MTF figures for all six apertures | `output_psf_mtf/` |
| 7 | `py run_part9_depth.py paper`<br>`py run_part9_depth.py lcd` | Part 9: depth map and all-in-focus image | `output_part9_paper/`, `output_part9_lcd/` |

Run times on a 14-core machine are roughly:

- paper lab: about 3 minutes; LCD lab: about 10 minutes
- `make_best_results.py`: 15–25 minutes
- the rest: a few minutes each

Results are cached in each output folder's `cache/`. The LCD methods cache theirs in `output_lcd_cache/`. Delete a cache to recompute it, or add `--fresh` to the lab script. If you change a lab result, such as a star centre, rerun every later step.

Some scripts also take arguments to run one part only:

- `make_best_results.py paper | lcd | report`
- `make_psf_mtf.py paper | lcd | combine`
- `run_part9_depth.py lcd --m5` uses TV (M5) instead of M4.
- `run_part9_depth.py lcd --quarter` runs at quarter resolution.

### Siemens star centres

The star centre sets every star score. Hand-picked centres are stored in `star_centres.json` and are used instead of the automatic search. To change them:

```
py pick_star_centres.py          # paper captures
py pick_star_centres.py lcd      # LCD captures
```

Each capture opens in a window. The keys are listed at the top of `pick_star_centres.py`. After picking, rerun from step 1 or 2.

## Files

| File | Contents |
|---|---|
| `lab3_paper_aperture.py` | The lab steps, plus the shared loading, PSF extraction, design fitting and star measurement that the other scripts import. `--lcd` switches between the paper and LCD captures. |
| `lcd_methods.py` | The LCD reconstruction methods M1–M5 on half-resolution colour planes (library, not run directly) |
| `multichannel.py` | Joint RGB Wiener and TV solvers with unknown out-of-frame content (used by M4 and M5) |
| `pipeline.py` | Wiener deconvolution, the LCD PSF model (design × diffraction orders + halo), Siemens-star measures |
| `rawio.py` | DNG reader (10-bit packed, GBRG) |
| `make_best_results.py` | Scores every method × PSF on the 1300 mm star and applies the winner to the star and scenes |
| `make_psf_mtf.py` | Per-aperture figure: measured PSF, fitted design, 2D MTF, radial MTF with the noise floor |
| `analyse_lcd_design_vs_grid.py` | LCD PSF light budget, similarity between masks, deconvolution with swapped PSFs |
| `compare_lcd_paper.py` | Paper and LCD side by side, from the two lab runs |
| `run_part9_depth.py` | Part 9 depth from defocus and all-in-focus image |
| `pick_star_centres.py` | Interactive star-centre picker |
| `star_centres.json` | Hand-picked star centres (raw sensor coordinates) |
| `designs/` | The Levin and Nayar mask designs (13 × 13 grids) |

## Methods in brief

**Preprocessing.** For every frame the code:

1. subtracts the black level (16) from the raw mosaic
2. subtracts the fixed pattern noise, the mean of the paper session's lens-cap dark frames
3. extracts the colour channels

Paper results use the full-resolution green channel. LCD captures also get a column-offset correction.

**Single frames.** Where the lab asks for one frame, `select_frame` picks one automatically: the frame most like the median of the capture, unsaturated and with the source in place for PSFs. The choices are saved to `frame_selection.json`.

**LCD PSFs.** These are extracted in a 1024 px window, because the LCD pixel grid acts as a diffraction grating. It repeats the aperture's blur along a vertical chain of orders, and light leaking through the 'off' pixels forms a wide halo. `pipeline.decompose_lcd_psf` splits a PSF into those three parts.

**Reconstruction.**

- Paper: Wiener deconvolution of the green channel.
- LCD: five methods.
  - M1: green, single frame
  - M2: green, 10-frame mean
  - M3: joint RGB Wiener
  - M4: M3 plus unknown out-of-frame content
  - M5: RGB total variation plus unknown out-of-frame content

  M3 helps because each colour's diffraction orders fall at a different spacing, so the colours lose different frequencies.

**Scoring.** Results are scored by ring fidelity on the 1300 mm Siemens star. This is the mean correlation between the angular profile on each circle and an ideal 36-spoke star. It runs from 0 (no spokes) to 1.
