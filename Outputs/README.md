# Outputs

Processed results generated from the raw captures in [`Captures/`](../Captures) by the code in [`Code/`](../Code). These are the figures and numbers used in the report.

## Contents

| Folder | What it contains |
|---|---|
| `psf_mtf/` | PSF and MTF analysis of all six apertures at 1300 mm: measured PSF, fitted design, 2D MTF and radial MTF for each, plus the numbers in `psf_mtf.json` |
| `deconvolution/` | Wiener deconvolution results: restored Siemens stars, natural scenes at each depth, *K* sweeps and ring-fidelity scores |
| `report_figures/` | The figures as they appear in the report |

Each folder can be regenerated from the captures with the scripts in `Code/`.
