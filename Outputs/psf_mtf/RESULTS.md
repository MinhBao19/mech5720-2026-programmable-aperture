# PSF and MTF of all six apertures (1300 mm)

Code: `make_psf_mtf.py`. Figures: `paper_*.png`, `lcd_*.png` (measured PSF, fitted design, 2D MTF, radial MTF), `mtf_all.png` (all six). Numbers: `psf_mtf.json`.

All PSFs are 10-frame means of the 1300 mm captures (green, full resolution). None is saturated.

**The point source drifts during the captures**, so the frames are registered before averaging (sub-pixel cross-correlation). The drift over the burst is in the table. The plain (unregistered) mean is also plotted where the drift is 3 px or more.

**LCD: the leakage halo is removed by a model, not by an all-black capture.** Each LCD PSF is fitted as one design copy repeated along the LCD diffraction orders plus a smooth halo (`pipeline.decompose_lcd_psf`), and only the copies are kept. This stands in for "minus all-black" subtraction. It removes the halo the model can see; real leakage that is not smooth would remain.

| Aperture | Exposure | Drift (px) | Design scale (px/cell) | Fitted blur σ (px) | Design correlation | Halo removed | MTF50 | MTF10 | Noise floor (median above 0.05) | Within 2× of noise from |
|---|---|---|---|---|---|---|---|---|---|---|
| paper Levin | 500 µs | 6 | 3.83 | 3.0 | 0.91 | – | 0.016 | 0.027 | 3.0e-04 | 0.199 |
| paper Nayar | 500 µs | 11 | 2.40 | 3.0 | 0.96 | – | 0.020 | 0.035 | 2.8e-04 | 0.262 |
| paper Circle | 500 µs | 30 | 5.55 | 3.0 | 0.97 | – | 0.016 | 0.023 | 2.0e-04 | 0.113 |
| LCD Levin | 60000 µs | 1 | 6.26 | 3.0 | 0.74 | 68 % | 0.003 | 0.008 | 9.5e-05 | 0.162 |
| LCD Nayar | 60000 µs | 21 | 8.41 | 3.0 | 0.68 | 52 % | 0.002 | 0.004 | 5.6e-05 | 0.081 |
| LCD Circle | 60000 µs | 10 | 3.38 | 3.0 | 0.80 | 72 % | 0.003 | 0.011 | 1.3e-04 | 0.255 |

Frequencies in cycles per full-resolution px. "Within 2× of noise from" is the first frequency where the measured MTF drops below twice the measured noise floor; above it the curve is mostly noise. The noise floor is measured from the difference of the two 5-frame halves. Design correlation is between the measured PSF and the fitted design with blur, over the whole PSF window (LCD: including the orders).

Limits of these figures:
- The design fit caps its extra blur at σ = 3 px, and every fit reaches the cap: the real optical blur is larger, which is why the measured MTF falls below the fitted design from about 0.05 cycles/px.
- LCD Nayar at 1300 mm: the design copies overlap along the order chain, so its decomposition and design fit are the least reliable of the six (correlation 0.68).
- At 1300 mm the paper masks are only 2–6 px per design cell, so the fine structure of the design is below what the lens and pixels resolve; matching a ~190 px PSF needs a capture at that magnification.
- There are no all-black LCD frames. The only black frames in the captures are the paper session's lens-cap dark frames (`Paper-Part0/LensBlackLevel`), already subtracted from every capture as fixed pattern noise. They contain no light, so they cannot stand in for an all-black LCD frame.
