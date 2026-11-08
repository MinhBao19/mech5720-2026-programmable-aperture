"""PSF and MTF of all six apertures (paper / LCD x Levin / Nayar / Circle) from the 1300 mm PSFs.

Four panels per aperture: measured PSF, fitted design, 2D MTF (dB), and the radial MTF against the fitted design,
the ideal design and the measured noise floor.

  Measured PSF  mean of the registered frames of the 1300 mm PSF capture (green, full resolution). The source
                drifts during most captures, so frames are aligned by sub-pixel cross-correlation first.
                paper: local background subtracted, connected blob (dilated 3 px) kept.
                LCD:   edge background subtracted, then the leakage halo removed by model
                       (pipeline.decompose_lcd_psf). The MTF with the halo kept is also plotted.
  Fitted design the binary design fitted to the PSF (scale, rotation, flip, offset, Gaussian blur). LCD: fitted to
                the single design copy, then repeated along the measured diffraction orders.
  Ideal design  the same without the blur.
  Noise floor   MTF of half the difference of the two half-burst means, over the PSF's window and support,
                normalised by the PSF's total counts, after regressing out source flicker and drift.

Run:  py make_psf_mtf.py                  (paper, LCD, then the combined figure and RESULTS.md)
      py make_psf_mtf.py paper | lcd | combine
Output: output_psf_mtf/{paper,lcd}_{levin,nayar,circle}.png, mtf_all.png, psf_mtf.json, RESULTS.md
"""
import sys
import json
import subprocess

STEP = sys.argv[1] if len(sys.argv) > 1 else 'all'
if STEP == 'lcd' and '--lcd' not in sys.argv:
    sys.argv.append('--lcd')

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

import pipeline as P
import lab3_paper_aperture as L

OUT = L.HERE / 'output_psf_mtf'
D = 1300
MCOL, INK2, MUTED, GRID = L.MCOL, L.INK2, L.MUTED, L.GRID
plt.rcParams['savefig.dpi'] = 150


def radial(H, n):
    yy, xx = np.indices(H.shape) - n // 2
    rr = np.hypot(yy, xx).round().astype(int)
    prof = np.bincount(rr.ravel(), H.ravel()) / np.bincount(rr.ravel())
    f = np.arange(len(prof)) / n
    m = f <= 0.5
    return f[m], prof[m]


def mtf2d(psf, n):
    H = np.abs(np.fft.fftshift(P.psf_to_otf(psf / psf.sum(), (n, n))))
    return H / H.max()


def shift_of(a, b):
    """Sub-pixel shift (dy, dx) that moves b onto a, and the normalised correlation peak."""
    A, B = np.fft.fft2(a - a.mean()), np.fft.fft2(b - b.mean())
    c = np.real(np.fft.ifft2(A * np.conj(B)))
    c /= np.sqrt((a - a.mean()).var() * (b - b.mean()).var()) * a.size
    iy, ix = np.unravel_index(np.argmax(c), c.shape)
    ny, nx = c.shape

    def par(m1, c0, p1):
        den = m1 - 2 * c0 + p1
        return 0.0 if den == 0 else 0.5 * (m1 - p1) / den
    dy = iy + par(c[iy - 1, ix], c[iy, ix], c[(iy + 1) % ny, ix])
    dx = ix + par(c[iy, ix - 1], c[iy, ix], c[iy, (ix + 1) % nx])
    dy = dy - ny if dy > ny / 2 else dy
    dx = dx - nx if dx > nx / 2 else dx
    return np.array([dy, dx]), float(c[iy, ix])


def registered_frames(folder):
    """All frames of a PSF capture shifted onto their median position (two passes: against the plain mean, then
    the registered mean). Frames with a correlation peak < 0.5 x the median are dropped.
    Returns frames, shifts, number dropped."""
    gs = [L.green(folder, i)[0] for i in range(len(L.dng_files(folder)))]
    if L.LCD:
        gs = [L.column_fix(g) for g in gs]
    sm = ndi.gaussian_filter(np.mean(gs, 0), 3)
    cy, cx = np.unravel_index(np.argmax(sm), sm.shape)
    h = 400 if L.LCD else 100
    ys, xs = slice(max(cy - h, 0), cy + h), slice(max(cx - h, 0), cx + h)
    win = lambda g: ndi.gaussian_filter(g[ys, xs], 1.5) - np.median(g)
    ref = win(np.mean(gs, 0))
    for _ in range(2):
        sh, pk = zip(*[shift_of(ref, win(g)) for g in gs])
        sh, pk = np.array(sh), np.array(pk)
        keep = pk > 0.5 * np.median(pk)
        sh -= np.median(sh[keep], 0)
        al = [ndi.shift(g, s, order=1, mode='nearest') for g, s, k in zip(gs, sh, keep) if k]
        ref = win(np.mean(al, 0))
    return al, sh[keep], int((~keep).sum())


def noise_mtf(gs, centre, size, weight, total, n, template):
    """Radial |FFT| of the mean's noise: half the difference of the two half-burst means, inside the PSF window
    (weight = support), normalised by the PSF's total counts."""
    k = len(gs)
    diff = (np.mean(gs[:k // 2], 0) - np.mean(gs[k // 2:], 0)) / 2
    cy, cx = centre
    h = size // 2
    win = np.zeros((size, size))
    y0, x0 = cy - h, cx - h
    ys = slice(max(y0, 0), min(y0 + size, diff.shape[0]))
    xs = slice(max(x0, 0), min(x0 + size, diff.shape[1]))
    win[ys.start - y0:ys.stop - y0, xs.start - x0:xs.stop - x0] = diff[ys, xs]
    win *= weight
    # flicker and drift between the halves are signal changes: regress out a scaled PSF and its shift terms
    t = template * weight
    gy, gx = np.gradient(template)
    A = np.stack([t.ravel(), (gy * weight).ravel(), (gx * weight).ravel()], 1)
    win = win - (A @ np.linalg.lstsq(A, win.ravel(), rcond=None)[0]).reshape(win.shape)
    pad = np.zeros((n, n)); o = (n - size) // 2
    pad[o:o + size, o:o + size] = win
    H = np.abs(np.fft.fftshift(np.fft.fft2(pad))) / total
    f, c = radial(H, n)
    return f, ndi.uniform_filter1d(c, 5)


def crossing(f, C, level):
    i = np.where(C < level)[0]
    return float(f[i[0]]) if len(i) else None


def run(exp):
    L.SELECTION.update(json.load(open(L.CACHE / 'frame_selection.json')))
    OUT.mkdir(exist_ok=True)
    res = {}
    for m in L.MASKS:
        print(f'{exp} {m}', flush=True)
        gs, shifts, dropped = registered_frames(L.PSF[m][D])
        drift = float(np.ptp(shifts[:, 0]) ** 2 + np.ptp(shifts[:, 1]) ** 2) ** 0.5
        plain = L.green_mean(L.PSF[m][D])
        g = np.mean(gs, 0)
        if exp == 'paper':
            thr, info = L.extract_psf(g)
            sup = ndi.binary_dilation(thr > 0, iterations=3).astype(float)
            psf = info['raw'] * sup
            info['total_counts'] = float(psf.sum())
            psf = psf / psf.sum()
            pp, pinfo = L.extract_psf(plain)
            unreg = pinfo['raw'] * ndi.binary_dilation(pp > 0, iterations=3)
            n, show = 256, 96
            nf = noise_mtf(gs, info['centre'], psf.shape[0], sup, info['total_counts'], n, info['raw'])[1]
            fit = L.fit_design(psf, m)
            model = ndi.gaussian_filter(L.design_psf(m, fit, psf.shape[0], centred=False), fit['blur_sigma_px'])
            ideal = L.design_psf(m, fit, psf.shape[0], centred=False)
            with_halo = None
            halo_frac = None
        else:
            psf_h, info = L.extract_psf_lcd(g, 1024)
            ref, _ = L.extract_psf_lcd(L.green_mean(L.PSF[m][1000]), 1024)
            d = P.order_spacing(ref)                            # orders are cleanly separated at 1000 mm
            unreg = np.clip(P.decompose_lcd_psf(L.extract_psf_lcd(plain, 1024)[0], d)['chain'], 0, None)
            dec = P.decompose_lcd_psf(psf_h, d)
            psf = np.clip(dec['chain'], 0, None)
            halo_frac = float(1 - psf.sum() / psf_h.sum())
            with_halo = psf_h
            n, show = 1024, 320
            sup = (ndi.gaussian_filter(dec['chain'], 2) > 0.01 * dec['chain'].max()).astype(float)
            nf = noise_mtf(gs, info['centre'], 1024, sup,
                           info['total_counts'] * psf.sum() / psf_h.sum(), n, info['raw'])[1]
            blob = dec['blob'] / dec['blob'].sum()
            c = blob.shape[0] // 2
            fit = L.fit_design(blob[c - 160:c + 160, c - 160:c + 160], m)
            one = L.design_psf(m, fit, 1024, centred=False)

            def chain(x):
                return sum(w * P.shift_y(x, k * d) for w, k in zip(dec['weights'], dec['orders']))
            model = np.clip(chain(ndi.gaussian_filter(one, fit['blur_sigma_px'])), 0, None)
            ideal = np.clip(chain(one), 0, None)
            fit['order_spacing_px'] = float(d)
        Hm, Hd, Hi = mtf2d(psf, n), mtf2d(model, n), mtf2d(ideal, n)
        f, cm = radial(Hm, n)
        _, cd = radial(Hd, n)
        _, ci = radial(Hi, n)
        ch = radial(mtf2d(with_halo, n), n)[1] if with_halo is not None else None
        cu = radial(mtf2d(unreg, n), n)[1]
        corr = float(np.corrcoef(psf.ravel(), model.ravel())[0, 1])
        f_noise = crossing(f, cm - 2 * nf, 0)
        floor = float(np.median(nf[f > 0.05]))
        res[m] = dict(exp=exp, mask=m, distance_mm=D, frames=len(L.dng_files(L.PSF[m][D])),
                      exposure_us=L.exposure_us(L.PSF[m][D]), saturated_px=int((g >= L.SAT - L.BL).sum()),
                      scale_px_per_cell=fit['scale_px_per_cell'], blur_sigma_px=fit['blur_sigma_px'],
                      design_corr=corr, noise_floor=floor, f_noise_floor=f_noise, noise=nf.tolist(),
                      drift_px=drift, frames_dropped=dropped, unregistered=cu.tolist(),
                      mtf50=crossing(f, cm, 0.5), mtf10=crossing(f, cm, 0.1), halo_removed_frac=halo_frac,
                      order_spacing_px=fit.get('order_spacing_px'),
                      f=f.tolist(), measured=cm.tolist(), model=cd.tolist(), ideal=ci.tolist(),
                      with_halo=None if ch is None else ch.tolist())
        figure(exp, m, psf, model, Hm, f, cm, cd, ci, ch, cu, nf, n, show, res[m])
    J = OUT / 'psf_mtf.json'
    allres = json.load(open(J)) if J.exists() else {}
    allres.update({f'{exp} {m}': v for m, v in res.items()})
    json.dump(allres, open(J, 'w'), indent=1)


def crop(a, s):
    c = a.shape[0] // 2
    return a[c - s // 2:c + s // 2, c - s // 2:c + s // 2]


def figure(exp, m, psf, model, Hm, f, cm, cd, ci, ch, cu, nf, n, show, r):
    up = L.up
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.9), gridspec_kw=dict(width_ratios=[1, 1, 1.12, 1.5]))
    p = crop(psf, show)
    lab = 'measured PSF, halo removed by model' if exp == 'LCD' else 'measured PSF'
    axs[0].imshow(up(p / p.max()) ** 0.5, cmap='magma')
    nfr = r['frames'] - r['frames_dropped']
    axs[0].set_title(f'{lab}\n{nfr}-frame registered mean (drift {r["drift_px"]:.0f} px), {r["exposure_us"]} µs',
                     fontsize=8.5)
    q = crop(model, show)
    axs[1].imshow(up(q / q.max()) ** 0.5, cmap='magma')
    axs[1].set_title(f'fitted design ({r["scale_px_per_cell"]:.2f} px/cell, blur σ {r["blur_sigma_px"]:.1f} px)\n'
                     f'correlation with measured {r["design_corr"]:.2f}', fontsize=8.5)
    for ax in axs[:2]:
        ax.set_xticks([]); ax.set_yticks([])
        bar = 10 if exp == 'paper' else 50
        ax.plot([show * 0.06, show * 0.06 + bar], [show * 0.94] * 2, color='white', lw=2)
        ax.text(show * 0.06, show * 0.91, f'{bar} px', color='white', fontsize=7)
    db = 20 * np.log10(np.maximum(up(Hm), 1e-4))
    im = axs[2].imshow(db, cmap='viridis', vmin=-60, vmax=0, extent=(-0.5, 0.5, -0.5, 0.5))
    axs[2].set_title('measured 2D MTF (dB)', fontsize=8.5)
    axs[2].set_xlabel('cycles/px'); axs[2].set_ylabel('cycles/px')
    fig.colorbar(im, ax=axs[2], fraction=0.046, pad=0.03)
    ax = axs[3]
    col = MCOL[m]
    ax.plot(f, cm, color=col, lw=2, label='measured' + (' (halo removed)' if ch is not None else ''))
    if ch is not None:
        ax.plot(f, ch, color=col, lw=1.2, ls=':', label='measured, halo kept')
    if r['drift_px'] >= 3:
        ax.plot(f, cu, color=col, lw=1, ls='--', alpha=0.6, label='plain mean, not registered')
    ax.plot(f, cd, color=INK2, lw=1.2, label='fitted design (with blur)')
    ax.plot(f, ci, color=MUTED, lw=1.2, ls='--', label='ideal design' + (' + orders' if ch is not None else ''))
    ax.fill_between(f, 1e-6, nf, color='#e1e0d9', lw=0, label='noise floor (10-frame mean, measured)')
    ax.set_yscale('log'); ax.set_ylim(max(nf.min() / 3, 1e-5), 1.2); ax.set_xlim(0, 0.5)
    ax.set_xlabel('spatial frequency (cycles per px)'); ax.set_ylabel('MTF (radial mean)')
    ax.grid(True, which='major'); ax.legend(frameon=False, fontsize=7.5, loc='upper right')
    fn = r['f_noise_floor']
    ax.set_title(f'radial MTF; within 2x of the noise from {fn:.3f} cycles/px' if fn else
                 'radial MTF; stays above 2x the noise floor', fontsize=8.5)
    fig.suptitle(f'{exp} {m}, {D} mm', fontsize=10.5, x=0.02, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / f'{exp.lower()}_{m.lower()}.png', bbox_inches='tight'); plt.close(fig)


def combine():
    R = json.load(open(OUT / 'psf_mtf.json'))
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, exp in zip(axs, ('paper', 'LCD')):
        for m in MCOL:
            r = R[f'{exp} {m}']
            f = np.array(r['f'])
            ax.plot(f, r['measured'], color=MCOL[m], lw=2, label=f'{m} measured')
            ax.plot(f, r['ideal'], color=MCOL[m], lw=1, ls='--', alpha=0.8, label=f'{m} ideal design')
            ax.plot(f, r['noise'], color=MCOL[m], lw=0.8, ls=':', alpha=0.9)
        ax.set_yscale('log'); ax.set_ylim(1e-5, 1.2); ax.set_xlim(0, 0.5)
        ax.set_xlabel('spatial frequency (cycles per full-res px)')
        ax.grid(True)
        ax.set_title(f'{exp} apertures, {D} mm' + (' (halo removed by model)' if exp == 'LCD' else '') +
                     '\ndotted = noise floor of each', fontsize=9)
    axs[0].set_ylabel('MTF (radial mean)')
    axs[1].legend(frameon=False, fontsize=7.5, ncol=1, loc='upper right')
    fig.tight_layout()
    fig.savefig(OUT / 'mtf_all.png', bbox_inches='tight'); plt.close(fig)
    fmt = lambda v, p=3: '–' if v is None else f'{v:.{p}f}'
    T = ['# PSF and MTF of all six apertures (1300 mm)', '',
         'Code: `make_psf_mtf.py`. Figures: `paper_*.png`, `lcd_*.png` (measured PSF, fitted design, 2D MTF, '
         'radial MTF), `mtf_all.png` (all six). Numbers: `psf_mtf.json`.', '',
         'All PSFs are 10-frame means of the 1300 mm captures (green, full resolution). None is saturated.', '',
         '**The point source drifts during the captures**, so the frames are registered before averaging '
         '(sub-pixel cross-correlation). The drift over the burst is in the table. The plain '
         '(unregistered) mean is also plotted where the drift is 3 px or more.', '',
         '**LCD: the leakage halo is removed by a model, not by an all-black capture.** Each LCD PSF is fitted as '
         'one design copy repeated along the LCD diffraction orders plus a smooth halo '
         '(`pipeline.decompose_lcd_psf`), and only the copies are kept. This stands in for "minus all-black" '
         'subtraction. It removes the halo the model can see; real leakage that is not smooth would remain.', '',
         '| Aperture | Exposure | Drift (px) | Design scale (px/cell) | Fitted blur σ (px) | Design correlation | Halo removed | '
         'MTF50 | MTF10 | Noise floor (median above 0.05) | Within 2× of noise from |',
         '|---|---|---|---|---|---|---|---|---|---|---|']
    for k, r in R.items():
        hf = '–' if r['halo_removed_frac'] is None else f'{100 * r["halo_removed_frac"]:.0f} %'
        T.append(f'| {k} | {r["exposure_us"]} µs | {r["drift_px"]:.0f} | {r["scale_px_per_cell"]:.2f} | {r["blur_sigma_px"]:.1f} | '
                 f'{r["design_corr"]:.2f} | {hf} | '
                 f'{fmt(r["mtf50"])} | {fmt(r["mtf10"])} | {r["noise_floor"]:.1e} | {fmt(r["f_noise_floor"])} |')
    T += ['', 'Frequencies in cycles per full-resolution px. "Within 2× of noise from" is the first frequency where '
          'the measured MTF drops below twice the measured noise floor; above it the curve is mostly noise.'
          ' The noise floor is measured from the difference of the two 5-frame halves. Design correlation is between the '
          'measured PSF and the fitted design with blur, over the whole PSF window (LCD: including the orders).', '',
          'Limits of these figures:',
          '- The design fit caps its extra blur at σ = 3 px, and every fit reaches the cap: the real optical blur is '
          'larger, which is why the measured MTF falls below the fitted design from about 0.05 cycles/px.',
          '- LCD Nayar at 1300 mm: the design copies overlap along the order chain, so its decomposition and design '
          'fit are the least reliable of the six (correlation 0.68).',
          '- At 1300 mm the paper masks are only 2–6 px per design cell, so the fine structure of the design is '
          'below what the lens and pixels resolve; matching a ~190 px PSF needs a capture at that magnification.',
          '- There are no all-black LCD frames. The only black frames in the captures are the paper session\'s '
          'lens-cap dark frames (`Paper-Part0/LensBlackLevel`), already subtracted from every capture as fixed '
          'pattern noise. They contain no light, so they cannot stand in for an all-black LCD frame.']
    (OUT / 'RESULTS.md').write_text('\n'.join(T) + '\n', encoding='utf-8')
    print('done ->', OUT)


if __name__ == '__main__':
    if STEP == 'all':
        for s in ('paper', 'lcd'):
            subprocess.run([sys.executable, __file__, s], check=True)
        combine()
    elif STEP == 'combine':
        combine()
    else:
        run('LCD' if STEP == 'lcd' else 'paper')
