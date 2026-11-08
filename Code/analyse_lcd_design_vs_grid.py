"""How much of an LCD-aperture PSF (and of its deconvolution) is set by the displayed aperture design, and how
much by the LCD pixel matrix (diffraction-order chain + leakage halo)?  5 Nov captures, lab extraction.

  1. Energy budget: each LCD PSF split (pipeline.decompose_lcd_psf) into the design image in the zero order,
     the copies in the other diffraction orders, and the leakage halo.
  2. Similarity of the three masks' PSFs at each distance (peak normalised cross-correlation, any shift):
     LCD vs paper, and LCD with the matrix pattern removed (zero-order blob only).
  3. Comb transfer function: how deep the order chain cuts the vertical spectrum, whatever the design.
  4. Deconvolution swap: each 1300 mm star deconvolved (lab green Wiener, best K) with every mask's 1300 mm
     PSF; for LCD also with its own design stripped of the matrix pattern (blob only, blob x orders).

Run:  py analyse_lcd_design_vs_grid.py lcd   then   py analyse_lcd_design_vs_grid.py paper
      ->  output_lcd_design_vs_grid/{lcd,paper}.json (+ PSFs / blobs as .npz)
"""
import sys
import json

MODE = sys.argv[1] if len(sys.argv) > 1 else 'lcd'
if MODE == 'lcd':
    sys.argv.append('--lcd')

import numpy as np
from scipy import ndimage as ndi

import lab3_paper_aperture as L
import pipeline as P

OUT = L.HERE / 'output_lcd_design_vs_grid'
D = 1300
KS = L.KS[::2]
K_LOCATE = (1e-6, 1e-5, 1e-4) if MODE == 'lcd' else (3e-3,)     # K values at which the star is located


def ncc_max(a, b):
    """Peak normalised cross-correlation over all shifts (same-size centred arrays)."""
    a = a - a.mean(); b = b - b.mean()
    n = [2 * s for s in a.shape]
    c = np.real(np.fft.ifft2(np.fft.fft2(a, n) * np.conj(np.fft.fft2(b, n))))
    return float(c.max() / np.sqrt((a ** 2).sum() * (b ** 2).sum()))


def fid(x, c, R, n_r=30):
    phi = P.star_phase(x, c, L.N_SPOKES, 0.6 * R)
    return P.ring_fidelity(x, c, L.N_SPOKES, phi, 0.08 * R, 0.9 * R, n_r=n_r)


def locate(x, c, R, span=40, step=4):
    """Find the star again on a result (the PSF's centring shifts it)."""
    best, bc = -2, c
    for dy in range(-span, span + 1, step):
        for dx in range(-span, span + 1, step):
            cc = (c[0] + dy, c[1] + dx)
            f = fid(x, cc, R, n_r=8)
            if f > best:
                best, bc = f, cc
    return L.refine_centre_fidelity(x, bc, R, half=3.0, step=1.0)


def deconv_score(img, psf, c, R):
    wb = L.wiener_bank(img, psf)
    cc = max((locate(wb(K), c, R) for K in K_LOCATE), key=lambda q: max(fid(wb(K), q, R) for K in K_LOCATE))
    f = np.array([fid(wb(K), cc, R) for K in KS])
    i = int(f.argmax())
    return float(f[i]), float(KS[i])


def comb_tf(w, ks, d, n=4096):
    """|sum_k w_k exp(-2 pi i f k d)| / its DC value over one period of the vertical frequency."""
    f = np.arange(n) / n / d                      # one period: 0 .. 1/d cyc/px
    W = np.abs(sum(wk * np.exp(-2j * np.pi * f * k * d) for wk, k in zip(w, ks)))
    return W / W[0]


def main():
    OUT.mkdir(exist_ok=True)
    L.SELECTION.update(json.load(open(L.CACHE / 'frame_selection.json')))
    geo = json.load(open(L.OUT / 'results.json'))['part7']
    psfs = {}
    for m in L.MASKS:
        for d in L.DISTANCES:
            g, _ = L.best_green(L.PSF[m][d], 'psf')
            psfs[(m, d)] = L.extract_psf(g)[0]
    res = dict(mode=MODE)

    # energy budget and comb transfer function (LCD)
    decs = {}
    if MODE == 'lcd':
        sp = L.lcd_order_spacing(psfs)
        budget, tf = {}, {}
        for (m, d), p in psfs.items():
            dec = P.decompose_lcd_psf(p, sp)
            decs[(m, d)] = dec
            w, ks, blob = dec['weights'], dec['orders'], dec['blob']
            tot = p.sum()
            zero = w[ks == 0][0] * blob.sum() / tot
            chain = dec['chain'].sum() / tot
            halo = dec['halo'].sum() / tot
            budget[f'{m} {d}'] = dict(zero_order_design=float(zero), other_orders=float(chain - zero),
                                      halo=float(halo), unexplained=float(1 - chain - halo))
            W = comb_tf(w, ks, sp)
            tf[f'{m} {d}'] = dict(min=float(W.min()), frac_below_0p3=float((W < 0.3).mean()))
        res['order_spacing_px'] = sp
        res['energy_budget'] = budget
        res['comb_tf'] = tf

    # similarity between masks
    pairs = [('Levin', 'Nayar'), ('Levin', 'Circle'), ('Nayar', 'Circle')]
    sim = {}
    for d in L.DISTANCES:
        for a, b in pairs:
            sim[f'{a}-{b} {d}'] = ncc_max(psfs[(a, d)], psfs[(b, d)])
    res['psf_similarity'] = sim
    if MODE == 'lcd':
        res['blob_similarity'] = {f'{a}-{b} {d}': ncc_max(decs[(a, d)]['blob'], decs[(b, d)]['blob'])
                                  for d in L.DISTANCES for a, b in pairs}
    # same mask at 1000 vs 1300 mm, for scale
    res['psf_similarity_same_mask'] = {f'{m} 1000-1300': ncc_max(psfs[(m, 1000)], psfs[(m, 1300)]) for m in L.MASKS}

    # deconvolution swap on the 1300 mm stars
    swap = {}
    for m in L.MASKS:
        if not geo['star_located'].get(m, True):
            continue
        g, _ = L.best_green(L.STAR[m], 'scene')
        img = L.norm_img(g)
        c, R = tuple(geo['star_geometry'][m]['centre_rc']), geo['star_geometry'][m]['R_px']
        row = {'captured': float(fid(img, c, R))}
        for pm in L.MASKS:
            row[f'{pm} PSF'] = deconv_score(img, psfs[(pm, D)], c, R)
        if MODE == 'lcd':
            dec = decs[(m, D)]
            blob = dec['blob'] / dec['blob'].sum()
            chain = dec['chain'] / dec['chain'].sum()
            row['own design, no LCD pattern (blob only)'] = deconv_score(img, blob, c, R)
            row['own design x orders, no halo'] = deconv_score(img, chain, c, R)
            disc = np.zeros_like(blob)
            yy, xx = np.indices(blob.shape) - blob.shape[0] // 2
            r_eq = np.sqrt((blob > 0.05 * blob.max()).sum() / np.pi)
            disc[np.hypot(yy, xx) <= r_eq] = 1
            dd = sum(wk * P.shift_y(ndi.gaussian_filter(disc, 3), k * dec['d']) for wk, k in zip(dec['weights'], dec['orders']))
            # same light in the chain as measured, measured halo added
            dd = np.clip(dd / dd.sum() * dec['chain'].sum() + dec['halo'], 0, None)
            row['plain disc x own orders + halo'] = deconv_score(img, dd / dd.sum(), c, R)
        swap[m] = row
        print(m, {k: (v if isinstance(v, float) else round(v[0], 3)) for k, v in row.items()}, flush=True)
    res['deconv_swap_1300'] = swap
    json.dump(res, open(OUT / f'{MODE}.json', 'w'), indent=1)
    np.savez_compressed(OUT / f'{MODE}_psfs.npz', **{f'{m}_{d}': p.astype(np.float32) for (m, d), p in psfs.items()},
                        **{f'blob_{m}_{d}': v['blob'].astype(np.float32) for (m, d), v in decs.items()})
    print(json.dumps({k: v for k, v in res.items() if k != 'deconv_swap_1300'}, indent=1))


if __name__ == '__main__':
    main()
