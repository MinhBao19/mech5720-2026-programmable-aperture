"""LCD reconstruction methods M1-M5 on half-resolution Bayer planes (every sample a real R, G or B value).

  M1  green, 1 frame          Wiener (as in the lab)
  M2  green, 10-frame mean    Wiener
  M3  joint RGB Wiener        each colour's diffraction orders sit at a different spacing, so its OTF notches
                              fall at different frequencies
  M4  + unknown boundary      M3 with out-of-frame content modelled (masked Wiener, multichannel.py)
  M5  RGB TV + boundary       total-variation prior instead of the Wiener K

PSFs: the lab's LCD extraction (1024 px full-res window with the leakage halo) on each colour of the demosaiced
10-frame mean, all cropped at the green PSF's centre, then binned 2x2. Scenes: paper-session dark frames
subtracted from the mosaic, then a per-plane column correction. Reconstructions are cached in output_lcd_cache/.

Import this module only after '--lcd' is in sys.argv (lab3_paper_aperture reads its mode at import).
"""
import json

import numpy as np
from scipy import ndimage as ndi

import rawio
import pipeline as P
import multichannel as MC
import lab3_paper_aperture as L

CACHE = L.HERE / 'output_lcd_cache'
KS = np.geomspace(1e-6, 1e-1, 16)
LAMS = [1e3, 3e3, 1e4, 3e4, 1e5]
COLFIX_W = 15                                   # half-res px
GBRG = (1, 2, 0, 1)


def mosaic_stack(folder):
    """Black-level and FPN corrected raw mosaics of every frame."""
    fpn = L.fpn()
    out = []
    for f in L.dng_files(folder):
        m, cfa = rawio.read_bayer(f)
        assert tuple(cfa) == GBRG
        out.append(m - rawio.BLACK_LEVEL - fpn)
    return np.stack(out)


def planes(m):
    return {'R': m[1::2, 0::2], 'G': 0.5 * (m[0::2, 0::2] + m[1::2, 1::2]), 'B': m[0::2, 1::2]}


def colfix(y):
    return y - np.median(y - ndi.uniform_filter1d(y, COLFIX_W, axis=1), axis=0)[None, :]


def load_scene(folder):
    """Normalised 10-frame mean planes, per-channel gains and data weights, and the selected single frame (G)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f'scene_{folder.name}.npz'
    if f.exists():
        z = np.load(f)
        return {c: z[c] for c in 'RGB'}, json.loads(str(z['meta'])), z['single']
    pl = [planes(m) for m in mosaic_stack(folder)]
    mean = {c: colfix(np.mean([p[c] for p in pl], 0)) for c in 'RGB'}
    noise = {c: np.std([p[c] for p in pl], 0, ddof=1) / np.sqrt(len(pl)) for c in 'RGB'}
    s = np.percentile(mean['G'], 99.9)
    ys = {c: mean[c] / s for c in 'RGB'}
    gains = {c: float(ys[c].mean() / ys['G'].mean()) for c in 'RGB'}
    var = {c: float(np.median((noise[c] / s) ** 2)) for c in 'RGB'}
    meta = dict(gains=gains, weights={c: var['G'] / var[c] for c in 'RGB'})
    single = colfix(pl[L.select_frame(folder, 'scene')]['G']) / s
    np.savez(f, **ys, single=single, meta=json.dumps(meta))
    return ys, meta, single


def demosaic(m, ch):
    """Bilinear demosaic of one colour by normalised convolution."""
    if ch == 'G':
        return rawio.green_channel(m, GBRG)
    site = {'R': (1, 0), 'B': (0, 1)}[ch]
    mask = np.zeros_like(m)
    mask[site[0]::2, site[1]::2] = 1
    k = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], float)
    return ndi.convolve(m * mask, k, mode='mirror') / ndi.convolve(mask, k, mode='mirror')


def bin2(p):
    n = p.shape[0] // 2 * 2
    q = p[:n, :n].reshape(n // 2, 2, n // 2, 2).sum((1, 3))
    return q / q.sum()


def load_psfs(folder):
    """Half-res R, G, B PSFs, all cropped at the green PSF's centre (keeps the real shift between colours)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f'psf_{folder.name}.npz'
    if f.exists():
        z = np.load(f)
        return {c: z[c] for c in 'RGB'}
    m = mosaic_stack(folder).mean(0)
    _, info = L.extract_psf_lcd(demosaic(m, 'G'), 1024)
    psfs = {c: bin2(L.extract_psf_lcd(demosaic(m, c), 1024, centre=info['centre'])[0]) for c in 'RGB'}
    np.savez(f, **psfs)
    return psfs


def joint_bank(ys, psfs, gains, weights):
    """Joint RGB Wiener (reflect padding) for many K: returns K -> result."""
    pad = psfs['G'].shape[0] // 2
    Yp = {c: P.pad_for_fft(ys[c], pad) for c in 'RGB'}
    H = {c: P.psf_to_otf(psfs[c], Yp['G'].shape) for c in 'RGB'}
    num = sum(weights[c] * gains[c] * np.conj(H[c]) * np.fft.fft2(Yp[c]) for c in 'RGB')
    den = sum(weights[c] * gains[c] ** 2 * np.abs(H[c]) ** 2 for c in 'RGB')
    return lambda K: np.real(np.fft.ifft2(num / (den + K)))[pad:-pad, pad:-pad]


def run_method(meth, ys, meta, single, psfs, param):
    g, w = meta['gains'], meta['weights']
    if meth.startswith('M1'):
        return L.wiener_bank(single, psfs['G'])(param)
    if meth.startswith('M2'):
        return L.wiener_bank(ys['G'], psfs['G'])(param)
    if meth.startswith('M3'):
        return joint_bank(ys, psfs, g, w)(param)
    if meth.startswith('M4'):
        return MC.joint_wiener_masked(ys, psfs, g, w, param, iters=40)
    if meth.startswith('M5'):
        return MC.joint_tv(ys, psfs, g, w, param, cg_iters=12)
    raise ValueError(meth)


def fidelity(x, c, R):
    """Ring fidelity, spoke phase taken from the result itself."""
    return float(P.ring_fidelity(x, c, L.N_SPOKES, P.star_phase(x, c, L.N_SPOKES, 0.6 * R), 0.08 * R, 0.9 * R))


def refine(x, c, R, half=3.0):
    """Sub-pixel star centre (0.5 px grid) that maximises ring fidelity."""
    best, bc = -2, c
    for dy in np.arange(-half, half + 0.5, 0.5):
        for dx in np.arange(-half, half + 0.5, 0.5):
            cc = (c[0] + dy, c[1] + dx)
            f = P.ring_fidelity(x, cc, L.N_SPOKES, P.star_phase(x, cc, L.N_SPOKES, 0.6 * R), 0.15 * R, 0.85 * R, n_r=10)
            if f > best:
                best, bc = f, cc
    return bc


def star_crop(x, c, R, pad=1.05):
    h = int(pad * R)
    xp = np.pad(x, h, mode='constant', constant_values=np.median(x))
    cy, cx = int(round(c[0])), int(round(c[1]))
    return xp[cy:cy + 2 * h, cx:cx + 2 * h]
