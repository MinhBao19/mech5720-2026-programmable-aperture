"""Lab 3 (MECH5720) coded aperture imaging, Parts 0-8 and Bonus 2, for the paper or the LCD apertures.

Every capture folder holds 10 DNGs. Where the lab asks for one image, one frame is picked automatically
(select_frame) and the choice is saved to frame_selection.json. Dark frames (Part 0) and the Gold Standard (Part 8)
use all 10. Black level and fixed pattern noise are subtracted from the raw mosaic before debayering, and the
full-resolution green channel is used unless stated.

The LCD session has no dark frames or no-aperture star of its own, so it uses the paper session's.
LCD PSFs are extracted in a 1024 px window: the LCD pixel grid repeats the aperture's blur along a vertical chain
of diffraction orders, and light leaking through 'off' pixels forms a wide halo.

Run:  py lab3_paper_aperture.py           -> output_lab3_paper/
      py lab3_paper_aperture.py --lcd     -> output_lab3_lcd/
      add --fresh to ignore the cached frames and design fits
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from scipy.optimize import minimize
from scipy.signal import find_peaks

import rawio
import pipeline as P

# ============================================================================ CONFIG
HERE = Path(__file__).resolve().parent
LCD = '--lcd' in sys.argv
CAPTURES = Path(os.environ.get('LAB3_DATA', HERE.parent / 'New Captures'))
PAPER_ROOT = CAPTURES / 'PaperAperture' / 'PaperAperture'
LCD_ROOT = CAPTURES / 'LCDApertures' / 'LCDApertures'
ROOT = LCD_ROOT if LCD else PAPER_ROOT
DESIGN_DIR = HERE / 'designs'
OUT = HERE / ('output_lab3_lcd' if LCD else 'output_lab3_paper')
CACHE = OUT / 'cache'
EXP = 'LCD' if LCD else 'paper'

MASKS = ['Levin', 'Nayar', 'Circle']
DISTANCES = [1000, 1150, 1300]
D_FAR = 1300                                    # Part 3 star, Part 6 spectra and corners
BL = rawio.BLACK_LEVEL
SAT = rawio.WHITE_LEVEL
CORNERS = ['TopLeft', 'TopRight', 'BottomLeft', 'BottomRight']
SCENES = ['Pengmin', 'NielLeSeal']
# Part 8 Gold Standard and Bonus 2: (mask, scene, distance)
GOLD = ('Levin', 'Pengmin', 1000) if LCD else ('Levin', 'Pengmin', 1300)
BONUS2 = ('Circle', 'Pengmin', 1000) if LCD else ('Nayar', 'Pengmin', 1300)

# shared with the LCD session (same camera, same day)
DARK = PAPER_ROOT / 'Paper-Part0' / 'LensBlackLevel'
LENS_STAR = PAPER_ROOT / 'Paper-Part1' / 'NoApertureSiemensStarG1E64000D1000'

if not LCD:
    LENS_PSF = ROOT / 'Paper-Part1' / 'LensPSF' / 'LensPSF400'
    LENS_PSF_ALT = {e: ROOT / 'Paper-Part1' / 'LensPSF' / f'LCDPSFG1E{e}D1000' for e in (500, 1000, 2000)}
    LENS_LABEL = 'no aperture'

    def mask_dir(m):
        return ROOT / f'{m} - Paper'

    PSF = {m: {d: mask_dir(m) / 'Part2' / f'{m}PSFG1E500D{d}' for d in DISTANCES} for m in MASKS}
    CORNER = {
        # all corner PSFs were taken at 1300 mm ('D1000' in the TopLeft name is a labelling slip)
        'Levin': {'TopLeft': 'LevinTopLeftPSFG1E500D1000',
                  'TopRight': 'LevinTopRightPSFG1E500D1300',
                  'BottomLeft': 'LevinBottomLeftPSFG1E500D1300',
                  'BottomRight': 'LevinBottomRightPSFG1E500D1300-Attempt2'},
        'Nayar': {c: f'NayarPSF{c}G1E500D1300' for c in CORNERS},
        'Circle': {c: f'Circle{c}PSFG1E500D1300' for c in CORNERS},
    }
    CORNER = {m: {c: mask_dir(m) / 'Part2' / f for c, f in v.items()} for m, v in CORNER.items()}
    STAR = {m: mask_dir(m) / 'Part3' / f'{m}SiemensStarG1E64000D{D_FAR}' for m in MASKS}
    SCENE = {m: {s: {d: mask_dir(m) / 'Part4' / f'{m}{s}G1E64000D{d}' for d in DISTANCES} for s in SCENES}
             for m in MASKS}
    PSF_SIZE = 160                              # PSF window (full-res px)
    LENS_SIZE = 160
    SHOW_SIZE = 160                             # display window
    FFT_SIZE = 256
else:
    # Part 1: the LCD showing a fully open square
    LENS_PSF = ROOT / 'LCD-Part1' / 'LCDSquarePSFG1E60000D1000'
    LENS_PSF_ALT = {e: ROOT / 'LCD-Part1' / f'LCDSquarePSFG1E{e}D1000' for e in (20000, 30000, 40000)}
    LENS_LABEL = 'open LCD square'

    def mask_dir(m):
        return ROOT / f'LCD{m}'

    PSF = {m: {d: mask_dir(m) / f'LCD{m}PSFG1E60000D{d}' for d in DISTANCES} for m in MASKS}
    CORNER = {m: {c: mask_dir(m) / f'LCD{m}{c}PSFG1E60000D1300' for c in CORNERS} for m in MASKS}
    # folder names as captured: the Levin scenes say E60000 but were taken at 256 ms
    STAR = {'Levin': mask_dir('Levin') / 'LCDLevinSiemensStarG1E60000D1300',
            'Nayar': mask_dir('Nayar') / 'LCDNayarSiemensStarG1E256000D1300',
            'Circle': mask_dir('Circle') / 'LCDCircleSiemensStarPSFG1E256000D1300'}
    SCENE = {'Levin': {s: {d: mask_dir('Levin') / f'LCDLevin{s}G1E60000D{d}' for d in DISTANCES} for s in SCENES},
             'Nayar': {s: {d: mask_dir('Nayar') / f'LCDNayar{s}G1E256000D{d}' for d in DISTANCES} for s in SCENES},
             'Circle': {s: {d: mask_dir('Circle') / f'LCDCircle{s}PSFG1E256000D{d}' for d in DISTANCES}
                        for s in SCENES}}
    PSF_SIZE = 1024
    LENS_SIZE = 1024
    SHOW_SIZE = 256
    FFT_SIZE = 1024


DB_FLOOR = -80 if LCD else -60                  # lower limit of the dB colour scales
KS = np.geomspace(1e-6, 5e-1, 41)               # Part 8 K sweep
N_SPOKES = 36                                   # black/white cycles on the Siemens star
SCENE_CROP = {'Pengmin': (600, 250, 560), 'NielLeSeal': (600, 250, 560)}   # upright (row, col, size)

INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#898781', '#e1e0d9'
MCOL = {'Levin': '#2a78d6', 'Nayar': '#eb6834', 'Circle': '#1baf7a'}
DCOL = {1000: '#86b6ef', 1150: '#2a78d6', 1300: '#104281'}
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.titlecolor': INK,
                     'axes.grid': False, 'grid.color': GRID, 'grid.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 160,
                     'figure.facecolor': 'white'})

FRESH = '--fresh' in sys.argv
STAR_PICKS = HERE / 'star_centres.json'         # hand-picked star centres (pick_star_centres.py)
RESULTS = {}
FIGDATA = {}                                    # arrays for compare_lcd_paper.py


def rel(p):
    try:
        return str(Path(p).relative_to(CAPTURES))
    except ValueError:
        return str(p)


# ============================================================================ RAW I/O
def dng_files(folder):
    return sorted(Path(folder, 'dng').glob('*.dng'))


def raw_frame(f):
    """Raw mosaic with black level subtracted (float), CFA, and saturation mask."""
    m, cfa = rawio.read_bayer(f)
    return m - BL, cfa, m >= SAT


_FPN = None


def fpn():
    """Part 0: mean of the black-level-corrected dark frames (raw mosaic, not clipped)."""
    global _FPN
    if _FPN is None:
        f = CACHE / 'fpn.npy'
        if f.exists() and not FRESH:
            _FPN = np.load(f)
        else:
            _FPN = np.mean([raw_frame(p)[0] for p in dng_files(DARK)], axis=0)
            np.save(f, _FPN)
    return _FPN


def green(folder, idx):
    """FPN-corrected full-resolution green plane of one frame (cached), and its saturation mask."""
    folder = Path(folder)
    key = CACHE / f'g_{folder.name.replace(" ", "")}_{idx:02d}.npz'
    if key.exists() and not FRESH:
        z = np.load(key)
        return z['g'].astype(np.float64), z['sat']
    m, cfa, sat = raw_frame(dng_files(folder)[idx])
    g = rawio.green_channel(m - fpn(), cfa)
    satg = rawio.green_channel(sat.astype(float), cfa) > 0
    np.savez(key, g=g.astype(np.float32), sat=satg)
    return g, satg


def green_mean(folder):
    """Mean of all frames, each FPN corrected."""
    g = np.mean([green(folder, i)[0] for i in range(len(dng_files(folder)))], axis=0)
    return column_fix(g) if LCD else g


def planes(folder, idx):
    """Half-resolution R, G (mean of both green sites) and B planes, FPN corrected (Bonus 2)."""
    m, cfa, _ = raw_frame(dng_files(folder)[idx])
    m = m - fpn()
    sites = {}
    for k, c in enumerate(cfa):
        sites.setdefault(c, []).append(m[k // 2::2, k % 2::2])
    return {'R': sites[0][0], 'G': 0.5 * (sites[1][0] + sites[1][1]), 'B': sites[2][0]}


# ============================================================================ FRAME SELECTION
SELECTION = {}


def select_frame(folder, kind):
    """Pick the single frame to use from a 10-frame capture.
    psf:   among frames where the source is lit and in its usual place, prefer unsaturated ones, then the one
           most similar to the median of the lit frames.
    scene: the frame most similar to the median of all frames."""
    folder = Path(folder)
    if str(folder) in SELECTION:
        return SELECTION[str(folder)]['frame'] - 1
    gs, nsat, pv, loc = [], [], [], []
    for f in dng_files(folder):
        m, _ = rawio.read_bayer(f)
        g = m[0::2, 0::2]
        gs.append(g)
        nsat.append(int((m >= SAT).sum()))
        s = ndi.gaussian_filter(g, 3)
        pv.append(float(s.max() - np.median(g)))
        loc.append(np.unravel_index(np.argmax(s), s.shape))
    gs, nsat, pv, loc = np.stack(gs), np.array(nsat), np.array(pv), np.array(loc)
    n = len(gs)
    if kind == 'psf':
        lit = np.where(pv >= 0.3 * pv.max())[0]
        home = np.median(loc[lit], 0)
        lit = np.array([i for i in lit if np.hypot(*(loc[i] - home)) <= 25])
        med = np.median(gs[lit], 0)
        corr = np.array([np.corrcoef(g.ravel(), med.ravel())[0, 1] for g in gs])
        cand = [i for i in lit if nsat[i] == 0] or list(lit)
        best = max(cand, key=lambda i: corr[i])
        why = (f'{len(lit)}/{n} frames lit; {"no" if nsat[best] == 0 else nsat[best]} saturated px '
               f'(min over lit frames {nsat[lit].min()}); corr to median of lit frames {corr[best]:.3f}')
    else:
        med = np.median(gs, 0)
        corr = np.array([np.corrcoef(g.ravel(), med.ravel())[0, 1] for g in gs])
        best = int(np.argmax(corr))
        why = f'most typical of {n} frames: corr to median {corr[best]:.4f} (lowest frame {corr.min():.4f})'
    SELECTION[str(folder)] = dict(folder=rel(folder), frame=int(best) + 1,
                                  kind=kind, reason=why, saturated_px=int(nsat[best]))
    return int(best)


def best_green(folder, kind):
    g, sat = green(folder, select_frame(folder, kind))
    if LCD and kind == 'scene':
        g = column_fix(g)
    return g, sat


def column_fix(g):
    """LCD only: remove the residual column pattern that the 64 ms dark frames leave on 256 ms captures
    (high-pass across columns, median along each column)."""
    return g - np.median(g - ndi.uniform_filter1d(g, 31, axis=1), axis=0)[None, :]


# ============================================================================ PSF
def extract_psf(g, size=PSF_SIZE):
    """Window centred on the PSF centroid, local background subtracted, noise floor soft-thresholded,
    connected blob kept, unit sum. Returns (psf, info). Large LCD windows go to extract_psf_lcd."""
    if LCD and size >= 256:
        return extract_psf_lcd(g, size)
    s = ndi.gaussian_filter(g, 3)
    cy, cx = np.unravel_index(np.argmax(s), s.shape)
    far = np.ones_like(g, bool)
    far[max(cy - size, 0):cy + size, max(cx - size, 0):cx + size] = False
    bg = np.median(g[far])
    noise = 1.4826 * np.median(np.abs(g[far] - bg))
    h = size // 2
    for _ in range(3):
        pad = np.pad(g - bg, h + 2, mode='constant')
        win = pad[cy + 2:cy + 2 + size, cx + 2:cx + 2 + size]
        w = np.clip(ndi.gaussian_filter(win, 1.5) - 4 * noise, 0, None)
        yy, xx = np.indices(win.shape)
        dy = (w * yy).sum() / w.sum() - (size - 1) / 2
        dx = (w * xx).sum() / w.sum() - (size - 1) / 2
        cy, cx = int(round(cy + dy)), int(round(cx + dx))
    support = ndi.binary_dilation(ndi.gaussian_filter(win, 2) > 3 * noise, iterations=4)
    lab, _ = ndi.label(support)
    core = lab[h, h] if lab[h, h] else np.bincount(lab.ravel())[1:].argmax() + 1
    psf = np.clip(win - 2 * noise, 0, None) * (lab == core)
    raw = win.copy()
    psf /= psf.sum()
    return psf, dict(centre=(int(cy), int(cx)), background=float(bg), noise=float(noise),
                     peak=float(win.max()), snr=float(win.max() / noise), total_counts=float((win * (lab == core)).sum()),
                     raw=raw)


def extract_psf_lcd(g, size, snr_keep=8.0, halo_sigma=6.0, centre=None):
    """LCD PSF in a large window. Background = the darker of the outer 200 columns on each side; the window
    is centred on the bright orders (or on `centre`); parts off the sensor are filled by point symmetry; high-SNR
    pixels are kept as measured and the low-SNR halo is replaced by a smoothed version."""
    edges = [g[:, :200], g[:, -200:]]
    bg = min(np.median(e) for e in edges)
    e = edges[int(np.argmin([np.median(e) for e in edges]))]
    noise = 1.4826 * np.median(np.abs(e - np.median(e)))
    g = g - bg
    if centre is None:
        s = ndi.gaussian_filter(g, 3)
        cy, cx = np.unravel_index(np.argmax(s), s.shape)
        # centroid within +/-80 px: a wider window is pulled by the halo and a ghost ~500 px to one side
        for _ in range(3):
            y0, x0 = max(cy - 80, 0), max(cx - 80, 0)
            win = s[y0:cy + 81, x0:cx + 81]
            w = np.clip(win - 0.3 * win.max(), 0, None)
            yy, xx = np.indices(win.shape)
            cy, cx = int(round(y0 + (w * yy).sum() / w.sum())), int(round(x0 + (w * xx).sum() / w.sum()))
    else:
        cy, cx = centre
    h = size // 2
    crop = np.full((size, size), np.nan)
    y0, x0 = cy - h, cx - h
    ys = slice(max(y0, 0), min(y0 + size, g.shape[0]))
    xs = slice(max(x0, 0), min(x0 + size, g.shape[1]))
    crop[ys.start - y0:ys.stop - y0, xs.start - x0:xs.stop - x0] = g[ys, xs]
    missing = np.isnan(crop)
    crop = np.where(missing, np.roll(crop[::-1, ::-1], (1, 1), axis=(0, 1)), crop)
    still = np.isnan(crop)
    crop[still] = 0.0
    raw = crop.copy()
    keep = ndi.gaussian_filter(crop, 1.5) > snr_keep * noise
    psf = np.clip(np.where(keep, crop, ndi.gaussian_filter(crop, halo_sigma)), 0, None)
    psf *= np.outer(np.hanning(size), np.hanning(size)) ** 0.1
    total = psf.sum()
    psf /= total
    c = size // 2
    r = 0.5 * SHOW_SIZE
    inner = psf[int(c - r):int(c + r), int(c - r):int(c + r)].sum()
    return psf, dict(centre=(int(cy), int(cx)), background=float(bg), noise=float(noise), peak=float(raw.max()),
                     snr=float(raw.max() / noise), total_counts=float(total),
                     frac_filled_by_symmetry=float(missing.mean()), energy_in_show_window=float(inner), raw=raw)


def load_design(name):
    a = np.loadtxt(DESIGN_DIR / name, delimiter=',', usecols=range(13))
    return a.astype(float)


DESIGN = {'Levin': load_design('levin_13x13.txt'), 'Nayar': load_design('nayar.txt')}
OPEN_CELLS = {k: int(v.sum()) for k, v in DESIGN.items()}
CIRCLE_R_CELLS = np.sqrt(OPEN_CELLS['Nayar'] / np.pi)          # equal-area circle, in mask cells


def render_design(mask, size, s, theta, cy, cx, variant, ss=4):
    """Area-sampled binary design on a size x size grid: s px per cell, rotation theta (rad), centre
    offset (cy, cx), variant 0..7 = one of the 8 flips / 90-degree rotations."""
    o = (np.arange(size * ss) + 0.5) / ss - size / 2
    yy, xx = np.meshgrid(o - cy, o - cx, indexing='ij')
    c, sn = np.cos(theta), np.sin(theta)
    u, v = (c * yy - sn * xx) / s, (sn * yy + c * xx) / s
    if mask == 'Circle':
        val = (np.hypot(u, v) <= CIRCLE_R_CELLS).astype(float)
    else:
        m = DESIGN[mask]
        m = np.rot90(m, variant % 4)
        if variant >= 4:
            m = m[:, ::-1]
        n = m.shape[0]
        iu, iv = np.floor(u + n / 2).astype(int), np.floor(v + n / 2).astype(int)
        ok = (iu >= 0) & (iu < n) & (iv >= 0) & (iv < n)
        val = np.zeros_like(u)
        val[ok] = m[iu[ok], iv[ok]]
    return val.reshape(size, ss, size, ss).mean((1, 3))


FIT_SIZE = 320 if LCD else 96                   # central crop used for the design fit


def comb_basis(d_img, spacing, size):
    """LCD model basis: the design at each diffraction order, plus a constant and a smooth halo term."""
    ks = np.arange(-int(size / 2 / spacing) - 1, int(size / 2 / spacing) + 2)
    cols = [ndi.shift(d_img, (k * spacing, 0), order=1, mode='constant').ravel() for k in ks]
    yy, xx = np.indices(d_img.shape) - (size - 1) / 2
    cols += [np.ones(d_img.size), np.exp(-(yy ** 2 + xx ** 2) / (2 * (size / 3) ** 2)).ravel()]
    return np.stack(cols, 1), ks


def comb_fit(d_img, target, spacing, size):
    """Least-squares order weights for the comb model; returns (model, order weights, ks)."""
    A, ks = comb_basis(d_img, spacing, size)
    coef = np.linalg.lstsq(A, target, rcond=None)[0]
    return A @ coef, coef[:len(ks)], ks


def fit_design(psf, mask, spacing=None):
    """Fit scale, rotation, offset, orientation and a Gaussian blur of the design to the PSF by maximising
    their correlation: coarse grid, then Powell refinement of the two best orientations. LCD (spacing given): the
    model is the design repeated along the diffraction orders with least-squares weights."""
    o = (psf.shape[0] - FIT_SIZE) // 2
    psf = psf[o:o + FIT_SIZE, o:o + FIT_SIZE]
    size = FIT_SIZE
    sm = ndi.gaussian_filter(psf, 1.5)
    support = sm > 0.06 * sm.max()
    # scale guess from the PSF's extent (LCD: its width across the order chain)
    extent_cells = 2 * CIRCLE_R_CELLS if mask == 'Circle' else 11.0
    if spacing:
        c = size // 2
        prof = sm[int(c - 2 * spacing):int(c + 2 * spacing)].mean(0)
        prof = prof - np.percentile(prof, 5)
        s0 = max((prof > 0.2 * prof.max()).sum() / extent_cells, 0.4)
        support = sm > 0.2 * sm.max()
    else:
        s0 = max(2 * np.sqrt(support.sum() / np.pi) / extent_cells, 0.4)
    yy, xx = np.indices(psf.shape)
    w = psf / psf.sum()
    c0 = ((w * yy).sum() - (size - 1) / 2, (w * xx).sum() - (size - 1) / 2)
    if spacing:
        c0 = (0.0, c0[1])
    # the PSF is clipped at a high percentile so its sharp core does not dominate; blur is capped at 3 px
    q = np.percentile(psf[support], 99.5 if spacing else 85)
    target = np.clip(psf, 0, q).ravel()

    def cost(p, variant, ss=3):
        s, th, cy, cx, ls = p
        d = render_design(mask, size, s, th, cy, cx, variant, ss=ss)
        d = ndi.gaussian_filter(d, np.exp(ls))
        if d.std() == 0:
            return 1.0
        model = comb_fit(d, target, spacing, size)[0] if spacing else d.ravel()
        return -np.corrcoef(model, target)[0, 1]

    variants = [0] if mask == 'Circle' else range(8)
    thetas = [0.0] if mask == 'Circle' else np.radians([-8, -4, 0, 4, 8])
    th_bound = (0, 0) if mask == 'Circle' else (-0.4, 0.4)
    coarse = []
    for v in variants:
        for th in thetas:
            for sc in ((0.75, 0.9, 1.0, 1.1, 1.25) if spacing else (0.6, 0.75, 0.9, 1.05, 1.2, 1.4, 1.7)):
                for ls in (np.log(0.7), np.log(1.5)):
                    p = [s0 * sc, th, c0[0], c0[1], ls]
                    coarse.append((cost(p, v, ss=2), v, p))
    coarse.sort(key=lambda t: t[0])
    best = None
    seen = set()
    for _, v, p in coarse:
        if v in seen:
            continue
        seen.add(v)
        r = minimize(cost, p, args=(v,), method='Powell',
                     bounds=[(0.7 * s0, 1.3 * s0) if spacing else (0.5 * s0, 1.8 * s0), th_bound, (-15, 15), (-15, 15), (np.log(0.3), np.log(3))],
                     options=dict(maxiter=1500, xtol=1e-3, ftol=1e-5))
        if best is None or r.fun < best[0]:
            best = (r.fun, v, r.x)
        if len(seen) == 2:
            break
    fun, v, (s, th, cy, cx, ls) = best
    out = dict(scale_px_per_cell=float(s), theta_deg=float(np.degrees(th)), offset=(float(cy), float(cx)),
               variant=int(v), blur_sigma_px=float(np.exp(ls)), corr=float(-fun))
    if spacing:
        d = ndi.gaussian_filter(render_design(mask, size, s, th, cy, cx, v), np.exp(ls))
        _, wts, ks = comb_fit(d, target, spacing, size)
        out.update(order_spacing_px=float(spacing), order_weights=(wts / wts[ks == 0][0]).tolist(),
                   orders=ks.tolist())
    return out


def design_psf(mask, fit, size=PSF_SIZE, centred=True):
    """Scaled design without blur, unit sum."""
    cy, cx = (0.0, 0.0) if centred else fit['offset']
    d = render_design(mask, size, fit['scale_px_per_cell'], np.radians(fit['theta_deg']), cy, cx, fit['variant'],
                      ss=2 if size > 512 else 4)
    return d / d.sum()


def design_with_orders(mask, fit, size):
    """LCD: blurred design repeated along the fitted diffraction orders."""
    d = ndi.gaussian_filter(design_psf(mask, fit, size, centred=False), fit['blur_sigma_px'])
    out = sum(w * ndi.shift(d, (k * fit['order_spacing_px'], 0), order=1)
              for w, k in zip(fit['order_weights'], fit['orders']))
    return np.clip(out, 0, None)


def _fit_job(args):
    key, mask, psf, spacing = args
    return key, fit_design(psf, mask, spacing)


# ============================================================================ HELPERS
def up(a):
    """Upright view (the camera is mounted rotated 90 degrees)."""
    return np.rot90(a, k=-1)


def stretch(a, lo=0.5, hi=99.5):
    l, h = np.percentile(a, [lo, hi])
    return np.clip((a - l) / (h - l + 1e-12), 0, 1)


def imshow(ax, a, title=None, cmap='gray', **kw):
    ax.imshow(a, cmap=cmap, **kw)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=8.5, color=INK)


def save(fig, name):
    p = OUT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  wrote', p.relative_to(HERE))


def spectrum_db(psf, n=FFT_SIZE):
    """|FFT| of the PSF on an n x n grid, dB re DC, centred."""
    H = np.abs(np.fft.fftshift(P.psf_to_otf(psf, (n, n))))
    return 20 * np.log10(np.maximum(H / H.max(), 1e-6))


def radial_mean(a, n=FFT_SIZE):
    yy, xx = np.indices(a.shape) - n // 2
    rr = np.hypot(yy, xx).round().astype(int)
    prof = np.bincount(rr.ravel(), a.ravel()) / np.bincount(rr.ravel())
    f = np.arange(len(prof)) / n
    m = f <= 0.5
    return f[m], prof[m]


def otf_radial_lin(psf, n=FFT_SIZE):
    H = np.abs(np.fft.fftshift(P.psf_to_otf(psf, (n, n))))
    return radial_mean(H / H.max(), n)


def norm_img(g):
    return g / np.percentile(g, 99.9)


# ============================================================================ PART 0
def part0():
    print('Part 0: fixed pattern noise')
    F = fpn()
    n = len(dng_files(DARK))
    y0, x0 = 508, 928
    crop = F[y0:y0 + 64, x0:x0 + 64]
    col = F.mean(0)
    RESULTS['part0'] = dict(n_dark_frames=n, exposure_us=64000, mean_counts=float(F.mean()),
                            std_counts=float(F.std()), min=float(F.min()), max=float(F.max()),
                            column_profile_std=float(col.std()), row_profile_std=float(F.mean(1).std()),
                            hot_pixels_gt_10=int((F > 10).sum()),
                            per_cfa_site_mean={f'{i // 2}{i % 2}': float(F[i // 2::2, i % 2::2].mean()) for i in range(4)},
                            crop_origin_rc=(y0, x0))
    fig, (ax, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 3.9), layout='constrained',
                                       gridspec_kw=dict(width_ratios=[2.6, 1.1, 1.2]))
    lo, hi = np.percentile(F, [0.5, 99.5])
    im = ax.imshow(F, cmap='gray', vmin=lo, vmax=hi, interpolation='nearest')
    ax.set_title(f'FPN, full frame: mean of {n} dark frames (64 ms), black level removed', fontsize=8.5)
    ax.add_patch(plt.Rectangle((x0, y0), 64, 64, fill=False, ec='#eb6834', lw=1.2))
    ax.set_xlabel('column (px)'); ax.set_ylabel('row (px)')
    fig.colorbar(im, ax=ax, shrink=0.8).set_label('counts (DN)')
    im2 = ax2.imshow(crop, cmap='gray', vmin=np.percentile(crop, 0.5), vmax=np.percentile(crop, 99.5),
                     interpolation='nearest', extent=(x0, x0 + 64, y0 + 64, y0))
    ax2.set_title('64 x 64 crop (orange box)', fontsize=8.5)
    ax2.set_xlabel('column (px)')
    fig.colorbar(im2, ax=ax2, shrink=0.8).set_label('counts (DN)')
    ax3.plot(col - col.mean(), color=MCOL['Levin'], lw=0.6)
    ax3.set_title('column means of the FPN (minus overall mean)', fontsize=8.5)
    ax3.set_xlabel('column (px)'); ax3.set_ylabel('DN'); ax3.grid(True, axis='y')
    save(fig, 'part0_fpn.png')


# ============================================================================ PART 1
def exposure_us(folder):
    return json.load(open(Path(folder) / 'camera_settings_paired_capture.json'))['controls']['ExposureTime']['current']


def centre_crop(a, n):
    o0, o1 = (a.shape[0] - n) // 2, (a.shape[1] - n) // 2
    return a[o0:o0 + n, o1:o1 + n] if o0 >= 0 else a


def part1():
    print(f'Part 1: lens PSF ({LENS_LABEL})')
    g, sat = best_green(LENS_PSF, 'psf')
    psf, info = extract_psf(g, LENS_SIZE)
    e0 = exposure_us(LENS_PSF)
    RESULTS['part1'] = dict(folder=rel(LENS_PSF), exposure_us=e0, what=LENS_LABEL,
                            saturated_green_px=int(sat.sum()), centre=info['centre'], snr=info['snr'],
                            fwhm_px=float(fwhm(psf)), note='Only captured at 1000 mm.',
                            energy_in_show_window=info.get('energy_in_show_window'))
    others = {}
    for e, f in LENS_PSF_ALT.items():
        ge, se = best_green(f, 'psf')
        others[e] = (extract_psf(ge, LENS_SIZE)[0], int(se.sum()))
    RESULTS['part1']['other_exposures_saturated_px'] = {e: v[1] for e, v in others.items()}

    fig, axs = plt.subplots(1, 4, figsize=(12, 3.2), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.2]))
    show = centre_crop(psf, SHOW_SIZE if LCD else LENS_SIZE)
    imshow(axs[0], up(show), f'{LENS_LABEL}, 1000 mm, {e0 / 1000:g} ms (linear)', vmin=0)
    big = centre_crop(psf, 512 if LCD else LENS_SIZE)
    imshow(axs[1], up(np.log10(np.maximum(big / big.max(), 1e-4))),
           f'log10, 4 decades ({big.shape[0]} px window)', vmin=-4, vmax=0)
    S = spectrum_db(psf)
    im = axs[2].imshow(S, cmap='magma', vmin=DB_FLOOR + 10, vmax=0, extent=(-.5, .5, .5, -.5))
    axs[2].set_title('|FFT| of PSF (dB)', fontsize=8.5)
    axs[2].set_xlabel('fx (cycles/px)'); axs[2].set_ylabel('fy (cycles/px)')
    fig.colorbar(im, ax=axs[2], fraction=0.046, pad=0.02).set_label('dB')
    f, r = radial_mean(S)
    axs[3].plot(f, r, color=INK, lw=2)
    for e, (p, ns), col in zip(others, others.values(), DCOL.values()):
        ff, rr = radial_mean(spectrum_db(p))
        axs[3].plot(ff, rr, lw=1, color=col, label=f'{e / 1000:g} ms ({ns} sat. px)')
    axs[3].plot([], [], color=INK, lw=2, label=f'{e0 / 1000:g} ms (used)')
    axs[3].legend(fontsize=7, frameon=False)
    axs[3].set_xlabel('spatial frequency (cycles/px)'); axs[3].set_ylabel('radial mean |H| (dB)')
    axs[3].set_title('Radial magnitude spectrum', fontsize=8.5); axs[3].grid(True)
    fig.tight_layout()
    save(fig, 'part1_lens_psf.png')


def fwhm(psf):
    p = psf / psf.max()
    return 2 * np.sqrt((p >= 0.5).sum() / np.pi)


# ============================================================================ PARTS 2 + 6
def load_mask_psfs():
    psfs, info = {}, {}
    for m in MASKS:
        for d in DISTANCES:
            g, sat = best_green(PSF[m][d], 'psf')
            psfs[(m, d)], info[(m, d)] = extract_psf(g)
            info[(m, d)]['saturated_px'] = int(sat.sum())
        for c in CORNERS:
            g, sat = best_green(CORNER[m][c], 'psf')
            psfs[(m, c)], info[(m, c)] = extract_psf(g)
            info[(m, c)]['saturated_px'] = int(sat.sum())
    return psfs, info


def lcd_order_spacing(psfs):
    """Diffraction-order spacing (px): median over the 1000 mm PSFs, where the orders are cleanly separated."""
    d = [P.order_spacing(psfs[(m, 1000)]) for m in MASKS]
    print('  LCD diffraction-order spacing (px):', [round(x, 1) for x in d])
    return float(np.nanmedian(d))


def design_fits(psfs, workers=14):
    f = CACHE / 'design_fits.json'
    if f.exists() and not FRESH:
        raw = json.load(open(f))
        return {tuple(json.loads(k)): v for k, v in raw.items()}
    from concurrent.futures import ProcessPoolExecutor
    spacing = lcd_order_spacing(psfs) if LCD else None
    jobs = [(k, k[0], p, spacing) for k, p in psfs.items()]
    with ProcessPoolExecutor(workers) as ex:
        fits = dict(ex.map(_fit_job, jobs))
    json.dump({json.dumps(list(k)): v for k, v in fits.items()}, open(f, 'w'), indent=1)
    return fits


def part6(psfs, info, fits):
    print('Part 6: PSF comparison')
    fig, axs = plt.subplots(3, 6, figsize=(12, 6.6))
    for i, m in enumerate(MASKS):
        for j, d in enumerate(DISTANCES):
            p = centre_crop(psfs[(m, d)], SHOW_SIZE)
            ft = fits[(m, d)]
            imshow(axs[i, 2 * j], up(p), f'{m} {d} mm: captured', vmin=0, vmax=p.max())
            dz = design_psf(m, ft, SHOW_SIZE)
            imshow(axs[i, 2 * j + 1], up(dz), f'design, {ft["scale_px_per_cell"]:.2f} px/cell', vmin=0, vmax=dz.max())
            if info[(m, d)]['saturated_px']:
                axs[i, 2 * j].text(3, SHOW_SIZE - 5, f'{info[(m, d)]["saturated_px"]} px clipped', color='#eda100',
                                   fontsize=6.5)
    extra = f' (central {SHOW_SIZE} px of the {PSF_SIZE} px PSF)' if LCD else ''
    fig.suptitle(f'{EXP}: captured PSF (green, unscaled){extra} vs scaled design', fontsize=10)
    fig.tight_layout()
    save(fig, 'part6_psf_vs_design.png')

    if LCD:
        fig, axs = plt.subplots(len(DISTANCES), 6, figsize=(12, 6.6))
        for j, m in enumerate(MASKS):
            for i, d in enumerate(DISTANCES):
                ft = fits[(m, d)]
                cap = centre_crop(psfs[(m, d)], FIT_SIZE)
                mod = design_with_orders(m, ft, FIT_SIZE)
                imshow(axs[i, 2 * j], up(cap), f'{m} {d} mm: captured', vmin=0, vmax=np.percentile(cap, 99.9))
                imshow(axs[i, 2 * j + 1], up(mod), f'design x orders (r={ft["corr"]:.2f})', vmin=0,
                       vmax=np.percentile(mod, 99.9))
        fig.suptitle(f'Central {FIT_SIZE} px: captured vs design repeated along the LCD diffraction orders '
                     f'(spacing {fits[(MASKS[0], D_FAR)]["order_spacing_px"]:.1f} px)', fontsize=10)
        fig.tight_layout()
        save(fig, 'part6_lcd_order_model.png')

    # --- 2D spectra at the furthest distance
    fig, axs = plt.subplots(2, 3, figsize=(11, 7))
    spectra = {}
    for j, m in enumerate(MASKS):
        cap = np.zeros((FFT_SIZE, FFT_SIZE)); o = (FFT_SIZE - PSF_SIZE) // 2
        cap[o:o + PSF_SIZE, o:o + PSF_SIZE] = psfs[(m, D_FAR)]
        des = design_psf(m, fits[(m, D_FAR)], FFT_SIZE)
        for i, (lab, p) in enumerate([('captured', cap), ('design', des)]):
            S = spectrum_db(up(p))
            spectra[(m, lab)] = S
            im = axs[i, j].imshow(S, cmap='magma', vmin=DB_FLOOR, vmax=0, extent=(-.5, .5, .5, -.5))
            axs[i, j].set_title(f'{m} {lab}, {D_FAR} mm', fontsize=9)
            axs[i, j].set_xlabel('fx (cycles/px)'); axs[i, j].set_ylabel('fy (cycles/px)')
            fig.colorbar(im, ax=axs[i, j], fraction=0.046, pad=0.02).set_label('|H| (dB re DC)')
    fig.tight_layout()
    save(fig, 'part6_fft2d.png')

    # --- 1D radial magnitude, solid = design, dashed = captured
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    radial = {}
    for m in MASKS:
        for lab, ls in [('design', '-'), ('captured', '--')]:
            H = 10 ** (spectra[(m, lab)] / 20)
            f, r = radial_mean(H)
            radial[(m, lab)] = (f, r)
            FIGDATA[f'radial_{m}_{lab}_f'], FIGDATA[f'radial_{m}_{lab}_H'] = f, r
            ax.plot(f, 20 * np.log10(r), ls=ls, color=MCOL[m], lw=2 if lab == 'design' else 1.6)
    ax.plot([], [], '-', color=INK2, label='design (solid)')
    ax.plot([], [], '--', color=INK2, label='captured (dashed)')
    for m in MASKS:
        ax.plot([], [], color=MCOL[m], lw=3, label=m)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.set_xlim(0, 0.5); ax.set_ylim(DB_FLOOR - 10, 2)
    ax.set_xlabel('spatial frequency (cycles/px)'); ax.set_ylabel('radially averaged |H| (dB re DC)')
    ax.set_title(f'Magnitude response at {D_FAR} mm', fontsize=10); ax.grid(True)
    fig.tight_layout()
    save(fig, 'part6_fft_radial.png')

    # --- corners
    cols = ['TopLeft', 'TopRight', 'centre', 'BottomLeft', 'BottomRight']
    fig, axs = plt.subplots(3, 5, figsize=(11, 7))
    corner_tab = {}
    for i, m in enumerate(MASKS):
        corner_tab[m] = {}
        for j, c in enumerate(cols):
            key = (m, D_FAR) if c == 'centre' else (m, c)
            p, ft, inf = centre_crop(psfs[key], SHOW_SIZE), fits[key], info[key]
            dist = D_FAR
            imshow(axs[i, j], up(p), f'{m} {c}', vmin=0, vmax=p.max())
            axs[i, j].text(3, SHOW_SIZE - 6, f'{ft["scale_px_per_cell"]:.2f} px/cell, blur {ft["blur_sigma_px"]:.1f} px, '
                                            f'r={ft["corr"]:.2f}', color='white', fontsize=6.5)
            corner_tab[m][c] = dict(distance_mm=dist, frame_position_rc=inf['centre'], **{k: ft[k] for k in
                                    ('scale_px_per_cell', 'theta_deg', 'blur_sigma_px', 'corr', 'variant')})
    fig.suptitle(f'{EXP}: PSF across the field of view at {D_FAR} mm (upright view)' +
                 ('' if LCD else '; bottom corners sit further into the frame than the top ones'), fontsize=10)
    fig.tight_layout()
    save(fig, 'part6_corners.png')

    # --- numbers
    px_open = {m: float((centre_crop(psfs[(m, D_FAR)], SHOW_SIZE) > 0.25 * psfs[(m, D_FAR)].max()).sum())
               for m in MASKS}
    RESULTS['part6'] = dict(
        design_fits={f'{k[0]} {k[1]}': v for k, v in fits.items()},
        open_area=dict(
            levin_open_cells=OPEN_CELLS['Levin'], nayar_open_cells=OPEN_CELLS['Nayar'], grid_cells=169,
            levin_open_fraction=OPEN_CELLS['Levin'] / 169, nayar_open_fraction=OPEN_CELLS['Nayar'] / 169,
            circle_equal_area_radius_cells=float(CIRCLE_R_CELLS),
            captured_psf_area_px_at_1300=px_open,
            captured_psf_area_in_cells_at_1300={m: px_open[m] / fits[(m, D_FAR)]['scale_px_per_cell'] ** 2 for m in MASKS}),
        psf_exposure_us=exposure_us(PSF[MASKS[0]][D_FAR]),
        psf_total_counts={f'{k[0]} {k[1]}': v['total_counts'] for k, v in info.items()},
        psf_energy_in_central_window={f'{k[0]} {k[1]}': v.get('energy_in_show_window') for k, v in info.items()},
        psf_saturated_px={f'{k[0]} {k[1]}': v['saturated_px'] for k, v in info.items()},
        min_db_radial_captured={m: float(20 * np.log10(radial[(m, 'captured')][1].min())) for m in MASKS},
        min_db_radial_design={m: float(20 * np.log10(radial[(m, 'design')][1].min())) for m in MASKS},
        corners=corner_tab)
    return radial


# ============================================================================ PART 7
def refine_centre_fidelity(x, c, R, half=4.0, step=0.5):
    """Sub-pixel star centre that maximises ring fidelity."""
    best, bc = -2, c
    for dy in np.arange(-half, half + step, step):
        for dx in np.arange(-half, half + step, step):
            cc = (c[0] + dy, c[1] + dx)
            phi = P.star_phase(x, cc, N_SPOKES, 0.6 * R)
            f = P.ring_fidelity(x, cc, N_SPOKES, phi, 0.15 * R, 0.85 * R, n_r=12)
            if f > best:
                best, bc = f, cc
    return (float(bc[0]), float(bc[1]))


def star_key(folder):
    """Key of a capture folder in STAR_PICKS (path relative to CAPTURES)."""
    return Path(folder).resolve().relative_to(CAPTURES.resolve()).as_posix()


def star_pick(folder):
    """Hand-picked {'centre_rc': [r, c], 'R_px': R or None} for this capture, or None."""
    if not STAR_PICKS.exists():
        return None
    return json.load(open(STAR_PICKS)).get(star_key(folder))


def star_geometry(g, guess=None, R_expected=None, centre=None):
    """Star centre (N-th harmonic power search, sub-pixel) and outer radius. With `centre` given, only the
    radius is estimated."""
    a = norm_img(g)
    if centre is not None:
        return star_radius(a, centre, R_expected)
    if guess is None:
        def hp(c):
            p = 0
            for r in (170, 250):
                _, v = P.sample_circle(a, c, r, 1024)
                if np.isnan(v).any():
                    return 0
                p += np.abs(np.fft.rfft(v - v.mean())[N_SPOKES]) / (np.std(v) + 1e-9)
            return p
        grid = [(y, x) for y in range(260, a.shape[0] - 250, 25) for x in range(260, a.shape[1] - 250, 25)]
        guess = max(grid, key=hp)
    c = guess
    for rlist, search, step in [((150, 200, 250), 30, 3), ((120, 160, 200, 250), 4, 1)]:
        c = P.refine_centre(a, c, rlist, N_SPOKES, search=search, step=step)
    def power(cc):
        p = 0
        for r in (120, 170, 220):
            _, v = P.sample_circle(a, cc, r, 2048)
            v = np.where(np.isnan(v), np.nanmean(v), v)
            p += np.abs(np.fft.rfft(v - v.mean())[N_SPOKES]) / (np.std(v) + 1e-9)
        return p
    sub = []
    for ax in (0, 1):
        e = np.eye(2)[ax]
        pm, p0, pp = power(np.array(c) - 0.5 * e), power(c), power(np.array(c) + 0.5 * e)
        den = pm - 2 * p0 + pp
        sub.append(0.5 * 0.5 * (pm - pp) / den if den < 0 else 0.0)
    c = (c[0] + sub[0], c[1] + sub[1])
    return star_radius(a, c, R_expected)


def star_radius(a, c, R_expected=None):
    """Outer radius: the steepest drop of the N-th harmonic amplitude, within +/-25 % of R_expected if
    given. Returns (centre, R)."""
    if R_expected:
        rs = np.arange(0.75 * R_expected, 1.25 * R_expected, 2)
        amp = np.array([P.harmonic_contrast(a, c, r, N_SPOKES) for r in rs])
        amp = ndi.gaussian_filter1d(np.nan_to_num(amp, nan=0.0), 3)
        R = float(rs[np.argmin(np.gradient(amp))])
        return (float(c[0]), float(c[1])), R
    rs = np.arange(20, 600, 2)
    amp = np.array([P.harmonic_contrast(a, c, r, N_SPOKES) for r in rs])
    ok = ~np.isnan(amp)
    rs, amp = rs[ok], amp[ok]
    plateau = np.median(amp[(rs > 100) & (rs < rs.max() * 0.6)])
    R = float(rs[np.where(amp > 0.5 * plateau)[0].max()])
    return (float(c[0]), float(c[1])), R


def circle_contrast(a, centre, r, N=N_SPOKES, half_width=2):
    """Part 7 Michelson contrast on one circle (averaged over r +/- half_width, lightly smoothed along the
    arc; I_max / I_min from find_peaks). Returns C, theta, profile, peak idx, trough idx."""
    n_theta = max(int(2 * np.pi * r * 2), 16 * N)
    vs = [P.sample_circle(a, centre, rr, n_theta) for rr in np.linspace(r - half_width, r + half_width, 5)]
    th = vs[0][0]
    v = np.nanmean([x[1] for x in vs], axis=0)
    period = n_theta / N
    valid = ~np.isnan(v)
    vv = ndi.gaussian_filter1d(np.where(valid, v, np.nanmean(v)), period / 10, mode='wrap')
    pk, _ = find_peaks(vv, distance=max(int(0.6 * period), 1))
    tr, _ = find_peaks(-vv, distance=max(int(0.6 * period), 1))
    pk, tr = pk[valid[pk]], tr[valid[tr]]
    if len(pk) < 3 or len(tr) < 3:
        return np.nan, th, vv, pk, tr
    imax, imin = vv[pk].mean(), vv[tr].mean()
    return (imax - imin) / (imax + imin), th, vv, pk, tr


def mtf_from_star(g, centre, R, r_min=None, n=40):
    """f = N / (2 pi r) [cycles/px], Michelson contrast, N-th harmonic contrast, radii."""
    a = norm_img(g)
    r_min = r_min or N_SPOKES / (np.pi * 0.9)        # f = 0.45 cycles/px
    radii = np.geomspace(r_min, 0.92 * R, n)
    C = np.array([circle_contrast(a, centre, r)[0] for r in radii])
    Ch = np.array([P.harmonic_contrast(a, centre, r, N_SPOKES) for r in radii])
    return N_SPOKES / (2 * np.pi * radii), C, Ch, radii


def part7(psfs):
    print('Part 7: MTF from the Siemens star')
    stars, geom = {}, {}
    for m in MASKS:
        stars[m], _ = best_green(STAR[m], 'scene')
    lens, _ = best_green(LENS_STAR, 'scene')
    source = {}

    def picked(k, folder, img, R_expected=None):
        p = star_pick(folder)
        if p is None:
            return None
        source[k] = 'picked'
        c = tuple(p['centre_rc'])
        R = p.get('R_px') or star_geometry(img, centre=c, R_expected=R_expected)[1]
        print(f'  {k}: hand-picked centre ({c[0]:.1f}, {c[1]:.1f}), R={R:.0f} px'
              + ('' if p.get('R_px') else ' (radius estimated)'))
        return (float(c[0]), float(c[1])), float(R)

    geom['Lens'] = picked('Lens', LENS_STAR, lens) or star_geometry(lens)
    star_ok = {}
    if not LCD:
        for m in MASKS:
            geom[m] = picked(m, STAR[m], stars[m]) or star_geometry(stars[m])
            star_ok[m] = True
    else:
        # LCD spokes are too faint to find on the capture, so the star is located on the Wiener result
        R_exp = geom['Lens'][1] * 1000 / D_FAR
        for m in MASKS:
            a = norm_img(stars[m])
            x = wiener_bank(a, psfs[(m, D_FAR)])(1e-4)
            xs = x * np.percentile(stars[m], 99.9)
            geom[m] = picked(m, STAR[m], xs, R_exp)
            if geom[m]:
                c, R = geom[m]
                counts = [P.estimate_spoke_count(x, c, f * R) for f in (0.5, 0.7, 0.85)]
                star_ok[m] = True
                print(f'  {m}: spoke counts on the Wiener result at the picked centre {counts}')
                continue
            c, R = star_geometry(xs, R_expected=R_exp)
            c = refine_centre_fidelity(x, c, R)
            geom[m] = (c, R)
            counts = [P.estimate_spoke_count(x, c, f * R) for f in (0.5, 0.7, 0.85)]
            star_ok[m] = sum(n == N_SPOKES for n in counts) >= 2
            print(f'  {m}: star located on the Wiener result, spoke counts {counts} -> '
                  f'{"OK" if star_ok[m] else "NOT FOUND (no measurable spokes)"}')

    fig, axs = plt.subplots(1, 4, figsize=(12, 3.6))
    for ax, k in zip(axs, MASKS + ['Lens']):
        img = stars[k] if k != 'Lens' else lens
        (cy, cx), R = geom[k]
        h = int(R * 1.15)
        crop = np.pad(img, h, mode='constant', constant_values=np.median(img))[
            int(cy):int(cy) + 2 * h, int(cx):int(cx) + 2 * h]
        imshow(ax, up(stretch(crop)), f'{k if k != "Lens" else "No aperture (1000 mm)"}: centre ({cy:.1f}, {cx:.1f}), R={R:.0f} px')
        for rf, col in [(0.12, '#eb6834'), (0.5, '#eda100'), (0.9, '#1baf7a')]:
            ax.add_patch(plt.Circle((h, h), rf * R, fill=False, ec=col, lw=1))
        ax.add_patch(plt.Circle((h, h), R, fill=False, ec='white', lw=0.8, ls=':'))
    save(fig, 'part7_star_geometry.png')

    fig, axs = plt.subplots(3, 3, figsize=(12, 7.5), sharex=True)
    circ = {}
    for i, m in enumerate(MASKS):
        (cy, cx), R = geom[m]
        a = norm_img(stars[m])
        circ[m] = {}
        for j, (nm, rf) in enumerate([('centre', 0.12), ('middle', 0.5), ('edge', 0.9)]):
            r = rf * R
            C, th, v, pk, tr = circle_contrast(a, (cy, cx), r)
            ax = axs[i, j]
            ax.plot(np.degrees(th), v, color=MCOL[m], lw=0.9)
            ax.plot(np.degrees(th[pk]), v[pk], 'v', ms=3.5, color=INK, mec='none')
            ax.plot(np.degrees(th[tr]), v[tr], '^', ms=3.5, color=INK2, mec='none')
            ax.axhline(np.nanmean(v[pk]), color=INK, lw=0.6, ls=':')
            ax.axhline(np.nanmean(v[tr]), color=INK2, lw=0.6, ls=':')
            f = N_SPOKES / (2 * np.pi * r)
            ax.set_title(f'{m}, {nm} circle r={r:.0f} px (f={f:.3f} cyc/px): C={C:.3f}', fontsize=8.5)
            ax.grid(True, axis='y')
            circ[m][nm] = dict(r_px=float(r), f_cyc_px=float(f), contrast=float(C))
            if j == 0:
                ax.set_ylabel('normalised intensity')
            if i == 2:
                ax.set_xlabel('angle (deg)')
    axs[0, 0].plot([], [], 'v', color=INK, label='peaks (I_max)'); axs[0, 0].plot([], [], '^', color=INK2, label='troughs (I_min)')
    axs[0, 0].legend(fontsize=7, frameon=False, loc='lower right')
    fig.tight_layout()
    save(fig, 'part7_contrast_vs_angle.png')

    curves = {}
    for k in MASKS + ['Lens']:
        img = stars[k] if k != 'Lens' else lens
        curves[k] = mtf_from_star(img, *geom[k])
        FIGDATA[f'mtf_{k}_f'], FIGDATA[f'mtf_{k}_C'] = curves[k][0], curves[k][1]
    # normalised by the target's own contrast (in-focus, no-aperture star)
    CL = curves['Lens'][1]
    C_target = float(np.nanmax(CL))
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for k in MASKS + ['Lens']:
        f, C = curves[k][0], curves[k][1] / C_target
        col = MCOL.get(k, INK2)
        ax.plot(f, C, '-o', ms=3, lw=1.6, color=col, ls='-' if k != 'Lens' else ':')
    for k in MASKS:
        ax.plot([], [], color=MCOL[k], lw=2, label=f'{k}, {D_FAR} mm')
    ax.plot([], [], ':', color=INK2, lw=2,
            label='no aperture, 1000 mm (in focus' + (', paper session)' if LCD else ')'))
    ax.legend(frameon=False, fontsize=8)
    ax.set_xscale('log'); ax.set_xlim(0.004, 0.6); ax.set_ylim(-0.02, 1.1)
    ax.set_xlabel('spatial frequency f = N / 2πr (cycles/px)'); ax.set_ylabel('MTF (normalised contrast)')
    ax.set_title(f'MTF from the Siemens star (Michelson contrast / target contrast {C_target:.2f})', fontsize=10)
    ax.grid(True)
    fig.tight_layout()
    save(fig, 'part7_mtf.png')

    fig, axs = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, m in zip(axs, MASKS):
        f, C, Ch, _ = curves[m]
        fo, Ho = otf_radial_lin(psfs[(m, D_FAR)], 1024)
        ax.plot(fo, Ho, color=INK2, lw=1.4, label='|OTF| of captured PSF (radial mean)')
        ax.plot(f, C / C_target, 'o-', ms=3, color=MCOL[m], lw=1.4, label='star: Michelson contrast (lab)')
        ax.plot(f, Ch / np.nanmax(curves['Lens'][2]), 's--', ms=2.5, color=MCOL[m], lw=1,
                alpha=0.7, label='star: fundamental (N-th harmonic)')
        ax.set_xscale('log'); ax.set_xlim(0.004, 0.6); ax.set_ylim(-0.02, 1.1)
        ax.set_title(f'{m}, {D_FAR} mm', fontsize=9); ax.grid(True)
        ax.set_xlabel('spatial frequency (cycles/px)')
    axs[0].set_ylabel('modulation'); axs[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    save(fig, 'part7_mtf_vs_psf_otf.png')

    RESULTS['part7'] = dict(
        star_geometry={k: dict(centre_rc=v[0], R_px=v[1]) for k, v in geom.items()},
        target_contrast=C_target, three_circles=circ, star_located=star_ok,
        centre_source={k: source.get(k, "automatic") for k in geom},
        mtf_50_cyc_px={k: _crossing(curves[k][0], curves[k][1] / C_target, 0.5) for k in curves},
        mtf_10_cyc_px={k: _crossing(curves[k][0], curves[k][1] / C_target, 0.1) for k in curves})
    return stars, geom


def _crossing(f, C, level):
    o = np.argsort(f)
    f, C = f[o], C[o]
    below = np.where(C < level)[0]
    return float(f[below[0]]) if len(below) else None


# ============================================================================ PART 8
def wiener_bank(img, psf):
    """pipeline.wiener for many K: returns K -> result, with the forward FFTs done once."""
    margin = psf.shape[0] // 2
    p = P.pad_for_fft(img, margin)
    H = P.psf_to_otf(psf, p.shape)
    HY = np.conj(H) * np.fft.fft2(p)
    H2 = np.abs(H) ** 2
    return lambda K: np.real(np.fft.ifft2(HY / (H2 + K)))[margin:-margin, margin:-margin]


def star_fidelity(x, c, R, phi):
    """Ring fidelity of a result (LCD: spoke phase from the result, paper: from the capture)."""
    if LCD:
        phi = P.star_phase(x, c, N_SPOKES, 0.6 * R)
    return P.ring_fidelity(x, c, N_SPOKES, phi, 0.08 * R, 0.9 * R)


def star_crop(img, centre, R, pad=1.05):
    h = int(pad * R)
    cy, cx = int(round(centre[0])), int(round(centre[1]))
    return img[cy - h:cy + h, cx - h:cx + h]


def result_centre(x, c, R, span=30, step=3):
    """Star centre on a deconvolved result. LCD only: the PSF's centring shifts the result (up to ~20 px),
    so search +/-span px for the best ring fidelity. Paper results keep the capture's centre."""
    if not LCD:
        return c
    best, bc = -2, c
    for dy in range(-span, span + 1, step):
        for dx in range(-span, span + 1, step):
            cc = (c[0] + dy, c[1] + dx)
            f = P.ring_fidelity(x, cc, N_SPOKES, P.star_phase(x, cc, N_SPOKES, 0.6 * R), 0.15 * R, 0.85 * R, n_r=8)
            if f > best:
                best, bc = f, cc
    return refine_centre_fidelity(x, bc, R, half=2.0, step=0.5)


def part8(psfs, stars, geom):
    print('Part 8: deconvolution')
    sweep = {}
    best = {}
    recon = {}
    rc = {}
    for m in MASKS:
        a = norm_img(stars[m])
        c, R = geom[m]
        phi = P.star_phase(a, c, N_SPOKES, 0.6 * R)
        vl, fid = [], []
        wb = wiener_bank(a, psfs[(m, D_FAR)])
        rc[m] = cx = result_centre(wb(1e-5), c, R)
        for K in KS:
            x = wb(K)
            cr = star_crop(x, cx, R)
            vl.append(P.var_laplacian(cr))
            fid.append(star_fidelity(x, cx, R, phi))
        vl, fid = np.array(vl), np.array(fid)
        sweep[m] = (vl, fid)
        kv, kf = KS[np.argmax(vl)], KS[np.argmax(fid)]
        best[m] = dict(K_varlap=float(kv), K_fidelity=float(kf), varlap_captured=float(P.var_laplacian(star_crop(a, c, R))),
                       fidelity_captured=float(P.ring_fidelity(a, c, N_SPOKES, phi, 0.08 * R, 0.9 * R)),
                       varlap_max=float(vl.max()), fidelity_max=float(fid.max()))
        recon[m] = {K: wb(K) for K in (kv, kf)}
        FIGDATA[f'star_{m}_captured'] = up(star_crop(a, c, R)).astype(np.float32)
        FIGDATA[f'star_{m}_wiener'] = up(star_crop(recon[m][kf], cx, R)).astype(np.float32)
        FIGDATA[f'star_{m}_K'] = np.array(kf)

    # the 1300 mm star deconvolved with each distance's PSF
    xc = {}
    for m in MASKS:
        a = norm_img(stars[m])
        c, R = geom[m]
        phi = P.star_phase(a, c, N_SPOKES, 0.6 * R)
        for d in DISTANCES:
            wb = wiener_bank(a, psfs[(m, d)])
            cx = result_centre(wb(1e-5), c, R)
            fids = np.array([star_fidelity(wb(K), cx, R, phi) for K in KS[::2]])
            xc[(m, d)] = (float(fids.max()), float(KS[::2][fids.argmax()]))
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    M = np.array([[xc[(m, d)][0] for d in DISTANCES] for m in MASKS])
    im = ax.imshow(M, cmap='Blues', vmin=0, vmax=max(0.6, M.max()))
    for i, m in enumerate(MASKS):
        for j, d in enumerate(DISTANCES):
            ax.text(j, i, f'{M[i, j]:.2f}\nK={xc[(m, d)][1]:.0e}', ha='center', va='center', fontsize=8,
                    color='white' if M[i, j] > 0.35 else INK)
    ax.set_xticks(range(3), [f'PSF {d} mm' for d in DISTANCES]); ax.set_yticks(range(3), [f'{m} star' for m in MASKS])
    ax.set_title(f'Best fidelity of the {D_FAR} mm star vs PSF used', fontsize=9.5)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).set_label('ring fidelity')
    fig.tight_layout()
    save(fig, 'part8_psf_cross_check.png')

    # K for the natural scenes: the star's fidelity optimum. If the matched PSF never beats the capture, the K
    # of the best-matching PSF; if no PSF restores spokes (< 0.03), the geometric mean of the other masks' K.
    K_use = {}
    for m in MASKS:
        if best[m]['fidelity_max'] > max(best[m]['fidelity_captured'], 0.03):
            K_use[m] = best[m]['K_fidelity']
        else:
            dbest = max(DISTANCES, key=lambda d: xc[(m, d)][0])
            if xc[(m, dbest)][0] < 0.03:
                continue
            K_use[m] = xc[(m, dbest)][1]
            best[m]['note'] = (f'matched {D_FAR} mm PSF never beats the capture; K for natural scenes taken from '
                               f'the best-matching PSF ({dbest} mm)')
    for m in MASKS:
        if m not in K_use:
            K_use[m] = float(np.exp(np.mean(np.log(list(K_use.values())))))
            best[m]['note'] = ('no PSF restores measurable spokes on this star (fidelity < 0.03); K for natural '
                               'scenes = geometric mean of the other masks\' K')

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    for m in MASKS:
        vl, fid = sweep[m]
        axs[0].plot(KS, vl, color=MCOL[m], lw=1.8)
        i = np.argmax(vl); axs[0].plot(KS[i], vl[i], 'o', color=MCOL[m], ms=6, mec='white')
        axs[0].axhline(best[m]['varlap_captured'], color=MCOL[m], lw=0.8, ls=':')
        axs[1].plot(KS, fid, color=MCOL[m], lw=1.8)
        i = np.argmax(fid); axs[1].plot(KS[i], fid[i], 'o', color=MCOL[m], ms=6, mec='white')
        axs[1].text(KS[-1] * 1.15, fid[-1], m, color=INK2, va='center', fontsize=8)
        axs[1].axhline(best[m]['fidelity_captured'], color=MCOL[m], lw=0.8, ls=':')
    axs[0].set_xscale('log'); axs[0].set_yscale('log')
    axs[0].set_xlabel('Wiener K'); axs[0].set_ylabel('variance of Laplacian (star crop)')
    axs[0].set_title(f'Variance of the Laplacian vs K ({D_FAR} mm star; dotted = captured)', fontsize=9.5)
    axs[1].set_xscale('log'); axs[1].set_xlabel('Wiener K'); axs[1].set_ylabel('ring fidelity (corr. with ideal star)')
    axs[1].set_title('Cross-check: fidelity to the ideal star vs K (dotted = captured)', fontsize=9.5)
    for ax in axs:
        ax.grid(True); ax.set_xlim(KS[0] / 1.5, KS[-1] * 6)
    for m in MASKS:
        axs[0].plot([], [], color=MCOL[m], lw=2, label=m)
    axs[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, 'part8_K_sweep.png')

    fig, axs = plt.subplots(3, 3, figsize=(10, 10.4))
    for i, m in enumerate(MASKS):
        c, R = geom[m]
        a = norm_img(stars[m])
        kv, kf = best[m]['K_varlap'], best[m]['K_fidelity']
        for j, (img, t) in enumerate([(a, 'captured'), (recon[m][kv], f'Wiener, K={kv:.1e} (max var. Laplacian)'),
                                      (recon[m][kf], f'Wiener, K={kf:.1e} (max fidelity)')]):
            imshow(axs[i, j], up(stretch(star_crop(img, c if j == 0 else rc[m], R))), f'{m}: {t}')
    fig.tight_layout()
    save(fig, 'part8_star_deconvolved.png')

    nat = {}
    for s in SCENES:
        for m in MASKS:
            for d in DISTANCES:
                g, _ = best_green(SCENE[m][s][d], 'scene')
                a = norm_img(g)
                nat[(s, m, d)] = (a, P.wiener(a, psfs[(m, d)], K_use[m]))
                FIGDATA[f'nat_{s}_{m}_{d}_captured'] = up(a).astype(np.float32)
                FIGDATA[f'nat_{s}_{m}_{d}_wiener'] = up(nat[(s, m, d)][1]).astype(np.float32)
        for which in ('crop', 'full'):
            fig, axs = plt.subplots(len(MASKS) * 2, len(DISTANCES), figsize=(9, 3.2 * 2 * len(MASKS)) if which == 'crop'
                                    else (8, 4.6 * 2 * len(MASKS)))
            for i, m in enumerate(MASKS):
                for j, d in enumerate(DISTANCES):
                    a, x = nat[(s, m, d)]
                    for k, (img, t) in enumerate([(a, 'captured'), (x, f'Wiener K={K_use[m]:.0e}')]):
                        u = up(img)
                        if which == 'crop':
                            u = u[SCENE_CROP[s][0]:SCENE_CROP[s][0] + SCENE_CROP[s][2],
                                  SCENE_CROP[s][1]:SCENE_CROP[s][1] + SCENE_CROP[s][2]]
                        imshow(axs[2 * i + k, j], stretch(u, 0.2, 99.8), f'{m} {d} mm: {t}')
            fig.tight_layout()
            save(fig, f'part8_natural_{s}{"_fullframe" if which == "full" else ""}.png')

    gm, gs, gd = GOLD
    folder = SCENE[gm][gs][gd]
    single, _ = best_green(folder, 'scene')
    gold = green_mean(folder)
    sc = np.percentile(single, 99.9)
    single, gold = single / sc, gold / sc
    psf = psfs[(gm, gd)]
    Kg = K_use[gm]
    res = {'single': {'captured': single, 'standard': P.inverse_filter(single, psf),
                      'wiener': P.wiener(single, psf, Kg)},
           'gold': {'captured': gold, 'standard': P.inverse_filter(gold, psf), 'wiener': P.wiener(gold, psf, Kg)}}
    y0, x0, n = SCENE_CROP[gs]
    fig, axs = plt.subplots(2, 3, figsize=(10, 7))
    noise = {}
    for i, k in enumerate(['single', 'gold']):
        for j, t in enumerate(['captured', 'standard', 'wiener']):
            u = up(res[k][t])[y0:y0 + n, x0:x0 + n]
            ref = up(res[k]['captured'])[y0:y0 + n, x0:x0 + n]
            lo, hi = np.percentile(ref, [0.2, 99.8])
            imshow(axs[i, j], np.clip((u - lo) / (hi - lo), 0, 1),
                   f'{"Single frame" if k == "single" else "Gold Standard"}: '
                   f'{ {"captured": "captured", "standard": "inverse filter", "wiener": f"Wiener K={Kg:.0e}"}[t] }')
            noise[f'{k}_{t}_std'] = float(np.std(u))
    fig.suptitle(f'{gm} mask, {gs} at {gd} mm. Gold Standard = mean of 10 FPN-corrected frames. '
                 f'Each row uses its capture\'s grey range', fontsize=10)
    fig.tight_layout()
    save(fig, 'part8_gold_standard.png')
    stack = np.stack([green(folder, i)[0] / sc for i in range(len(dng_files(folder)))])
    RESULTS['part8'] = dict(psf_cross_check={f'{m} star / PSF {d}': dict(fidelity=v[0], K=v[1]) for (m, d), v in xc.items()},
                            K_sweep=dict(K=KS.tolist(), **{m: dict(varlap=sweep[m][0].tolist(), fidelity=sweep[m][1].tolist())
                                                          for m in MASKS}),
                            star_centre_on_result={m: [float(v) for v in rc[m]] for m in MASKS},
                            best=best, K_used_natural=K_use, gold=dict(mask=gm, scene=gs, distance=gd, **noise,
                            temporal_noise_single=float(stack.std(0).mean()),
                            temporal_noise_mean_expected=float(stack.std(0).mean() / np.sqrt(len(stack)))),
                            min_abs_otf={m: float(np.abs(P.psf_to_otf(psfs[(m, D_FAR)], (1080, 1920))).min()) for m in MASKS})
    return K_use


# ============================================================================ BONUS 2
def bonus2(K_use):
    print('Bonus 2: chromatic PSFs')
    m, s, d = BONUS2
    pf = PSF[m][d]
    pidx = select_frame(pf, 'psf')
    pp = planes(pf, pidx)
    psf = {c: extract_psf(pp[c], PSF_SIZE // 2)[0] for c in 'RGB'}
    sp = planes(SCENE[m][s][d], select_frame(SCENE[m][s][d], 'scene'))
    K = K_use[m]
    out = {}
    for mode in ('green PSF', 'per-channel PSF'):
        out[mode] = np.dstack([P.wiener(sp[c] / np.percentile(sp['G'], 99.9), psf['G' if mode == 'green PSF' else c], K)
                               for c in 'RGB'])
    cap = np.dstack([sp[c] / np.percentile(sp['G'], 99.9) for c in 'RGB'])
    wb = np.array([np.percentile(cap[..., i], 99) for i in range(3)])

    def rgb(x):
        x = x / wb
        return np.clip(up(x) / np.percentile(x, 99.7), 0, 1) ** (1 / 2.2)

    diff = np.abs(out['green PSF'] - out['per-channel PSF']).sum(2)
    dd = ndi.uniform_filter(up(diff), 120)
    edge = 150
    inner = dd[edge:-edge, edge:-edge]
    yy, xx = np.unravel_index(np.argmax(inner), inner.shape)
    yy = int(np.clip(yy + edge - 120, 0, dd.shape[0] - 240))
    xx = int(np.clip(xx + edge - 120, 0, dd.shape[1] - 240))
    sl = (slice(yy, yy + 240), slice(xx, xx + 240))
    fig, axs = plt.subplots(2, 4, figsize=(13, 7), gridspec_kw=dict(height_ratios=[1, 1]))
    for j, c in enumerate('RGB'):
        imshow(axs[0, j], up(psf[c]), f'{c} PSF ({m}, {d} mm, half-res plane)', vmin=0)
    for c, col in zip('RGB', ['#e34948', '#008300', '#2a78d6']):
        ff, rr = otf_radial_lin(psf[c], 512)
        axs[0, 3].plot(ff, 20 * np.log10(rr), color=col, lw=1.4, label=c)
    axs[0, 3].set_xlabel('cycles/px (half-res)'); axs[0, 3].set_ylabel('dB'); axs[0, 3].legend(frameon=False)
    axs[0, 3].set_title('radial |OTF| per channel', fontsize=8.5); axs[0, 3].grid(True); axs[0, 3].set_ylim(-60, 2)
    for j, (img, t) in enumerate([(cap, 'captured'), (out['green PSF'], 'Wiener, green PSF on R,G,B'),
                                  (out['per-channel PSF'], 'Wiener, R/G/B PSF per channel')]):
        imshow(axs[1, j], rgb(img)[sl], t)
    imshow(axs[1, 3], dd, 'where the two results differ most', cmap='magma')
    axs[1, 3].add_patch(plt.Rectangle((sl[1].start, sl[0].start), 240, 240, fill=False, ec='white'))
    fig.tight_layout()
    save(fig, 'bonus2_chromatic.png')
    cent = {c: tuple(np.round(np.array(ndi.center_of_mass(psf[c])) - (PSF_SIZE // 2 - 1) / 2, 2).tolist()) for c in 'RGB'}
    RESULTS['bonus2'] = dict(mask=m, scene=s, distance=d, K=K, psf_centroid_offset_halfres_px=cent,
                             psf_rms_radius_halfres_px={c: float(np.sqrt((psf[c] * ((np.indices(psf[c].shape) - np.array(
                                 ndi.center_of_mass(psf[c]))[:, None, None]) ** 2).sum(0)).sum())) for c in 'RGB'})
    if LCD:
        RESULTS['bonus2']['order_spacing_halfres_px'] = {c: P.order_spacing(psf[c], max_lag=120) for c in 'RGB'}


# ============================================================================ MAIN
def main():
    OUT.mkdir(exist_ok=True); CACHE.mkdir(exist_ok=True)
    sel = CACHE / 'frame_selection.json'
    if sel.exists() and not FRESH:
        SELECTION.update(json.load(open(sel)))
    part0()
    part1()
    psfs, info = load_mask_psfs()
    fits = design_fits(psfs)
    for (m, k), p in psfs.items():
        q = np.zeros((256, 256))
        if p.shape[0] >= 256:
            q[:] = centre_crop(p, 256)
        else:
            o = (256 - p.shape[0]) // 2
            q[o:o + p.shape[0], o:o + p.shape[0]] = p
        FIGDATA[f'psf_{m}_{k}'] = up(q).astype(np.float32)
        FIGDATA[f'scale_{m}_{k}'] = np.array(fits[(m, k)]['scale_px_per_cell'])
    part6(psfs, info, fits)
    stars, geom = part7(psfs)
    K_use = part8(psfs, stars, geom)
    bonus2(K_use)
    np.savez_compressed(OUT / 'figdata.npz', **FIGDATA)
    json.dump(SELECTION, open(sel, 'w'), indent=1)
    json.dump(sorted(SELECTION.values(), key=lambda r: r['folder']), open(OUT / 'frame_selection.json', 'w'), indent=1)
    json.dump(RESULTS, open(OUT / 'results.json', 'w'), indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    print('done ->', OUT)


if __name__ == '__main__':
    main()
