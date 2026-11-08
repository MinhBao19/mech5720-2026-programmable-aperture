"""Lab 3 Part 9: depth from defocus and an all-in-focus image, for the paper and the LCD Nayar apertures.

Data: New Captures/Part9_extract/Part9 (extracted from Part9.zip)
  PaperNayarDepthSceneG1E256000D800-1300   paper Nayar, objects at 800 and 1300 mm
  LCDNayarDepthSceneG1E256000D800-1300     LCD Nayar, same scene
The camera was refocused to ~650 mm and no PSFs were captured at that focus, so the PSF for each depth is
synthesised from the Part 2 Nayar PSF at 1300 mm: the defocus blur is the aperture's shadow scaled about its centre.
  paper: the whole PSF is scaled (energy preserving)
  LCD:   the PSF is split into blob x diffraction orders + halo (pipeline.decompose_lcd_psf), per colour; blob and
         halo are scaled, the order spacing is kept
Paper uses green Wiener; LCD uses M4 on half-resolution planes (lcd_methods.py), as chosen in output_best.

For each PSF scale the result is scored per patch with the normalised gradient sparsity L1/L2 (Krishnan et al.
2011): too small a PSF leaves blur, too large rings, and both raise L1/L2. The smoothed cost gives the depth map
(textureless patches marked unknown) and the weights of the all-in-focus blend. The scale is mapped to distance
with a thin-lens model fitted to the Part 2 PSFs and a Part 9 focus estimated from the two objects (paper only;
the LCD Nayar defocus fit is unreliable).

Run:  py run_part9_depth.py paper                -> output_part9_paper/
      py run_part9_depth.py lcd                  -> output_part9_lcd/
      py run_part9_depth.py lcd --m5             M5 (RGB TV + boundary, lambda 1e5) -> output_part9_lcd_m5/
      py run_part9_depth.py lcd --quarter        M4 on 2x2-binned planes            -> output_part9_lcd_quarter/
"""
import json
import sys
from pathlib import Path

MODE = 'lcd' if 'lcd' in sys.argv[1:] else 'paper'
if MODE == 'lcd' and '--lcd' not in sys.argv:
    sys.argv.append('--lcd')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

import pipeline as P
import lab3_paper_aperture as L
if MODE == 'lcd':
    import lcd_methods as N

HERE = Path(__file__).resolve().parent
R9 = L.CAPTURES / 'Part9_extract' / 'Part9'
VARIANT = 'm5' if '--m5' in sys.argv else 'quarter' if '--quarter' in sys.argv else ''
OUT = HERE / ('_'.join(filter(None, ['output_part9', MODE, VARIANT])))
SCENE = R9 / ('LCDNayarDepthSceneG1E256000D800-1300' if MODE == 'lcd' else 'PaperNayarDepthSceneG1E256000D800-1300')
MASK = 'Nayar'
MARKINGS = [700, 850, 1000, 1150, 1300]
# Nayar optimum on the 1300 mm star (paper: lab Part 8; LCD: M4, output_best/lcd_nayar)
K = {'paper': 1.4e-3, 'lcd': 1e-6}[MODE]
LCD_METH, LCD_PARAM = ('M5 RGB TV + boundary', 1e5) if VARIANT == 'm5' else ('M4 + boundary', K)
# relative scales of the base PSF
S = np.geomspace(0.4, 5.0, 16) if MODE == 'paper' else np.geomspace(0.2, 3.0, 16)
BASE_D = 1300
PX = 1 if MODE == 'paper' else 4 if VARIANT == 'quarter' else 2   # full-res px per image px
WIN = 151 // PX                                 # patch size of the depth cost
CANVAS = 768 if MODE == 'paper' else 1024
DIST = MODE == 'paper'

plt.rcParams['savefig.dpi'] = 130
up, stretch, imshow = L.up, L.stretch, L.imshow


# ----------------------------------------------------------------------------- PSFs
def zoom_c(img, s, size):
    """Scale an image about its centre by s onto a size x size canvas, energy preserving."""
    c0 = (img.shape[0] - 1) / 2; c = (size - 1) / 2
    yy, xx = np.indices((size, size), dtype=float)
    out = ndi.map_coordinates(img, [(yy - c) / s + c0, (xx - c) / s + c0], order=1, cval=0.0)
    return np.clip(out, 0, None)


def scaled_psf(base, s):
    """Paper: the base PSF scaled by s, unit sum."""
    p = zoom_c(base, s, CANVAS)
    return p / p.sum()


def lcd_decompose(psfs):
    """Per-colour decomposition of the half-res LCD PSFs. Each colour's order spacing is measured on its 1000 mm PSF,
    where the orders are cleanly separated."""
    ref = N.load_psfs(L.PSF[MASK][1000])
    out = {}
    for c, p in psfs.items():
        d = P.order_spacing(ref[c], max_lag=100)
        out[c] = P.decompose_lcd_psf(p, d, blob_half_w=80, halo_sigma=6)
    return out


def bin_img(a):
    """2x2 mean (quarter-resolution variant)."""
    h, w = a.shape[0] // 2 * 2, a.shape[1] // 2 * 2
    return a[:h, :w].reshape(h // 2, 2, w // 2, 2).mean((1, 3))


def lcd_scene():
    ys, meta, single = N.load_scene(SCENE)
    if VARIANT == 'quarter':
        ys, single = {c: bin_img(y) for c, y in ys.items()}, bin_img(single)
    return ys, meta, single


def lcd_job(args):
    """LCD reconstruction with the PSF at relative scale s (None = the PSF as given). Worker process."""
    s, psfs = args
    L.SELECTION.update(json.load(open(L.CACHE / 'frame_selection.json')))
    ys, meta, single = lcd_scene()
    if s is not None:
        psfs = {c: P.rebuild_lcd_psf(dec, scale=s) for c, dec in lcd_decompose(psfs).items()}
    if VARIANT == 'quarter':
        psfs = {c: N.bin2(p) for c, p in psfs.items()}
    return N.run_method(LCD_METH, ys, meta, single, psfs, LCD_PARAM)


def deconv_many(y, psf_sets, workers=8):
    """Reconstructions for a list of (scale or None, PSF): paper Wiener, LCD in parallel."""
    if MODE == 'paper':
        return [L.wiener_bank(y, p)(K) for _, p in psf_sets]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(lcd_job, psf_sets))


# ----------------------------------------------------------------------------- depth model
def defocus_k(fits):
    """Blur scale (px per mask cell) = k |1/F - 1/d|, fitted to the 1150 / 1300 mm design fits (1000 mm is near
    focus, where optics blur dominates). Returns k and the Part 2 focus."""
    s1, s2 = fits[(MASK, 1150)]['scale_px_per_cell'], fits[(MASK, 1300)]['scale_px_per_cell']
    k = (s2 - s1) / (1 / 1150 - 1 / 1300)
    inv_f2 = s2 / k + 1 / 1300
    return k, 1 / inv_f2


def scale_to_distance(s_rel, base_scale, k, F9):
    """Relative scale -> object distance behind the Part 9 focus F9 (thin lens)."""
    inv_d = 1 / F9 - s_rel * base_scale / k
    return np.where(inv_d > 0, 1 / np.maximum(inv_d, 1e-12), np.inf)


def distance_to_scale(d, base_scale, k, F9):
    return k * abs(1 / F9 - 1 / d) / base_scale


# ----------------------------------------------------------------------------- depth from defocus
def sparsity_cost(x, win=WIN):
    gy, gx = np.gradient(x)
    m = np.hypot(gx, gy)
    l1 = ndi.uniform_filter(m, win)
    l2 = np.sqrt(ndi.uniform_filter(m ** 2, win))
    return l1 / (l2 + 1e-9), l2


def main():
    OUT.mkdir(exist_ok=True)
    sel = L.CACHE / 'frame_selection.json'
    L.SELECTION.update(json.load(open(sel)))
    if MODE == 'paper':
        y = L.norm_img(L.best_green(SCENE, 'scene')[0])
    else:
        y = lcd_scene()[0]['G']
    fits = {tuple(json.loads(k)): v for k, v in json.load(open(L.CACHE / 'design_fits.json')).items()}
    meth = 'Wiener' if MODE == 'paper' else {'': 'M4 RGB Wiener + boundary, K=1e-6',
                                             'm5': 'M5 RGB TV + boundary, lambda=1e5',
                                             'quarter': 'M4 RGB Wiener + boundary, K=1e-6, quarter res'}[VARIANT]

    # the measured Part 2 PSFs (focus ~1 m, not the Part 9 focus)
    meas = {}
    for d in L.DISTANCES:
        if MODE == 'paper':
            gp, _ = L.best_green(L.PSF[MASK][d], 'psf')
            meas[d] = L.extract_psf(gp)[0]
        else:
            meas[d] = N.load_psfs(L.PSF[MASK][d])
    rec_meas = deconv_many(y, [(None, meas[d]) for d in L.DISTANCES])
    fig, axs = plt.subplots(1, 4, figsize=(13, 6))
    imshow(axs[0], stretch(up(y)), 'captured (green)')
    for ax, d, x in zip(axs[1:], L.DISTANCES, rec_meas):
        imshow(ax, stretch(up(x)), f'measured Part 2 PSF, {d} mm')
    fig.suptitle(f'{MODE}: Part 9 scene (objects at 800 and 1300 mm, focus ~650 mm) deconvolved with the '
                 f'PSFs measured in Part 2 (focus ~1 m), {meth}', fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / 'part9_measured_psfs.png', bbox_inches='tight'); plt.close(fig)

    base = meas[BASE_D]
    sets = [(s, scaled_psf(base, s)) for s in S] if MODE == 'paper' else [(s, base) for s in S]
    xs, cost, energy = {}, {}, {}
    for s, x in zip(S, deconv_many(y, sets)):
        xs[s] = x
        cost[s], energy[s] = sparsity_cost(x)
    C = np.stack([ndi.gaussian_filter(cost[s], 40 / PX) for s in S])
    best_i = np.argmin(C, 0)
    # textureless patches carry no depth information
    _, e0 = sparsity_cost(y)
    unknown = e0 < np.percentile(e0, 25)

    # Part 9 focus F9 from the two objects' best scales (boxes on the upright image)
    k, F2 = defocus_k(fits)
    base_scale = fits[(MASK, BASE_D)]['scale_px_per_cell']
    B = up(best_i)
    boxes = {800: (1100, 250), 1300: (400, 750)} if MODE == 'paper' else {800: (1250, 250), 1300: (450, 750)}
    boxes = {dd: (r0 // PX, c0 // PX) for dd, (r0, c0) in boxes.items()}
    h = 120 // PX
    obj_s = {}
    for dd, (r0, c0) in boxes.items():
        blk = B[r0 - h:r0 + h, c0 - h:c0 + h].ravel()
        obj_s[dd] = float(np.exp(np.median(np.log(S[blk]))))
    F9 = {dd: 1 / (obj_s[dd] * base_scale / k + 1 / dd) for dd in obj_s}
    F9_mean = float(np.mean(list(F9.values())))
    print(f'  Part 2 defocus model: k={k:.0f}, Part 2 focus {F2:.0f} mm; object scales {obj_s} -> '
          f'Part 9 focus estimates {({d: round(v) for d, v in F9.items()})} mm')

    pick = list(range(0, len(S), 2))
    fig, axs = plt.subplots(2, 4, figsize=(12, 11))
    for ax, i in zip(axs.ravel(), pick):
        s = S[i]
        dist = scale_to_distance(s, base_scale, k, F9_mean) if DIST else np.inf
        imshow(ax, stretch(up(xs[s])), f'PSF scale {s:.2f}x  (~{dist:.0f} mm)' if np.isfinite(dist)
               else f'PSF scale {s:.2f}x')
    fig.suptitle(f'{MODE} ({meth}): deconvolution with the synthesised PSF for each depth (Part 2 {BASE_D} mm PSF scaled' +
                 (f'; distance from thin-lens model, focus {F9_mean:.0f} mm)' if DIST else
                  '; no distance scale: the LCD defocus fit is unreliable)'), fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / 'part9_scale_sweep.png', bbox_inches='tight'); plt.close(fig)

    # one panel per distance marking (needs the distance model)
    mark_s = {}
    fig, axs = plt.subplots(1, len(MARKINGS) + 1, figsize=(3 * (len(MARKINGS) + 1), 6))
    imshow(axs[0], stretch(up(y)), 'captured')
    for ax, d in (zip(axs[1:], MARKINGS) if DIST else []):
        s = distance_to_scale(d, base_scale, k, F9_mean)
        mark_s[d] = float(s)
        imshow(ax, stretch(up(L.wiener_bank(y, scaled_psf(base, s))(K))), f'PSF for {d} mm\n(scale {s:.2f}x)')
    fig.suptitle(f'{MODE}: scene deconvolved with the PSF of each distance marking (synthesised, focus '
                 f'{F9_mean:.0f} mm). Near object: 800 mm, far object: 1300 mm', fontsize=10)
    if DIST:
        fig.tight_layout(); fig.savefig(OUT / 'part9_markings.png', bbox_inches='tight')
    plt.close(fig)

    # all-in-focus: per-pixel blend of the reconstructions, weights from the smoothed cost
    T = 0.02
    W = np.exp(-(C - C.min(0)) / T)
    W /= W.sum(0)
    aif = sum(W[i] * xs[s] for i, s in enumerate(S))
    depth = (scale_to_distance(S[best_i], base_scale, k, F9_mean) if DIST else S[best_i]).astype(float)
    depth[unknown] = np.nan
    fig, axs = plt.subplots(1, 4, figsize=(14, 6.4))
    imshow(axs[0], stretch(up(y)), 'captured')
    im = axs[1].imshow(up(np.clip(depth, 500, 2500) if DIST else depth), cmap='viridis_r')
    axs[1].set_xticks([]); axs[1].set_yticks([])
    axs[1].set_title(('estimated depth (mm)' if DIST else 'best PSF scale') + ', grey = no texture', fontsize=8.5)
    axs[1].set_facecolor('#e1e0d9')
    for dd, (r0, c0) in boxes.items():
        axs[1].add_patch(plt.Rectangle((c0 - h, r0 - h), 2 * h, 2 * h, fill=False, ec='white', lw=1))
        est = f'{scale_to_distance(obj_s[dd], base_scale, k, F9_mean):.0f} mm' if DIST else f'scale {obj_s[dd]:.2f}x'
        axs[1].text(c0, r0 + h - 8 // PX, f'{dd} mm object\n-> {est}',
                    color='white', ha='center', va='bottom', fontsize=7.5, fontweight='bold')
    fig.colorbar(im, ax=axs[1], fraction=0.046, pad=0.02).set_label('mm' if DIST else 'relative PSF scale')
    imshow(axs[2], stretch(up(aif)), 'all-in-focus (blend of the best scale per patch)')
    imshow(axs[3], stretch(up(xs[S[int(np.argmin(np.abs(S - obj_s[800])))]])), 'single PSF (near object\'s scale)')
    fig.suptitle(f'{MODE} ({meth}): depth from defocus with the coded aperture, and an all-in-focus image',
                 fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / 'part9_depth_allinfocus.png', bbox_inches='tight'); plt.close(fig)

    json.dump(dict(mode=MODE, variant=VARIANT, scene=SCENE.name, method=meth, K=K, base_psf_mm=BASE_D, scales=S.tolist(),
                   defocus_k=k, part2_focus_mm=F2, object_best_scale=obj_s,
                   part9_focus_estimates_mm=F9, part9_focus_mm=F9_mean, marking_scales=mark_s,
                   depth_estimate_of_objects_mm={dd: float(scale_to_distance(obj_s[dd], base_scale, k, F9_mean))
                                                 for dd in obj_s}),
              open(OUT / 'part9_results.json', 'w'), indent=1)
    print('done ->', OUT)


if __name__ == '__main__':
    main()
