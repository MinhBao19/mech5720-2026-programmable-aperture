"""Best reconstruction for each aperture x mask (paper / LCD x Levin / Nayar / Circle), with figures and tables.

Every candidate (method x PSF distance) is scored on the 1300 mm Siemens star (ring fidelity), the best is chosen
and applied to the star and to the natural scenes.

  paper  green, full resolution, Wiener. Candidates: single best frame or 10-frame mean, x each distance's PSF.
  LCD    half-resolution Bayer planes (lcd_methods.py). Candidates: M1-M4 x each distance's PSF, and M5 with the
         PSF that suits M4 best.

Selection rules:
  * Highest star fidelity wins, but M5 (TV) must beat M4 by more than TV_MARGIN: TV adds line artefacts on the
    natural scenes that the star score does not see.
  * Scenes use the PSF of their own distance, except at 1300 mm, where they use the PSF that won on the star.
    Exception: LCD Circle uses its 1300 mm PSF at every distance (SCENE_PSF_ALL).
  * If no candidate restores spokes (best fidelity < MIN_FID), the method chosen for the other masks of that
    aperture is used with the matched PSF and the geometric mean of their K.

Run:  py make_best_results.py                     (paper, LCD, then the tables)
      py make_best_results.py paper | lcd | report
Output: output_best/{paper,lcd}_{levin,nayar,circle}/, output_best/TABLES.md, summary.csv, candidates.csv
"""
import sys
import json
import subprocess
from pathlib import Path

MODE = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in ('paper', 'lcd', 'report') else 'all'
if MODE == 'lcd' and '--lcd' not in sys.argv:
    sys.argv.append('--lcd')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pipeline as P
import lab3_paper_aperture as L
if MODE == 'lcd':
    import lcd_methods as N

HERE = Path(__file__).resolve().parent
OUT = HERE / 'output_best'
CACHE = OUT / 'cache'
MASKS, DISTANCES, SCENES = L.MASKS, L.DISTANCES, L.SCENES
D_STAR = L.D_FAR
NSP = L.N_SPOKES
SCENE_PSF_ALL = {('LCD', 'Circle'): 1300}       # variations whose scenes all use one PSF
MIN_FID = 0.03                                  # below this no spokes are restored
TV_MARGIN = 0.02
INK, INK2, MUTED, GRID = L.INK, L.INK2, '#898781', L.GRID
MCOL = L.MCOL


def vdir(exp, m):
    return OUT / f'{exp.lower()}_{m.lower()}'


# ============================================================================ scoring helpers
def locate(x, c, R, span, step, refine):
    """Find the star again on a result (the PSF's centring shifts it)."""
    best, bc = -2, c
    for dy in np.arange(-span, span + step / 2, step):
        for dx in np.arange(-span, span + step / 2, step):
            cc = (c[0] + dy, c[1] + dx)
            f = P.ring_fidelity(x, cc, NSP, P.star_phase(x, cc, NSP, 0.6 * R), 0.15 * R, 0.85 * R, n_r=8)
            if f > best:
                best, bc = f, cc
    return refine(x, bc, R)


def mtf_curve(img, c, R, px_scale):
    """N-th harmonic contrast vs frequency; px_scale = full-res px per image px."""
    radii = np.geomspace(NSP / (np.pi * 0.9) / px_scale, 0.92 * R, 40)
    C = np.array([P.harmonic_contrast(img, c, r, NSP) for r in radii])
    return NSP / (2 * np.pi * radii * px_scale), C


def crossing(f, C, level):
    """Lowest frequency where the smoothed curve drops below level; nan if it never reaches it."""
    o = np.argsort(f)
    f, C = f[o], np.convolve(np.nan_to_num(C[o]), np.ones(3) / 3, mode='same')
    if not (C > level).any():
        return float('nan')
    i0 = int(np.argmax(C > level))
    below = np.where(C[i0:] < level)[0]
    if not len(below):
        return float(f[-1])
    j = i0 + below[0]
    return float(np.interp(level, [C[j], C[j - 1]], [f[j], f[j - 1]]))


def lens_reference():
    """Harmonic contrast of the in-focus no-aperture star: the target's own contrast."""
    geo = json.load(open(HERE / 'output_lab3_paper' / 'results.json'))['part7']['star_geometry']['Lens']
    g, _ = L.best_green(L.LENS_STAR, 'scene')
    f, C = mtf_curve(L.norm_img(g), tuple(geo['centre_rc']), geo['R_px'], 1)
    return f, C, float(np.nanmax(C))


def choose(cands):
    """Best candidate; M5 only if it beats the best M4 by more than TV_MARGIN."""
    best = max(cands, key=lambda c: c['fidelity'])
    if best['method'].startswith('M5'):
        m4 = [c for c in cands if c['method'].startswith('M4')]
        if m4:
            b4 = max(m4, key=lambda c: c['fidelity'])
            if best['fidelity'] - b4['fidelity'] <= TV_MARGIN:
                return b4, f'M5 TV scores {best["fidelity"]:.3f} vs M4 {b4["fidelity"]:.3f} (within {TV_MARGIN}); M4 kept (TV adds line artefacts on scenes)'
    return best, ''


# ============================================================================ paper
def paper_star(m):
    geo = json.load(open(L.OUT / 'results.json'))['part7']['star_geometry'][m]
    c, R = tuple(geo['centre_rc']), geo['R_px']
    imgs = {'1 frame': L.norm_img(L.best_green(L.STAR[m], 'scene')[0]),
            '10-frame mean': L.norm_img(L.green_mean(L.STAR[m]))}
    psfs = {d: L.extract_psf(L.best_green(L.PSF[m][d], 'psf')[0])[0] for d in DISTANCES}
    cap = imgs['1 frame']
    out = dict(centre=c, R=R, captured=float(P.ring_fidelity(cap, c, NSP, P.star_phase(cap, c, NSP, 0.6 * R),
                                                             0.08 * R, 0.9 * R)))
    cands, results = [], {}
    for fr, img in imgs.items():
        for d in DISTANCES:
            wb = L.wiener_bank(img, psfs[d])
            cc = L.refine_centre_fidelity(wb(3e-3), c, R)
            phi = P.star_phase(img, cc, NSP, 0.6 * R)
            fs = [float(P.ring_fidelity(wb(K), cc, NSP, phi, 0.08 * R, 0.9 * R)) for K in L.KS]
            i = int(np.argmax(fs))
            name = f'Wiener, {fr}, PSF {d} mm'
            cands.append(dict(name=name, method='Wiener', frames=fr, psf_d=d, param=float(L.KS[i]),
                              fidelity=fs[i], centre=cc, sweep=dict(K=L.KS.tolist(), fidelity=fs)))
            results[name] = wb(L.KS[i])
    out['candidates'] = cands
    return out, imgs, psfs, results


def run_paper():
    rows = {}
    fL, CL, Cmax = lens_reference()
    for m in MASKS:
        print(f'paper {m}: star candidates', flush=True)
        st, imgs, psfs, results = paper_star(m)
        ch, note = choose(st['candidates'])
        scene_psf = {d: (ch['psf_d'] if d == D_STAR else d) for d in DISTANCES}
        restored = results[ch['name']]
        f, Cc = mtf_curve(imgs['1 frame'], st['centre'], st['R'], 1)
        _, Cr = mtf_curve(restored, ch['centre'], st['R'], 1)
        scenes = {}
        for s in SCENES:
            for d in DISTANCES:
                fol = L.SCENE[m][s][d]
                y = L.norm_img(L.green_mean(fol) if ch['frames'] == '10-frame mean' else L.best_green(fol, 'scene')[0])
                scenes[(s, d)] = (y, L.wiener_bank(y, psfs[scene_psf[d]])(ch['param']))
        summ = summary('paper', m, st, ch, note, scene_psf, f, Cc, Cr, Cmax, res='full', px_scale=1)
        figures('paper', m, summ, imgs['1 frame'], st['centre'], restored, ch['centre'], st['R'], psfs, scenes,
                (f, Cc, Cr, fL, CL, Cmax), sweeps=[c for c in st['candidates'] if c['psf_d'] == ch['psf_d']])
        rows[m] = summ
        print(f'  -> {ch["name"]}  fidelity {ch["fidelity"]:.3f}', flush=True)
    return rows


# ============================================================================ LCD
def _init():
    """Worker processes: load the lab's frame selection."""
    if not L.SELECTION:
        L.SELECTION.update(json.load(open(L.CACHE / 'frame_selection.json')))


def lcd_star(m):
    """All LCD candidates on the 1300 mm star (half-res)."""
    _init()
    geo = json.load(open(L.OUT / 'results.json'))['part7']['star_geometry'][m]
    R = geo['R_px'] / 2
    c0 = ((geo['centre_rc'][0] - 0.5) / 2, (geo['centre_rc'][1] - 0.5) / 2)
    ys, meta, single = N.load_scene(L.STAR[m])
    psfs = {d: N.load_psfs(L.PSF[m][d]) for d in DISTANCES}
    out = dict(centre=c0, R=R, captured=N.fidelity(ys['G'], c0, R))
    cands, imgs = [], {}
    ref = lambda x, c, R: N.refine(x, c, R, half=2.0)
    for d in DISTANCES:
        banks = {'M1 G, 1 frame': L.wiener_bank(single, psfs[d]['G']),
                 'M2 G, 10-frame mean': L.wiener_bank(ys['G'], psfs[d]['G']),
                 'M3 RGB joint Wiener': N.joint_bank(ys, psfs[d], meta['gains'], meta['weights'])}
        cc = locate(banks['M3 RGB joint Wiener'](1e-4), c0, R, span=12, step=2, refine=ref)
        for meth, bank in banks.items():
            fs = [N.fidelity(bank(K), cc, R) for K in N.KS]
            i = int(np.argmax(fs))
            name = f'{meth}, PSF {d} mm'
            cands.append(dict(name=name, method=meth, psf_d=d, param=float(N.KS[i]), fidelity=fs[i], centre=cc,
                              sweep=dict(K=N.KS.tolist(), fidelity=fs)))
            imgs[name] = bank(N.KS[i])
        K3 = cands[-1]['param']
        best = None
        pts = []
        for K in (K3 / 3, K3, K3 * 3):
            x = N.run_method('M4 + boundary', ys, meta, single, psfs[d], K)
            f = N.fidelity(x, cc, R)
            pts.append((float(K), f))
            if best is None or f > best[0]:
                best = (f, K, x)
        name = f'M4 + boundary, PSF {d} mm'
        cands.append(dict(name=name, method='M4 + boundary', psf_d=d, param=float(best[1]), fidelity=best[0],
                          centre=cc, points=pts))
        imgs[name] = best[2]
    b4 = max((c for c in cands if c['method'] == 'M4 + boundary'), key=lambda c: c['fidelity'])
    d, cc = b4['psf_d'], b4['centre']
    best = None
    for lam in N.LAMS:
        x = N.run_method('M5 RGB TV + boundary', ys, meta, single, psfs[d], lam)
        f = N.fidelity(x, cc, R)
        if best is None or f > best[0]:
            best = (f, lam, x)
    name = f'M5 RGB TV + boundary, PSF {d} mm'
    cands.append(dict(name=name, method='M5 RGB TV + boundary', psf_d=d, param=float(best[1]), fidelity=best[0],
                      centre=cc))
    imgs[name] = best[2]
    for cnd in cands:
        cnd['centre'] = [float(v) for v in cnd['centre']]
    out['candidates'] = cands
    np.savez_compressed(CACHE / f'lcd_star_{m}.npz', captured=ys['G'].astype(np.float32),
                        **{k: v.astype(np.float32) for k, v in imgs.items()})
    return m, out


def lcd_scene(args):
    m, s, d, meth, psf_d, param = args
    _init()
    f = CACHE / f'lcd_scene_{m}_{s}_{d}_{meth[:2]}_{psf_d}_{param:.3g}.npz'
    if not f.exists():
        ys, meta, single = N.load_scene(L.SCENE[m][s][d])
        x = N.run_method(meth, ys, meta, single, N.load_psfs(L.PSF[m][psf_d]), param)
        np.savez_compressed(f, captured=ys['G'].astype(np.float32), restored=x.astype(np.float32))
    return (m, s, d), str(f)


def run_lcd(workers=9):
    from concurrent.futures import ProcessPoolExecutor
    L.SELECTION.update(json.load(open(L.CACHE / 'frame_selection.json')))
    sf = CACHE / 'lcd_star_scores.json'
    if sf.exists():
        stars = json.load(open(sf))
    else:
        with ProcessPoolExecutor(3) as ex:
            stars = dict(ex.map(lcd_star, MASKS))
        json.dump(stars, open(sf, 'w'), indent=1)
    chosen, notes = {}, {}
    for m in MASKS:
        chosen[m], notes[m] = choose(stars[m]['candidates'])
    ok = [m for m in MASKS if chosen[m]['fidelity'] >= MIN_FID]
    for m in MASKS:
        if m in ok:
            continue
        meths = [chosen[o]['method'] for o in ok]
        meth = max(set(meths), key=meths.count)
        K = float(np.exp(np.mean([np.log(chosen[o]['param']) for o in ok if chosen[o]['method'] == meth])))
        own = next(c for c in stars[m]['candidates'] if c['method'] == meth and c['psf_d'] == D_STAR)
        best = chosen[m]
        chosen[m] = dict(own, param=K, fidelity=own['fidelity'], fallback=True)
        notes[m] = (f'no candidate restores spokes (best {best["name"]}: {best["fidelity"]:.3f} < {MIN_FID}); '
                    f'uses {meth} with the matched PSF and the geometric mean of the other masks\' K ({K:.1e}). '
                    f'Its star fidelity at the sweep optimum is {own["fidelity"]:.3f}')
    jobs = []
    scene_psf = {}
    for m in MASKS:
        ch = chosen[m]
        scene_psf[m] = {d: SCENE_PSF_ALL.get(('LCD', m), ch['psf_d'] if d == D_STAR else d) for d in DISTANCES}
        jobs += [(m, s, d, ch['method'], scene_psf[m][d], ch['param']) for s in SCENES for d in DISTANCES]
    with ProcessPoolExecutor(workers) as ex:
        scene_files = dict(ex.map(lcd_scene, jobs))
    fL, CL, Cmax = lens_reference()
    rows = {}
    for m in MASKS:
        st, ch = stars[m], chosen[m]
        z = np.load(CACHE / f'lcd_star_{m}.npz')
        cap = z['captured']
        if ch.get('fallback'):
            ys, meta, single = N.load_scene(L.STAR[m])
            restored = N.run_method(ch['method'], ys, meta, single, N.load_psfs(L.PSF[m][ch['psf_d']]), ch['param'])
            ch['fidelity'] = N.fidelity(restored, tuple(ch['centre']), st['R'])
        else:
            restored = z[ch['name']]
        c, cc, R = tuple(st['centre']), tuple(ch['centre']), st['R']
        f, Cc = mtf_curve(cap, c, R, 2)
        _, Cr = mtf_curve(restored, cc, R, 2)
        scenes = {}
        for s in SCENES:
            for d in DISTANCES:
                zz = np.load(scene_files[(m, s, d)])
                scenes[(s, d)] = (zz['captured'], zz['restored'])
        psfs = {d: N.load_psfs(L.PSF[m][d])['G'] for d in DISTANCES}
        summ = summary('LCD', m, st, ch, notes[m], scene_psf[m], f, Cc, Cr, Cmax, res='half', px_scale=2)
        sweeps = [c_ for c_ in st['candidates'] if c_['psf_d'] == ch['psf_d']]
        figures('LCD', m, summ, cap, c, restored, cc, R, psfs, scenes, (f, Cc, Cr, fL, CL, Cmax), sweeps=sweeps)
        rows[m] = summ
        print(f'LCD {m} -> {ch["name"]}  fidelity {ch["fidelity"]:.3f}  {notes[m]}', flush=True)
    return rows


# ============================================================================ summary + figures
def summary(exp, m, st, ch, note, scene_psf, f, Cc, Cr, Cmax, res, px_scale):
    Cc, Cr = Cc / Cmax, Cr / Cmax
    s = dict(aperture=exp, mask=m, resolution=res,
             chosen=dict(name=ch['name'], method=ch['method'], frames=ch.get('frames'), psf_star_mm=ch['psf_d'],
                         param=ch['param'], param_name='lambda' if ch['method'].startswith('M5') else 'K',
                         fallback=bool(ch.get('fallback'))),
             note=note, scene_psf_mm={str(k): v for k, v in scene_psf.items()},
             star=dict(centre_rc_full=[float(st['centre'][0] * px_scale + (0.5 if px_scale == 2 else 0)),
                                       float(st['centre'][1] * px_scale + (0.5 if px_scale == 2 else 0))],
                       R_full=float(st['R'] * px_scale), fidelity_captured=float(st['captured']),
                       fidelity_restored=float(ch['fidelity'])),
             mtf=dict(captured_50=crossing(f, Cc, 0.5), captured_10=crossing(f, Cc, 0.1),
                      restored_50=crossing(f, Cr, 0.5), restored_10=crossing(f, Cr, 0.1),
                      f=f.tolist(), captured=Cc.tolist(), restored=Cr.tolist()),
             candidates=[{k: v for k, v in c.items() if k in ('name', 'method', 'frames', 'psf_d', 'param', 'fidelity')}
                         for c in st['candidates']])
    d = vdir(exp, m)
    d.mkdir(parents=True, exist_ok=True)
    json.dump(s, open(d / 'summary.json', 'w'), indent=1)
    return s


def show(ax, a, title, lo=0.5, hi=99.5):
    L.imshow(ax, L.stretch(a, lo, hi), title, vmin=0, vmax=1)


def figures(exp, m, s, cap, c, restored, cc, R, psfs, scenes, mtf, sweeps):
    d = vdir(exp, m)
    (d / 'images').mkdir(parents=True, exist_ok=True)
    ch = s['chosen']
    pn = 'λ' if ch['param_name'] == 'lambda' else 'K'
    tag = f'{exp} {m}'
    crop = L.star_crop if exp == 'paper' else N.star_crop
    col = MCOL[m]

    fig, axs = plt.subplots(1, 3, figsize=(9.6, 3.6))
    win = 160 if exp == 'paper' else 256
    for ax, dd in zip(axs, DISTANCES):
        p = L.centre_crop(psfs[dd], win) if psfs[dd].shape[0] > win else psfs[dd]
        L.imshow(ax, L.up(p / p.max()) ** 0.5, f'PSF {dd} mm', cmap='magma')
        if dd == ch['psf_star_mm']:
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_color(col); sp.set_linewidth(3)
            used = 'all scenes' if (exp, m) in SCENE_PSF_ALL else f'{D_STAR} mm scenes'
            ax.set_title(f'PSF {dd} mm (used for the star / {used})', fontsize=8.5, color=INK)
    fig.suptitle(f'{tag}: measured PSFs (upright, sqrt stretch; {"full-res" if exp == "paper" else "half-res green"} '
                 f'{win} px window)', fontsize=10)
    fig.tight_layout()
    fig.savefig(d / '01_psfs.png', bbox_inches='tight'); plt.close(fig)

    cands = sorted(s['candidates'], key=lambda q: q['fidelity'])
    fig, ax = plt.subplots(figsize=(7.5, 0.32 * len(cands) + 1.3))
    y = np.arange(len(cands))
    ax.barh(y, [q['fidelity'] for q in cands], height=0.62,
            color=[col if q['name'] == ch['name'] else '#c3c2bb' for q in cands])
    for yi, q in zip(y, cands):
        ax.text(max(q['fidelity'], 0) + 0.005, yi, f'{q["fidelity"]:.3f}', va='center', fontsize=7.5, color=INK2)
    ax.axvline(s['star']['fidelity_captured'], color=INK, lw=1, ls=':')
    ax.text(s['star']['fidelity_captured'], len(cands) - 0.4, ' captured', fontsize=7.5, color=INK, va='bottom')
    ax.set_yticks(y, [q['name'] for q in cands], fontsize=7.5)
    ax.set_xlabel(f'ring fidelity, {D_STAR} mm star (best parameter each)')
    ax.grid(True, axis='x')
    ax.set_title(f'{tag}: candidates (chosen: {ch["name"]}' + (', fallback' if ch['fallback'] else '') + ')', fontsize=9.5)
    fig.tight_layout()
    fig.savefig(d / '02_star_candidates.png', bbox_inches='tight'); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(9, 4.8))
    show(axs[0], L.up(crop(cap, c, R)), f'captured\nfidelity {s["star"]["fidelity_captured"]:.3f}')
    show(axs[1], L.up(crop(restored, cc, R)), f'restored: {ch["name"]}, {pn}={ch["param"]:.1e}\n'
                                               f'fidelity {s["star"]["fidelity_restored"]:.3f}', 1, 99)
    fig.suptitle(f'{tag}: {D_STAR} mm Siemens star', fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(d / '03_star.png', bbox_inches='tight'); plt.close(fig)
    plt.imsave(d / 'images' / 'star_captured.png', L.stretch(L.up(crop(cap, c, R))), cmap='gray')
    plt.imsave(d / 'images' / 'star_restored.png', L.stretch(L.up(crop(restored, cc, R)), 1, 99), cmap='gray')

    fig, ax = plt.subplots(figsize=(6.8, 4))
    shades = ['#9ec5f4', '#3987e5', '#104281']
    k = 0
    for c_ in sweeps:
        if 'sweep' in c_:
            ax.plot(c_['sweep']['K'], c_['sweep']['fidelity'], '-', lw=1.6, color=shades[k % 3],
                    label=c_['name'].split(', PSF')[0])
            k += 1
        elif 'points' in c_:
            pk, pf = zip(*c_['points'])
            ax.plot(pk, pf, 'o', ms=8, color=MCOL['Nayar'] if exp == 'LCD' else col, label=c_['name'].split(', PSF')[0])
    if ch['param_name'] == 'K':
        ax.plot([ch['param']], [s['star']['fidelity_restored']], marker='*', ms=15, color=col, mec=INK, mew=0.6,
                ls='none', label='chosen', zorder=5)
    ax.axhline(s['star']['fidelity_captured'], color=INK, lw=1, ls=':', label='captured')
    ax.set_xscale('log'); ax.grid(True)
    ax.set_xlabel('Wiener K'); ax.set_ylabel(f'ring fidelity, {D_STAR} mm star')
    ax.legend(frameon=False, fontsize=7.5)
    ax.set_title(f'{tag}: K sweep with the PSF {ch["psf_star_mm"]} mm', fontsize=9.5)
    fig.tight_layout()
    fig.savefig(d / '04_K_sweep.png', bbox_inches='tight'); plt.close(fig)

    f, Cc, Cr, fL, CL, Cmax = mtf
    fig, ax = plt.subplots(figsize=(6.8, 4))
    ax.plot(fL, CL / Cmax, ':', color=INK2, lw=1.6, label='no aperture, in focus (1000 mm)')
    ax.plot(f, Cc / Cmax, '-', color='#c3c2bb', lw=2, label='captured')
    ax.plot(f, Cr / Cmax, '-', color=col, lw=2, label=f'restored ({ch["method"]})')
    for lv in (0.5, 0.1):
        ax.axhline(lv, color=GRID, lw=0.8)
    ax.set_xscale('log'); ax.set_xlim(0.004, 0.6); ax.set_ylim(-0.02, 1.15); ax.grid(True)
    ax.set_xlabel('spatial frequency (cycles per full-res px)'); ax.set_ylabel('MTF (36th harmonic / target)')
    mt = s['mtf']
    ax.set_title(f'{tag}: MTF from the star. MTF50 {fmt(mt["captured_50"])} → {fmt(mt["restored_50"])}, '
                 f'MTF10 {fmt(mt["captured_10"])} → {fmt(mt["restored_10"])} cycles/px\n'
                 '(– = never reaches that level)', fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(d / '05_mtf.png', bbox_inches='tight'); plt.close(fig)

    for sc in SCENES:
        fig, axs = plt.subplots(len(DISTANCES), 2, figsize=(8.6, 6.2 * len(DISTANCES) * 0.95))
        for i, dd in enumerate(DISTANCES):
            y_, x_ = scenes[(sc, dd)]
            show(axs[i, 0], L.up(y_), f'{dd} mm: captured', 0.2, 99.8)
            show(axs[i, 1], L.up(x_), f'{dd} mm: restored ({ch["method"]}, {pn}={ch["param"]:.1e})', 0.2, 99.8)
            plt.imsave(d / 'images' / f'{sc}_{dd}mm_captured.png', L.stretch(L.up(y_), 0.2, 99.8), cmap='gray')
            plt.imsave(d / 'images' / f'{sc}_{dd}mm_restored.png', L.stretch(L.up(x_), 0.2, 99.8), cmap='gray')
        fig.suptitle(f'{tag}: {sc}', fontsize=10.5)
        fig.tight_layout(rect=(0, 0, 1, 0.985))
        fig.savefig(d / f'06_scene_{sc}.png', bbox_inches='tight'); plt.close(fig)
    print(f'  figures -> {d.relative_to(HERE)}', flush=True)


# ============================================================================ tables
def fmt(v, p=3):
    return '–' if v is None or (isinstance(v, float) and np.isnan(v)) else f'{v:.{p}f}'


def report():
    S = {(e, m): json.load(open(vdir(e, m) / 'summary.json')) for e in ('paper', 'LCD') for m in MASKS
         if (vdir(e, m) / 'summary.json').exists()}
    labs = {'paper': json.load(open(HERE / 'output_lab3_paper' / 'results.json')),
            'LCD': json.load(open(HERE / 'output_lab3_lcd' / 'results.json'))}
    energy = {}
    ef = HERE / 'output_lcd_design_vs_grid' / 'lcd.json'
    if ef.exists():
        energy = json.load(open(ef))['energy_budget']
    T = ['# Best-performing results, 5 Nov captures', '',
         'Made by `make_best_results.py`. One folder per variation, each with:',
         '- `01_psfs.png`: the PSFs, with the one used for the star outlined.',
         '- `02_star_candidates.png`: every candidate\'s score.',
         '- `03_star.png`: the captured and restored star.',
         '- `04_K_sweep.png`: the K sweep with the chosen PSF.',
         '- `05_mtf.png`: the MTF before and after.',
         '- `06_scene_*.png`: the natural scenes at each distance.',
         '- `images/`: every captured and restored image on its own, full frame.',
         '- `summary.json`: the numbers below.', '',
         'Scores are ring fidelity on the 1300 mm Siemens star: the correlation with an ideal 36-spoke star, '
         'from 0 (no spokes) to 1. Paper runs at full resolution on the green channel. LCD runs at half '
         'resolution on the Bayer planes. Frequencies are in cycles per full-resolution pixel.', '',
         '## 1. Chosen configuration', '',
         '| Variation | Method | Frames | PSF for star | PSFs for scenes (1000 / 1150 / 1300 mm) | Parameter | Note |',
         '|---|---|---|---|---|---|---|']
    csv1 = ['variation,method,frames,psf_star_mm,scene_psf_1000,scene_psf_1150,scene_psf_1300,param_name,param,'
            'fid_captured,fid_restored,mtf50_captured,mtf50_restored,mtf10_captured,mtf10_restored']
    for (e, m), s in S.items():
        ch = s['chosen']
        frames = ch['frames'] or ('10-frame mean' if not ch['method'].startswith('M1') else '1 frame')
        sp = ' / '.join(str(s['scene_psf_mm'][str(d)]) for d in DISTANCES)
        T.append(f'| {e} {m} | {ch["method"]} | {frames} | {ch["psf_star_mm"]} mm | {sp} | '
                 f'{ch["param_name"]} = {ch["param"]:.1e} | {s["note"] or "–"} |')
        mt = s['mtf']
        csv1.append(','.join(map(str, [f'{e} {m}', ch['method'], frames, ch['psf_star_mm'],
                                       *[s['scene_psf_mm'][str(d)] for d in DISTANCES], ch['param_name'], ch['param'],
                                       s['star']['fidelity_captured'], s['star']['fidelity_restored'],
                                       mt['captured_50'], mt['restored_50'], mt['captured_10'], mt['restored_10']])))
    T += ['', '## 2. Star results (1300 mm)', '',
          '| Variation | Fidelity: captured | Fidelity: restored | Gain | MTF50: captured → restored | MTF10: captured → restored |',
          '|---|---|---|---|---|---|']
    for (e, m), s in S.items():
        st, mt = s['star'], s['mtf']
        T.append(f'| {e} {m} | {fmt(st["fidelity_captured"])} | **{fmt(st["fidelity_restored"])}** | '
                 f'{fmt(st["fidelity_restored"] - st["fidelity_captured"], 2)} | '
                 f'{fmt(mt["captured_50"])} → {fmt(mt["restored_50"])} | {fmt(mt["captured_10"])} → {fmt(mt["restored_10"])} |')
    T += ['', 'MTF is the 36th angular harmonic of the star divided by that of the in-focus no-aperture star. MTF50 / '
          'MTF10 are the lowest frequency where it falls below 0.5 / 0.1. "–" means it never reaches that level, which '
          'for LCD captures means the contrast is at the noise floor at every frequency. A restored MTF can exceed the '
          'captured one at frequencies where the restoration also amplifies noise, so read it alongside fidelity.', '',
          '## 3. Every candidate (1300 mm star fidelity, best parameter each)', '']
    for e in ('paper', 'LCD'):
        names = []
        for m in MASKS:
            if (e, m) in S:
                for c in S[(e, m)]['candidates']:
                    if c['name'] not in names:
                        names.append(c['name'])
        T += [f'**{e}**', '', '| Candidate | ' + ' | '.join(MASKS) + ' |', '|---|' + '---|' * len(MASKS)]
        for n in names:
            cells = []
            for m in MASKS:
                c = next((c for c in S.get((e, m), {}).get('candidates', []) if c['name'] == n), None)
                v = fmt(c['fidelity']) if c else '–'
                if c and S[(e, m)]['chosen']['name'] == n:
                    v = f'**{v}**'
                cells.append(v)
            T.append(f'| {n} | ' + ' | '.join(cells) + ' |')
        T += ['', 'Captured: ' + ', '.join(f'{m} {fmt(S[(e, m)]["star"]["fidelity_captured"])}' for m in MASKS
                                         if (e, m) in S) + '. Bold = chosen.', '']
    T += ['## 4. PSFs', '', '| Variation | Design scale (px/cell) at 1000 / 1150 / 1300 mm | Design-fit correlation at '
          '1000 / 1150 / 1300 | Light in straight-through copy / other diffraction copies / halo (star PSF) |',
          '|---|---|---|---|']
    for (e, m), s in S.items():
        fits = labs[e]['part6']['design_fits']
        sc = ' / '.join(fmt(fits[f'{m} {d}']['scale_px_per_cell'], 2) for d in DISTANCES)
        co = ' / '.join(fmt(fits[f'{m} {d}']['corr'], 2) for d in DISTANCES)
        eb = energy.get(f'{m} {s["chosen"]["psf_star_mm"]}') if e == 'LCD' else None
        ebs = (f'{eb["zero_order_design"]:.0%} / {eb["other_orders"]:.0%} / {eb["halo"]:.0%}' if eb else '–')
        T.append(f'| {e} {m} | {sc} | {co} | {ebs} |')
    T += ['', 'LCD design scales come from the model with the diffraction copies (Part 6). Nayar 1150 / 1300 mm are '
          'unreliable there. The LCD light split is from `analyse_lcd_design_vs_grid.py`; the halo may include some '
          'room light (see the ghosting discussion).', '',
          '## 5. Raw star contrast on the three Part 7 circles (Michelson, captured)', '',
          '| Variation | Centre circle | Middle circle | Edge circle | Star centre (row, col), R (full-res px) | Centre from |',
          '|---|---|---|---|---|---|']
    for (e, m), s in S.items():
        p7 = labs[e]['part7']
        tc = p7['three_circles'][m]
        src = p7.get('centre_source', {}).get(m, 'automatic')
        st = s['star']
        T.append(f'| {e} {m} | {fmt(tc["centre"]["contrast"])} | {fmt(tc["middle"]["contrast"])} | '
                 f'{fmt(tc["edge"]["contrast"])} | ({st["centre_rc_full"][0]:.1f}, {st["centre_rc_full"][1]:.1f}), '
                 f'{st["R_full"]:.0f} | {src} |')
    (OUT / 'TABLES.md').write_text('\n'.join(T) + '\n', encoding='utf-8')
    (OUT / 'summary.csv').write_text('\n'.join(csv1) + '\n', encoding='utf-8')
    cand_csv = ['variation,candidate,method,psf_d,param,fidelity,chosen']
    for (e, m), s in S.items():
        for c in s['candidates']:
            cand_csv.append(','.join(map(str, [f'{e} {m}', f'"{c["name"]}"', c['method'], c['psf_d'], c['param'],
                                               c['fidelity'], c['name'] == s['chosen']['name']])))
    (OUT / 'candidates.csv').write_text('\n'.join(cand_csv) + '\n', encoding='utf-8')
    print('tables ->', OUT / 'TABLES.md')


# ============================================================================ main
def main():
    OUT.mkdir(exist_ok=True); CACHE.mkdir(exist_ok=True)
    if MODE == 'all':
        for mode in ('paper', 'lcd'):
            subprocess.run([sys.executable, __file__, mode], check=True)
        subprocess.run([sys.executable, __file__, 'report'], check=True)
        return
    if MODE == 'report':
        report()
        return
    L.SELECTION.update(json.load(open(L.CACHE / 'frame_selection.json')))
    run_paper() if MODE == 'paper' else run_lcd()


if __name__ == '__main__':
    main()
