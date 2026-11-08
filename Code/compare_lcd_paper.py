"""Side-by-side figures of the paper and LCD apertures, from what lab3_paper_aperture.py saved for each
(figdata.npz, results.json). Run both lab modes first.

Run:  py compare_lcd_paper.py   -> output_lab3_compare/
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
EXPS = {'paper': HERE / 'output_lab3_paper', 'LCD': HERE / 'output_lab3_lcd'}
OUT = HERE / 'output_lab3_compare'
MASKS = ['Levin', 'Nayar', 'Circle']
DISTANCES = [1000, 1150, 1300]
D_FAR = 1300
SCENES = ['Pengmin', 'NielLeSeal']
SCENE_CROP = (600, 250, 560)            # upright (row, col, size)

COL = {'paper': '#2a78d6', 'LCD': '#eb6834'}
INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#898781', '#e1e0d9'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.titlecolor': INK,
                     'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.spines.top': False,
                     'axes.spines.right': False, 'savefig.dpi': 160, 'figure.facecolor': 'white'})

D = {e: np.load(p / 'figdata.npz') for e, p in EXPS.items()}
R = {e: json.load(open(p / 'results.json')) for e, p in EXPS.items()}


# ---------------------------------------------------------------------------- helpers
def stretch(a, lo=0.5, hi=99.5):
    l, h = np.percentile(a, [lo, hi])
    return np.clip((a - l) / (h - l + 1e-12), 0, 1)


def imshow(ax, a, title=None, cmap='gray', **kw):
    ax.imshow(a, cmap=cmap, **kw)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=8.5)


def tag(ax, exp):
    ax.text(0.02, 0.98, exp, transform=ax.transAxes, ha='left', va='top', fontsize=8, fontweight='bold',
            color='white', bbox=dict(boxstyle='round,pad=0.25', fc=COL[exp], lw=0))


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / name, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  wrote', (OUT / name).relative_to(HERE))


def star_ok(exp, m):
    return R[exp]['part7'].get('star_located', {}).get(m, True)


def crop(a):
    r, c, n = SCENE_CROP
    return a[r:r + n, c:c + n]


# ---------------------------------------------------------------------------- figures
def fig_psfs():
    fig, axs = plt.subplots(3, 4, figsize=(10.5, 8.2))
    for i, m in enumerate(MASKS):
        for j, (exp, log) in enumerate([('paper', False), ('LCD', False), ('paper', True), ('LCD', True)]):
            p = D[exp][f'psf_{m}_{D_FAR}']
            if log:
                imshow(axs[i, j], np.log10(np.maximum(p / p.max(), 1e-4)), cmap='magma', vmin=-4, vmax=0)
            else:
                imshow(axs[i, j], p, vmin=0, vmax=p.max())
            tag(axs[i, j], exp)
            if i == 0:
                axs[i, j].set_title('linear' if not log else 'log10 (4 decades)', fontsize=9)
        axs[i, 0].set_ylabel(m, fontsize=10, color=INK)
        axs[i, 0].set_yticks([])
    fig.suptitle(f'PSFs at {D_FAR} mm, same 256 x 256 px window (green, upright view)', fontsize=10.5)
    fig.tight_layout()
    save(fig, 'compare_psfs.png')


def fig_psf_scale():
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for ax, m in zip(axs, MASKS):
        for exp in COL:
            s = [float(D[exp][f'scale_{m}_{d}']) for d in DISTANCES]
            ax.plot(DISTANCES, s, '-o', color=COL[exp], lw=2, ms=6, mec='white', label=exp)
            if exp == 'LCD' and m == 'Nayar':
                ax.annotate('fit unreliable', (DISTANCES[-1], s[-1]), xytext=(-60, -4), textcoords='offset points',
                            fontsize=7.5, color=INK2, arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.8))
        ax.set_title(m, fontsize=10); ax.grid(True, axis='y')
        ax.set_xticks(DISTANCES); ax.set_xlabel('distance (mm)')
    axs[0].set_ylabel('fitted aperture scale (px per mask cell)')
    axs[0].legend(frameon=False, fontsize=8)
    fig.suptitle('Size of the aperture blur vs distance (camera focused near 1 m)', fontsize=10.5)
    fig.tight_layout()
    save(fig, 'compare_psf_scale.png')


def fig_spectra():
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.9), sharey=True)
    for ax, m in zip(axs, MASKS):
        f, H = D['paper'][f'radial_{m}_design_f'], D['paper'][f'radial_{m}_design_H']
        ax.plot(f, 20 * np.log10(H), ':', color=INK2, lw=1.4, label='design (at the paper PSF\'s scale)')
        for exp in COL:
            f, H = D[exp][f'radial_{m}_captured_f'], D[exp][f'radial_{m}_captured_H']
            ax.plot(f, 20 * np.log10(np.maximum(H, 1e-6)), color=COL[exp], lw=1.8, label=f'{exp} (captured)')
        ax.set_title(f'{m}, {D_FAR} mm', fontsize=10); ax.grid(True)
        ax.set_xlim(0, 0.5); ax.set_ylim(-90, 2)
        ax.set_xlabel('spatial frequency (cycles/px)')
    axs[0].set_ylabel('radially averaged |H| (dB re DC)')
    axs[0].legend(frameon=False, fontsize=8)
    fig.suptitle('Magnitude spectrum of the PSF: what each aperture passes', fontsize=10.5)
    fig.tight_layout()
    save(fig, 'compare_spectra.png')


def fig_mtf():
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.9), sharey=True)
    for ax, m in zip(axs, MASKS):
        Ct = R['paper']['part7']['target_contrast']
        f, C = D['paper']['mtf_Lens_f'], D['paper']['mtf_Lens_C'] / Ct
        ax.plot(f, C, ':', color=INK2, lw=1.6, label='no aperture, 1000 mm (in focus)')
        for exp in COL:
            Ct = R[exp]['part7']['target_contrast']
            f, C = D[exp][f'mtf_{m}_f'], D[exp][f'mtf_{m}_C'] / Ct
            ok = star_ok(exp, m)
            ax.plot(f, C, '-o' if ok else '--', ms=2.5, color=COL[exp], lw=1.6,
                    label=exp + ('' if ok else ' (star not located)'))
        ax.set_xscale('log'); ax.set_xlim(0.008, 0.6); ax.set_ylim(-0.02, 1.08)
        ax.set_title(f'{m}, {D_FAR} mm', fontsize=10); ax.grid(True)
        ax.set_xlabel('spatial frequency (cycles/px)')
        ax.legend(frameon=False, fontsize=7.5, loc='upper right')
    axs[0].set_ylabel('MTF (Michelson contrast / target contrast)')
    fig.suptitle('MTF measured on the Siemens star (raw captures)', fontsize=10.5)
    fig.tight_layout()
    save(fig, 'compare_mtf.png')


def fig_star_recon():
    fig, axs = plt.subplots(3, 4, figsize=(11, 8.6))
    for i, m in enumerate(MASKS):
        for j, (exp, kind) in enumerate([('paper', 'captured'), ('paper', 'wiener'), ('LCD', 'captured'), ('LCD', 'wiener')]):
            best = R[exp]['part8']['best'][m]
            if kind == 'captured':
                t = f'{m}: captured (fidelity {best["fidelity_captured"]:.2f})'
            else:
                t = f'Wiener K={float(D[exp][f"star_{m}_K"]):.0e} (fidelity {best["fidelity_max"]:.2f})'
            if not star_ok(exp, m):
                t += '\nstar not located'
            elif kind == 'wiener' and best['fidelity_max'] < best['fidelity_captured']:
                t += '\nmatched PSF does not fit this star'
            imshow(axs[i, j], stretch(D[exp][f'star_{m}_{kind}']), t)
            tag(axs[i, j], exp)
    fig.suptitle(f'Siemens star at {D_FAR} mm: capture and Wiener deconvolution (K at maximum fidelity)',
                 fontsize=10.5)
    fig.tight_layout()
    save(fig, 'compare_star_recon.png')


def fig_natural():
    for s in SCENES:
        for d in DISTANCES:
            fig, axs = plt.subplots(3, 4, figsize=(10.5, 8.4))
            for i, m in enumerate(MASKS):
                for j, (exp, kind) in enumerate([('paper', 'captured'), ('paper', 'wiener'),
                                                 ('LCD', 'captured'), ('LCD', 'wiener')]):
                    a = crop(D[exp][f'nat_{s}_{m}_{d}_{kind}'])
                    K = R[exp]['part8']['K_used_natural'][m]
                    imshow(axs[i, j], stretch(a, 0.2, 99.8), f'{m}: ' + ('captured' if kind == 'captured'
                                                                         else f'Wiener K={K:.0e}'))
                    tag(axs[i, j], exp)
            fig.suptitle(f'{s} at {d} mm: same crop in both sessions, each depth deconvolved with its own PSF',
                         fontsize=10.5)
            fig.tight_layout()
            save(fig, f'compare_natural_{s}_{d}mm.png')


def fig_summary():
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    x = np.arange(len(MASKS))
    w = 0.36
    for k, exp in enumerate(COL):
        fid = [R[exp]['part8']['best'][m]['fidelity_max'] for m in MASKS]
        cap = [R[exp]['part8']['best'][m]['fidelity_captured'] for m in MASKS]
        xs = x + (k - 0.5) * (w + 0.02)
        axs[0].bar(xs, fid, w, color=COL[exp], label=f'{exp}: after Wiener')
        axs[0].scatter(xs, cap, marker='_', s=260, color=INK, linewidths=2, zorder=3,
                       label='captured (before)' if k == 0 else None)
        for xi, v, c, m in zip(xs, fid, cap, MASKS):
            axs[0].text(xi, max(v, 0) + 0.012, f'{v:.2f}', ha='center', fontsize=8, color=INK2)
            if v < c:
                best_xc = max(v2['fidelity'] for k2, v2 in R[exp]['part8']['psf_cross_check'].items()
                              if k2.startswith(f'{m} star'))
                txt = (f'PSF mismatch\n({best_xc:.2f} with\nanother distance\'s PSF)' if best_xc > c + 0.05
                       else 'no spokes\nrecovered')
                axs[0].annotate(txt, (xi, max(c, v)), xytext=(0, 22), textcoords='offset points', ha='center',
                                fontsize=7, color=INK2)
        edge = [R[exp]['part7']['three_circles'][m]['edge']['contrast'] for m in MASKS]
        axs[1].bar(xs, edge, w, color=COL[exp], label=exp)
        for xi, v in zip(xs, edge):
            axs[1].text(xi, v + 0.012, f'{v:.2f}', ha='center', fontsize=8, color=INK2)
    for ax in axs:
        ax.set_xticks(x, MASKS); ax.grid(True, axis='y'); ax.axhline(0, color=MUTED, lw=0.8)
        ax.legend(frameon=False, fontsize=7.5, loc='upper center', ncol=3)
    axs[0].set_ylabel('ring fidelity (corr. with ideal star)')
    axs[0].set_title(f'Star restoration at {D_FAR} mm (bar: best Wiener, tick: raw capture)', fontsize=9.5)
    axs[1].set_ylabel('Michelson contrast')
    axs[1].set_title(f'Raw star contrast, outer circle (~0.015 cycles/px), {D_FAR} mm', fontsize=9.5)
    axs[0].set_ylim(-0.05, 0.75); axs[1].set_ylim(0, 0.68)
    fig.tight_layout()
    save(fig, 'compare_summary.png')


if __name__ == '__main__':
    fig_psfs()
    fig_psf_scale()
    fig_spectra()
    fig_mtf()
    fig_star_recon()
    fig_natural()
    fig_summary()
    print('done ->', OUT)
