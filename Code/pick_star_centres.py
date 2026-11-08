"""Pick the Siemens star centres (and optionally radii) by hand, one pop-up window per capture.

Run:  py pick_star_centres.py            paper session: no-aperture star + Levin / Nayar / Circle
      py pick_star_centres.py lcd        LCD session
      py pick_star_centres.py lcd Nayar  only some captures (Lens, Levin, Nayar, Circle)
Then re-run  py lab3_paper_aperture.py [--lcd]  and everything after it.

Picks are saved to star_centres.json (raw sensor coordinates, keyed by capture folder) and replace the automatic
star search in lab3_paper_aperture.py Part 7. The window shows the capture upright; d cycles the views (capture,
shading removed, and for LCD the Wiener-restored star).

  left click      set the centre
  right click     set the radius (click the outer edge of the spokes)
  arrow keys      nudge the centre by 0.5 px      + / -   radius +/- 2 px
  f               refine the centre to sub-pixel (best ring fidelity within +/-3 px)
  z / x           zoom to the star / full frame
  d               cycle the views
  enter           save this pick, next capture
  a               no pick: use the automatic search for this capture, next capture
  n               skip (leave any existing pick unchanged), next capture
  q / escape      save and quit
"""
import sys
import json

MODE = 'lcd' if 'lcd' in sys.argv[1:] else 'paper'
ONLY = [a for a in sys.argv[1:] if a not in ('lcd', 'paper')]
if MODE == 'lcd':
    sys.argv.append('--lcd')

import numpy as np
from scipy import ndimage as ndi
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import lab3_paper_aperture as L
plt.switch_backend('TkAgg')                     # the lab module selects Agg on import

SPOKE_RINGS = (0.12, 0.5, 0.9)          # the three Part 7 circles, as fractions of R
# free the keys used below from matplotlib's default shortcuts
for _k in ('keymap.fullscreen', 'keymap.quit', 'keymap.save', 'keymap.back', 'keymap.forward',
           'keymap.home', 'keymap.xscale', 'keymap.yscale', 'keymap.zoom', 'keymap.pan'):
    plt.rcParams[_k] = []


def to_up(r, c, H):
    """Raw (row, col) -> upright view (row, col)."""
    return c, H - 1 - r


def to_raw(i, j, H):
    return H - 1 - j, i


def enhance(g):
    """The capture with its slow shading removed and pixel noise smoothed (shows faint spokes)."""
    return ndi.gaussian_filter(g - ndi.gaussian_filter(g, 40), 2)


def load_items():
    """(key, label, folder, images {name: raw array}, automatic geometry or None) per capture."""
    L.SELECTION.update(json.load(open(L.CACHE / 'frame_selection.json')))
    auto = {}
    rf = L.OUT / 'results.json'
    if rf.exists():
        auto = {k: (tuple(v['centre_rc']), v['R_px'])
                for k, v in json.load(open(rf)).get('part7', {}).get('star_geometry', {}).items()}
    items = [('Lens', 'No aperture, 1000 mm', L.LENS_STAR)] + [(m, f'{L.EXP} {m}, {L.D_FAR} mm', L.STAR[m]) for m in L.MASKS]
    for k, label, folder in items:
        if ONLY and k not in ONLY:
            continue
        g, _ = L.best_green(folder, 'scene')
        imgs = {'captured': g, 'captured, shading removed': enhance(g)}
        if MODE == 'lcd' and k != 'Lens':
            gp, _ = L.best_green(L.PSF[k][L.D_FAR], 'psf')
            psf = L.extract_psf(gp)[0]
            imgs = {'captured, shading removed': imgs['captured, shading removed'],
                    'restored (Wiener, K=1e-4)': L.wiener_bank(L.norm_img(g), psf)(1e-4), 'captured': g}
        yield k, label, folder, imgs, auto.get(k)


class Picker:
    def __init__(self, key, label, folder, imgs, auto, pick):
        self.key, self.label, self.folder, self.imgs = key, label, folder, imgs
        self.names = list(imgs)
        self.view = 0
        self.H = next(iter(imgs.values())).shape[0]
        self.auto = auto
        if pick:
            self.c, self.R, self.state = tuple(pick['centre_rc']), pick.get('R_px'), 'existing pick'
        elif auto:
            self.c, self.R, self.state = auto[0], auto[1], 'automatic (not saved unless you press enter)'
        else:
            self.c, self.R, self.state = None, None, 'none'
        self.changed = False
        self.result = None                           # 'save' | 'auto' | 'skip' | 'quit'
        self.fig, self.ax = plt.subplots(figsize=(9, 11))
        self.fig.canvas.mpl_connect('button_press_event', self.click)
        self.fig.canvas.mpl_connect('key_press_event', self.key_press)
        self.im = None
        self.artists = []
        self.draw_image()
        self.draw_overlay()
        if self.c is not None and self.R:
            self.zoom()

    # ---- drawing
    def draw_image(self):
        a = L.up(self.imgs[self.names[self.view]])
        lo, hi = (0.5, 99.5) if self.names[self.view] == 'captured' else (1, 99)
        a = L.stretch(a, lo, hi)
        if self.im is None:
            self.im = self.ax.imshow(a, cmap='gray', interpolation='nearest')
        else:
            self.im.set_data(a)
        self.ax.set_xticks([]); self.ax.set_yticks([])

    def draw_overlay(self):
        for a in self.artists:
            a.remove()
        self.artists = []
        if self.auto and self.state != 'automatic (not saved unless you press enter)':
            i, j = to_up(*self.auto[0], self.H)
            self.artists += self.ax.plot(j, i, 'x', color='#eda100', ms=10, mew=1.5)
            self.artists.append(self.ax.add_patch(plt.Circle((j, i), self.auto[1], fill=False, ec='#eda100',
                                                             lw=0.8, ls='--')))
        if self.c is not None:
            i, j = to_up(*self.c, self.H)
            self.artists += self.ax.plot(j, i, '+', color='#1baf7a', ms=22, mew=1.2)
            if self.R:
                for f in SPOKE_RINGS:
                    self.artists.append(self.ax.add_patch(plt.Circle((j, i), f * self.R, fill=False,
                                                                     ec='#1baf7a', lw=0.8, alpha=0.7)))
                self.artists.append(self.ax.add_patch(plt.Circle((j, i), self.R, fill=False, ec='#2a78d6', lw=1.4)))
        c = 'none' if self.c is None else f'centre (row {self.c[0]:.1f}, col {self.c[1]:.1f}) raw'
        R = '' if not self.R else f', R = {self.R:.0f} px'
        self.ax.set_title(f'{self.label}: {self.names[self.view]}\n{c}{R}  [{self.state}]\n'
                          'L-click centre · R-click radius · arrows nudge · +/- radius · f refine · z/x zoom'
                          + ' · d view' +
                          '\nenter save+next · a use automatic · n skip · q quit'
                          + ('\n(orange dashed: automatic result)' if self.auto else ''), fontsize=9)
        self.fig.canvas.draw_idle()

    def zoom(self):
        i, j = to_up(*self.c, self.H)
        h = 1.3 * (self.R or 300)
        self.ax.set_xlim(j - h, j + h); self.ax.set_ylim(i + h, i - h)
        self.fig.canvas.draw_idle()

    def set(self, c=None, R=None):
        if c is not None:
            self.c = (float(c[0]), float(c[1]))
        if R is not None:
            self.R = float(R)
        self.changed = True
        self.state = 'edited (enter to save)'
        self.draw_overlay()

    # ---- events
    def click(self, ev):
        tb = self.fig.canvas.toolbar
        if ev.inaxes is not self.ax or (tb is not None and getattr(tb.mode, 'value', tb.mode) != ''):
            return
        if ev.button == 1:
            self.set(c=to_raw(ev.ydata, ev.xdata, self.H))
        elif ev.button == 3 and self.c is not None:
            i, j = to_up(*self.c, self.H)
            self.set(R=np.hypot(ev.ydata - i, ev.xdata - j))

    def key_press(self, ev):
        k = ev.key
        if k in ('enter', 'a', 'n', 'q', 'escape'):
            if k == 'enter' and self.c is None:
                return
            self.result = {'enter': 'save', 'a': 'auto', 'n': 'skip', 'q': 'quit', 'escape': 'quit'}[k]
            if k == 'enter' and not self.changed and self.state.startswith('automatic'):
                self.changed = True
            plt.close(self.fig)
        elif k in ('up', 'down', 'left', 'right') and self.c is not None:
            di, dj = {'up': (-0.5, 0), 'down': (0.5, 0), 'left': (0, -0.5), 'right': (0, 0.5)}[k]
            i, j = to_up(*self.c, self.H)
            self.set(c=to_raw(i + di, j + dj, self.H))
        elif k in ('+', '=', '-') and self.R:
            self.set(R=self.R + (2 if k != '-' else -2))
        elif k == 'f' and self.c is not None and self.R:
            img = self.imgs[self.names[self.view]]
            self.set(c=L.refine_centre_fidelity(img / np.percentile(img, 99.9), self.c, self.R, half=3.0, step=0.25))
        elif k == 'z' and self.c is not None:
            self.zoom()
        elif k == 'x':
            self.ax.set_xlim(-0.5, self.H - 0.5); self.ax.set_ylim(self.im.get_array().shape[0] - 0.5, -0.5)
            self.fig.canvas.draw_idle()
        elif k == 'd':
            self.view = (self.view + 1) % len(self.names)
            self.draw_image()
            self.draw_overlay()


def main():
    picks = json.load(open(L.STAR_PICKS)) if L.STAR_PICKS.exists() else {}
    print(__doc__.split('\n\n')[-1])
    for key, label, folder, imgs, auto in load_items():
        sk = L.star_key(folder)
        p = Picker(key, label, folder, imgs, auto, picks.get(sk))
        plt.show()
        if p.result == 'auto':
            picks.pop(sk, None)
            print(f'{key}: automatic')
        elif p.result in ('save', 'quit', None) and p.changed and p.c is not None:
            picks[sk] = dict(centre_rc=[round(p.c[0], 2), round(p.c[1], 2)],
                             R_px=round(p.R, 1) if p.R else None)
            print(f'{key}: picked {picks[sk]}')
        else:
            print(f'{key}: unchanged')
        json.dump(picks, open(L.STAR_PICKS, 'w'), indent=1)
        if p.result in ('quit', None):               # closing the window counts as quit
            break
    print(f'saved -> {L.STAR_PICKS}\nnow re-run:  py lab3_paper_aperture.py' + (' --lcd' if MODE == 'lcd' else ''))


if __name__ == '__main__':
    main()
