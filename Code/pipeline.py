"""Shared processing: LCD PSF decomposition, Wiener deconvolution and Siemens-star measures."""
import numpy as np
import cv2
from scipy import ndimage as ndi
from scipy.optimize import nnls
from scipy.signal import find_peaks


# ----------------------------------------------------------------------------- LCD PSF model
def order_spacing(psf, max_lag=200):
    """Vertical diffraction-order spacing (px) from the PSF autocorrelation."""
    c = psf.shape[0] // 2
    strip = psf[:, c - 40:c + 40].sum(1)
    strip = strip - ndi.gaussian_filter1d(strip, 40)
    ac = np.correlate(strip, strip, 'full')[len(strip) - 1:]
    lags = np.arange(len(ac))
    m = (lags > 20) & (lags < max_lag)
    pk, _ = find_peaks(ac * m)
    return float(lags[pk[0]]) if len(pk) else np.nan


def shift_y(img, dy):
    """Sub-pixel vertical shift (Fourier shift theorem)."""
    fy = np.fft.fftfreq(img.shape[0])[:, None]
    return np.real(np.fft.ifft2(np.fft.fft2(img) * np.exp(-2j * np.pi * fy * dy)))


def decompose_lcd_psf(psf, d, n_orders=6, blob_half_h=None, blob_half_w=160, iters=30, halo_sigma=12):
    """Split an LCD PSF into  psf = comb_d * blob + halo.

    comb_d = sum_k w_k delta(y - k d) are the LCD diffraction orders (spacing d, independent of focus), blob is
    the defocused image of the displayed aperture and halo is the smooth leakage through 'off' pixels.
    Alternating least squares: w_k by NNLS given the blob, the blob by support-constrained comb deconvolution."""
    n = psf.shape[0]
    c = n // 2
    blob_half_h = blob_half_h or int(0.75 * d)
    ks = np.arange(-n_orders, n_orders + 1)
    support = np.zeros_like(psf, bool)
    support[c - blob_half_h:c + blob_half_h + 1, c - blob_half_w:c + blob_half_w + 1] = True
    w = np.ones(len(ks)) / len(ks)
    fy = np.fft.fftfreq(n)[:, None]
    F = np.fft.fft2(psf)
    for _ in range(iters):
        W = sum(wk * np.exp(-2j * np.pi * fy * k * d) for wk, k in zip(w, ks))
        blob = np.real(np.fft.ifft2(F * np.conj(W) / (np.abs(W) ** 2 + 1e-2 * np.max(np.abs(W)) ** 2)))
        blob = np.clip(np.where(support, blob, 0), 0, None)
        A = np.stack([shift_y(blob, k * d).ravel() for k in ks], 1)
        w, _ = nnls(A, psf.ravel())
        s = w.sum()
        w /= s
        blob *= s
    chain = sum(wk * shift_y(blob, k * d) for wk, k in zip(w, ks))
    halo = np.clip(ndi.gaussian_filter(psf - chain, halo_sigma), 0, None)
    return dict(blob=blob, weights=w, orders=ks, d=d, halo=halo, chain=chain)


def rebuild_lcd_psf(dec, scale=1.0):
    """PSF for another defocus: blob and halo scaled about the centre, order spacing kept."""
    def zoom_c(img, s):
        c = (img.shape[0] - 1) / 2
        yy, xx = np.indices(img.shape, dtype=float)
        return ndi.map_coordinates(img, [(yy - c) / s + c, (xx - c) / s + c], order=1, cval=0.0) / s ** 2
    blob = zoom_c(dec['blob'], scale)
    psf = sum(wk * shift_y(blob, k * dec['d']) for wk, k in zip(dec['weights'], dec['orders']))
    psf = np.clip(psf + zoom_c(dec['halo'], scale), 0, None)
    return psf / psf.sum()


# ----------------------------------------------------------------------------- deconvolution
def psf_to_otf(psf, shape):
    """Zero-pad a centred PSF to `shape` and shift its centre to (0, 0)."""
    pad = np.zeros(shape)
    ph, pw = psf.shape
    pad[:ph, :pw] = psf
    pad = np.roll(pad, (-(ph // 2), -(pw // 2)), axis=(0, 1))
    return np.fft.fft2(pad)


def pad_for_fft(img, margin):
    """Reflect-pad and taper to suppress wrap-around at the frame border."""
    p = cv2.copyMakeBorder(img, margin, margin, margin, margin, cv2.BORDER_REFLECT)
    H, W = p.shape
    wy, wx = np.ones(H), np.ones(W)
    t = np.hanning(2 * margin)
    wy[:margin], wy[-margin:] = t[:margin], t[margin:]
    wx[:margin], wx[-margin:] = t[:margin], t[margin:]
    mean = img.mean()
    return (p - mean) * np.outer(wy, wx) + mean


def wiener(img, psf, K, margin=None):
    """Wiener deconvolution X = H* Y / (|H|^2 + K) with reflective padding."""
    margin = margin or psf.shape[0] // 2
    p = pad_for_fft(img, margin)
    H = psf_to_otf(psf, p.shape)
    out = np.real(np.fft.ifft2(np.conj(H) * np.fft.fft2(p) / (np.abs(H) ** 2 + K)))
    return out[margin:-margin, margin:-margin]


def inverse_filter(img, psf, eps=1e-12, margin=None):
    """Naive inverse filter X = Y / H."""
    margin = margin or psf.shape[0] // 2
    p = pad_for_fft(img, margin)
    H = psf_to_otf(psf, p.shape)
    H = np.where(np.abs(H) < eps, eps, H)
    out = np.real(np.fft.ifft2(np.fft.fft2(p) / H))
    return out[margin:-margin, margin:-margin]


def var_laplacian(img):
    return cv2.Laplacian(img.astype(np.float64), cv2.CV_64F).var()


# ----------------------------------------------------------------------------- Siemens star
def sample_circle(img, centre, r, n_theta):
    """Bilinear samples along a circle; NaN where it leaves the frame."""
    th = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    x = centre[1] + r * np.cos(th)
    y = centre[0] + r * np.sin(th)
    v = ndi.map_coordinates(img, [y, x], order=1, mode='constant', cval=np.nan)
    inside = (x >= 0) & (x <= img.shape[1] - 1) & (y >= 0) & (y <= img.shape[0] - 1)
    v[~inside] = np.nan
    return th, v


def estimate_spoke_count(img, centre, r):
    """Dominant angular frequency around the circle (number of black/white cycles)."""
    _, v = sample_circle(img, centre, r, 4096)
    v = np.where(np.isnan(v), np.nanmean(v), v) - np.nanmean(v)
    spec = np.abs(np.fft.rfft(v))
    spec[:8] = 0
    return int(np.argmax(spec))


def refine_centre(img, centre, r_list, N, search=25, step=1):
    """Star centre that maximises the power at the spoke frequency N on several circles."""
    best, best_c = -1, centre
    for dy in range(-search, search + 1, step):
        for dx in range(-search, search + 1, step):
            c = (centre[0] + dy, centre[1] + dx)
            p = 0
            for r in r_list:
                _, v = sample_circle(img, c, r, 2048)
                if np.isnan(v).mean() > 0.3:
                    continue
                v = np.where(np.isnan(v), np.nanmean(v), v)
                p += np.abs(np.fft.rfft(v - v.mean())[N]) / (np.std(v) + 1e-9)
            if p > best:
                best, best_c = p, c
    return best_c


def star_phase(img, centre, N, r):
    """Angular phase of the spoke pattern."""
    _, v = sample_circle(img, centre, r, 4096)
    v = np.where(np.isnan(v), np.nanmean(v), v)
    return np.angle(np.fft.rfft(v - v.mean())[N])


def harmonic_contrast(img, centre, r, N):
    """Amplitude of the N-th angular harmonic / mean, times pi/4: 1 for an ideal binary star, MTF(f) when blurred."""
    n_theta = max(int(2 * np.pi * r * 2), 8 * N)
    _, v = sample_circle(img, centre, r, n_theta)
    v = np.where(np.isnan(v), np.nan, np.clip(v, 0, None))
    valid = ~np.isnan(v)
    if valid.mean() < 0.7:
        return np.nan
    th = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)[valid]
    A = np.column_stack([np.ones_like(th), np.cos(N * th), np.sin(N * th)])
    a, b, c = np.linalg.lstsq(A, v[valid], rcond=None)[0]
    return np.pi / 4 * np.hypot(b, c) / a


def ring_fidelity(img, centre, N, phi, r_min, r_max, n_r=30):
    """Mean over log-spaced circles of corr(angular profile, ideal square wave). 0 = no spokes, 1 = ideal."""
    out = []
    for r in np.geomspace(r_min, r_max, n_r):
        n_theta = max(int(2 * np.pi * r * 2), 8 * N)
        th, v = sample_circle(img, centre, r, n_theta)
        ok = ~np.isnan(v)
        if ok.mean() < 0.7:
            continue
        out.append(np.corrcoef(v[ok], np.sign(np.cos(N * th + phi))[ok])[0, 1])
    return float(np.mean(out))
