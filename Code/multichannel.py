"""Joint RGB deconvolution with unknown out-of-frame content (LCD methods M4 and M5).

The LCD diffraction-order spacing scales with wavelength, so each colour's OTF has its notches at different
frequencies: y_c = a_c (h_c * L) + n_c with one shared latent image L. L is solved on a canvas extended by the PSF
half-width, and only in-frame pixels enter the data term (mask M), so light from outside the frame is modelled
instead of invented by padding. The mask breaks FFT diagonalisation, so the solves use preconditioned CG with the
unmasked Fourier-diagonal operator as preconditioner.
"""
import numpy as np

import pipeline as P


def _setup(ys, psfs, pad):
    chans = list(ys)
    h, w = ys[chans[0]].shape
    pad = pad or psfs[chans[0]].shape[0] // 2
    shape = (h + 2 * pad, w + 2 * pad)
    M = np.zeros(shape)
    M[pad:pad + h, pad:pad + w] = 1.0
    H = {c: P.psf_to_otf(psfs[c], shape) for c in chans}
    return chans, h, w, pad, shape, M, H


def _conv(x, Hc, conj=False):
    return np.real(np.fft.ifft2(np.fft.fft2(x) * (np.conj(Hc) if conj else Hc)))


def joint_wiener_masked(ys, psfs, gains, weights, K, pad=None, iters=80, tol=1e-6):
    """min_L  sum_c w_c ||M (a_c h_c * L) - y_c||^2 + K ||L||^2   (M4)."""
    chans, h, w, pad, _, M, H = _setup(ys, psfs, pad)
    precond = sum(weights[c] * gains[c] ** 2 * np.abs(H[c]) ** 2 for c in chans) + K

    def A(x):
        return sum(weights[c] * gains[c] ** 2 * _conv(M * _conv(x, H[c]), H[c], True) for c in chans) + K * x

    b = sum(weights[c] * gains[c] * _conv(M * np.pad(ys[c], pad), H[c], True) for c in chans)
    x = np.real(np.fft.ifft2(np.fft.fft2(b) / precond))
    r = b - A(x)
    z = np.real(np.fft.ifft2(np.fft.fft2(r) / precond))
    p = z.copy()
    rz = (r * z).sum()
    b_norm = np.linalg.norm(b)
    for _ in range(iters):
        Ap = A(p)
        alpha = rz / (p * Ap).sum()
        x = x + alpha * p
        r = r - alpha * Ap
        if np.linalg.norm(r) / b_norm < tol:
            break
        z = np.real(np.fft.ifft2(np.fft.fft2(r) / precond))
        rz_new = (r * z).sum()
        p = z + (rz_new / rz) * p
        rz = rz_new
    return x[pad:pad + h, pad:pad + w]


def joint_tv(ys, psfs, gains, weights, lam, pad=None, beta0=1.0, beta_max=2 ** 10, beta_rate=2.0, cg_iters=8):
    """Joint TV deconvolution by half-quadratic splitting (M5); lam weights the data term against unit TV.
    Each L-step solves (sum_c lam w_c a_c^2 H_c^T M H_c + beta D^T D) L = sum_c lam w_c a_c H_c^T M y_c + beta D^T w."""
    chans, h, w, pad, shape, M, H = _setup(ys, psfs, pad)
    fy = np.fft.fftfreq(shape[0])[:, None]
    fx = np.fft.fftfreq(shape[1])[None, :]
    Dx, Dy = 1 - np.exp(-2j * np.pi * fx), 1 - np.exp(-2j * np.pi * fy)
    D2 = np.abs(Dx) ** 2 + np.abs(Dy) ** 2
    wl = {c: lam * weights[c] for c in chans}
    HtHsum = sum(wl[c] * gains[c] ** 2 * np.abs(H[c]) ** 2 for c in chans)
    rhs_data = sum(wl[c] * gains[c] * _conv(M * np.pad(ys[c], pad), H[c], True) for c in chans)

    def A(x, beta):
        X = np.fft.fft2(x)
        data = sum(wl[c] * gains[c] ** 2 * _conv(M * np.real(np.fft.ifft2(X * H[c])), H[c], True) for c in chans)
        return data + beta * np.real(np.fft.ifft2(X * D2))

    x = np.real(np.fft.ifft2(np.fft.fft2(rhs_data) / (HtHsum + 1.0 * D2 + 1e-12)))
    beta = beta0
    while beta <= beta_max:
        X = np.fft.fft2(x)
        gx, gy = np.real(np.fft.ifft2(Dx * X)), np.real(np.fft.ifft2(Dy * X))
        t = 1.0 / beta
        wx = np.sign(gx) * np.maximum(np.abs(gx) - t, 0)
        wy = np.sign(gy) * np.maximum(np.abs(gy) - t, 0)
        b = rhs_data + beta * np.real(np.fft.ifft2(np.conj(Dx) * np.fft.fft2(wx) + np.conj(Dy) * np.fft.fft2(wy)))
        precond = HtHsum + beta * D2 + 1e-12
        r = b - A(x, beta)
        z = np.real(np.fft.ifft2(np.fft.fft2(r) / precond))
        p = z.copy()
        rz = (r * z).sum()
        for _ in range(cg_iters):
            Ap = A(p, beta)
            alpha = rz / (p * Ap).sum()
            x = x + alpha * p
            r = r - alpha * Ap
            z = np.real(np.fft.ifft2(np.fft.fft2(r) / precond))
            rz_new = (r * z).sum()
            p = z + (rz_new / rz) * p
            rz = rz_new
        beta *= beta_rate
    return x[pad:pad + h, pad:pad + w]
