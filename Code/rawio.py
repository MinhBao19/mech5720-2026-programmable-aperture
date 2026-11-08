"""Reader for the uncompressed 10-bit DNGs written by the capture tool (ov5647, GBRG, black 16, white 1023)."""
import struct
from pathlib import Path

import numpy as np

BLACK_LEVEL = 16
WHITE_LEVEL = 1023


def _tags(data):
    bo = '<' if data[:2] == b'II' else '>'
    off = struct.unpack(bo + 'I', data[4:8])[0]
    n = struct.unpack(bo + 'H', data[off:off + 2])[0]
    tags = {}
    for i in range(n):
        e = data[off + 2 + 12 * i: off + 14 + 12 * i]
        tag, typ, cnt = struct.unpack(bo + 'HHI', e[:8])
        if typ == 3 and cnt == 1:
            tags[tag] = struct.unpack(bo + 'H', e[8:10])[0]
        elif typ == 4 and cnt == 1:
            tags[tag] = struct.unpack(bo + 'I', e[8:12])[0]
        elif typ == 1 and cnt <= 4:
            tags[tag] = tuple(e[8:8 + cnt])
    return tags


def read_bayer(path):
    """Raw Bayer mosaic as float64 (black level not subtracted) and the CFA pattern (0=R 1=G 2=B)."""
    data = Path(path).read_bytes()
    t = _tags(data)
    w, h, bps = t[256], t[257], t[258]
    off, nbytes = t[273], t[279]
    assert bps == 10 and t[259] == 1, 'expected uncompressed 10-bit DNG'
    buf = np.frombuffer(data, np.uint8, nbytes, off).reshape(h, -1)[:, : w * 5 // 4]
    b = buf.reshape(h, -1, 5).astype(np.uint16)
    px = np.empty((h, w // 4, 4), np.uint16)
    px[..., 0] = (b[..., 0] << 2) | (b[..., 1] >> 6)
    px[..., 1] = ((b[..., 1] & 0x3F) << 4) | (b[..., 2] >> 4)
    px[..., 2] = ((b[..., 2] & 0x0F) << 6) | (b[..., 3] >> 2)
    px[..., 3] = ((b[..., 3] & 0x03) << 8) | b[..., 4]
    cfa = t.get(33422, (1, 2, 0, 1))
    return px.reshape(h, w).astype(np.float64), cfa


def green_channel(mosaic, cfa):
    """Full-resolution green plane: green sites kept, other sites the mean of their 4 green neighbours."""
    h, w = mosaic.shape
    gmask = np.zeros((h, w), bool)
    for k, c in enumerate(cfa):
        if c == 1:
            gmask[k // 2::2, k % 2::2] = True
    g = np.where(gmask, mosaic, 0.0)
    pad = np.pad(g, 1, mode='reflect')
    padm = np.pad(gmask.astype(float), 1, mode='reflect')
    s = pad[:-2, 1:-1] + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:]
    c = padm[:-2, 1:-1] + padm[2:, 1:-1] + padm[1:-1, :-2] + padm[1:-1, 2:]
    return np.where(gmask, mosaic, s / np.maximum(c, 1))
