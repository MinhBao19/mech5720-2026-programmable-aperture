# Captures

Raw captures for every aperture, target and distance used in the report.

The full set is about **10 GB**, which is too large for GitHub (100 MB per-file limit), so it is stored on the University of Sydney's OneDrive:

**Download:** [MECH5720 captures (OneDrive)](https://unisydneyedu-my.sharepoint.com/shared?listurl=https%3A%2F%2Funisydneyedu%2Dmy%2Esharepoint%2Ecom%2Fpersonal%2Fpcru2199%5Funi%5Fsydney%5Fedu%5Fau%2FDocuments&id=%2Fpersonal%2Fpcru2199%5Funi%5Fsydney%5Fedu%5Fau%2FDocuments%2F5720Images&shareLink=1&ga=1)

To use the data:

1. Open the link and download the folder. OneDrive packages a folder download as a single zip.
2. Extract it anywhere. The layout below is what you should see inside.
3. Point the processing scripts in [`Code/`](../Code) at the extracted folder.

If the link does not open, you may need to sign in with a University of Sydney account. Contact the group through this repository for access.

---

## What is in the zip

```
captures_MECH/
├── PaperAperture/
│   ├── Paper-Part0/LensBlackLevel/          Dark frames (lens covered)
│   ├── Paper-Part1/                         No-aperture lens reference (Siemens star, lens PSF)
│   ├── Levin - Paper/                       Levin paper mask
│   ├── Nayar - Paper/                       Zhou–Nayar paper mask
│   └── Circle - Paper/                      Circle paper mask
└── LCDApertures/
    ├── LCD-Part1/                           Open-square LCD PSF at five exposures
    ├── LCDLevin/                            Levin pattern on the LCD
    ├── LCDNayar/                            Zhou–Nayar pattern on the LCD
    └── LCDCircle/                           Circle pattern on the LCD
```

Inside each paper-mask folder, captures follow the Lab 3 parts:

| Part | Target |
|---|---|
| `Part2/` | Point spread function (PSF) at 1000, 1150 and 1300 mm, plus the four corners at 1300 mm |
| `Part3/` | 36-spoke Siemens star |
| `Part4/` | Natural scenes (`Pengmin`, `NielLeSeal`) at 1000, 1150 and 1300 mm |

The LCD folders contain the same targets directly, without the `PartN` level.

Every capture folder holds:

- `dng/`: 10 raw frames (`close_001.dng` … `close_010.dng`)
- `png/`: the same 10 frames as 16-bit linear RGB PNGs, made from the DNGs with no automatic brightening
- `manifest.json`: frame list and timestamps
- `camera_settings_paired_capture.json`: camera settings requested for the capture

All captures are 1920 × 1080 with analogue gain 1 and auto-exposure off.

## Folder naming

Folder names encode the aperture, the target and the camera settings:

```
LCDLevinPSFG1E60000D1000
│  │    │  │ │     └── D1000:  target distance 1000 mm
│  │    │  │ └──────── E60000: exposure 60 000 µs
│  │    │  └────────── G1:     analogue gain 1
│  │    └───────────── PSF:    target (PSF, SiemensStar, Pengmin, NielLeSeal)
│  └────────────────── Levin:  pattern (Levin, Nayar, Circle)
└───────────────────── LCD:    aperture (paper-mask folders omit this)
```

Corner PSFs add the corner to the name, for example `LCDLevinTopLeftPSFG1E60000D1300`. A folder ending in `-Attempt2` is a repeat of a capture; the original is kept alongside it.

## Exposures

| | Paper mask | LCD |
|---|---|---|
| PSF | 500 µs | 60 000 µs |
| Siemens star and scenes | 64 000 µs | 60 000 – 256 000 µs |

The LCD's polarisers absorb most of the light, so it needs much longer exposures than the paper masks.
