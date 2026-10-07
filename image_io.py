"""
image_io.py - Digital image I/O
=====================================
Supports DICOM, PNG, JPG, JPEG, BMP, TIFF at any bit depth (8 or 16)
and any channel count (grayscale or colour).

Only pixel data is returned; no DICOM metadata / PHI ever leaves this module.
Uses OpenCV (cv2) for standard formats and pydicom for DICOM.
"""
import logging
import os

import cv2
import numpy as np

logger = logging.getLogger(__name__)

try:
    import pydicom
    _DICOM_OK = True
except ImportError:
    _DICOM_OK = False

SUPPORTED_EXTENSIONS = {".dcm", ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}


# ── public API ────────────────────────────────────────────────────────────────

def load(path: str) -> tuple:
    """
    Load a digital image file.

    Returns
    -------
    array : np.ndarray
        Shape (H, W) for grayscale or (H, W, 3) for colour.
        dtype uint8 (8-bit) or uint16 (16-bit). Always C-contiguous.
    meta  : dict
        {"channels": int, "bit_depth": int, "format": str, "h": int, "w": int}
    """
    ext = os.path.splitext(path)[1].lower()
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image not found: {path}")
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported format: {ext!r}")

    if ext == ".dcm":
        return _load_dicom(path)
    return _load_standard(path)


def save(path: str, array: np.ndarray, meta: dict = None) -> str:
    """
    Save a recovered image to *path*, preserving bit depth.

    JPEG and BMP cannot store 16-bit; the extension is silently changed to PNG.
    Returns the actual path written.
    """
    array = np.ascontiguousarray(array)
    ext   = os.path.splitext(path)[1].lower()

    if array.dtype == np.uint16 and ext in (".jpg", ".jpeg", ".bmp"):
        path = os.path.splitext(path)[0] + ".png"
        logger.info("16-bit image saved as PNG instead of %s", ext)

    out = _to_bgr(array)
    ok  = cv2.imwrite(path, out)
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed for: {path}")
    return path


# ── DICOM loader ──────────────────────────────────────────────────────────────

def _load_dicom(path: str) -> tuple:
    if not _DICOM_OK:
        raise ImportError(
            "pydicom is required for DICOM files. "
            "Install it with: pip install pydicom"
        )

    ds  = pydicom.dcmread(path)
    logger.info("DICOM metadata / PHI discarded for: %s", os.path.basename(path))

    arr = ds.pixel_array   # ndarray provided by pydicom

    # Multi-frame DICOM → use first frame only
    if arr.ndim == 3 and hasattr(ds, "NumberOfFrames") and int(ds.NumberOfFrames) > 1:
        arr = arr[0]
        logger.warning("Multi-frame DICOM: only frame 0 is used from %s",
                       os.path.basename(path))

    bits = int(getattr(ds, "BitsAllocated", 8))

    # Signed → unsigned shift
    if int(getattr(ds, "PixelRepresentation", 0)) == 1 and bits == 16:
        arr = arr.astype(np.int32)
        arr = arr - int(arr.min())
        max_v = int(arr.max()) or 1
        arr = (arr.astype(np.float32) / max_v * 65535).astype(np.uint16)

    # Normalise dtype
    if bits <= 8:
        arr = arr.astype(np.uint8)
    else:
        arr = arr.astype(np.uint16)

    arr = np.ascontiguousarray(arr)
    H, W     = arr.shape[:2]
    channels = 1 if arr.ndim == 2 else arr.shape[2]
    bit_depth = arr.dtype.itemsize * 8

    return arr, {"channels": channels, "bit_depth": bit_depth,
                 "format": "dcm", "h": H, "w": W}


# ── standard format loader ────────────────────────────────────────────────────

def _load_standard(path: str) -> tuple:
    # cv2.IMREAD_UNCHANGED preserves 16-bit and alpha channels
    arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise IOError(f"cv2.imread could not read: {path}")

    # cv2 returns BGR for colour; convert to RGB
    if arr.ndim == 3:
        if arr.shape[2] == 4:          # BGRA → RGB (drop alpha)
            arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB)
        elif arr.shape[2] == 3:        # BGR → RGB
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)

    arr = np.ascontiguousarray(arr)
    H, W     = arr.shape[:2]
    channels = 1 if arr.ndim == 2 else arr.shape[2]
    bit_depth = arr.dtype.itemsize * 8
    ext       = os.path.splitext(path)[1].lower().lstrip(".")

    return arr, {"channels": channels, "bit_depth": bit_depth,
                 "format": ext, "h": H, "w": W}


# ── internal helpers ──────────────────────────────────────────────────────────

def _to_bgr(array: np.ndarray) -> np.ndarray:
    """Convert RGB/grayscale array to BGR for cv2.imwrite."""
    if array.ndim == 3 and array.shape[2] == 3:
        return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    return array   # grayscale stays as-is


def to_uint8_display(array: np.ndarray) -> np.ndarray:
    """
    Convert any image to uint8 RGB for on-screen display.
    16-bit values are scaled by /256.  Grayscale is stacked to 3 channels.
    """
    if array.dtype == np.uint16:
        display = (array >> 8).astype(np.uint8)
    else:
        display = array.astype(np.uint8)

    if display.ndim == 2:
        display = np.stack([display] * 3, axis=-1)
    return display
