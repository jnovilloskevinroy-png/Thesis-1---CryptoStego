"""
pipeline.py - Multi-image cryptostego pipeline
============================================
Encrypts N digital images (any shape/dtype) and embeds the combined
cipher pack into a single cover speech WAV via LSB steganography.

Multi-image payload format (inside the audio carrier):
  Offset  Size  Field
  ------  ----  -----
  0       4     magic  b"MSIG"
  4       2     N      uint16 big-endian — number of images
  --- per image (43 bytes header + data) ---
  0       2     H         uint16 BE
  2       2     W         uint16 BE
  4       1     channels  uint8 (1 = grayscale, 3 = colour)
  5       1     bit_depth uint8 (8 or 16)
  6       1     ndim      uint8 (2 or 3)
  7       32    digest    SHA-256 of plaintext pixels (per-image nonce)
  39      4     nbytes    uint32 BE — byte count of encrypted pixel data
  43      nbytes          raw encrypted pixels (little-endian, C-contiguous)
"""
import os
import struct
import numpy as np
from PIL import Image
from scipy.io import wavfile

import image_crypto
import audio_stego
import metrics


# ── format constants (exported for app layers) ─────────────────────────────────
_MULTI_MAGIC    = b"MSIG"
_MULTI_HDR_FMT  = ">4sH"
_MULTI_HDR_SIZE = struct.calcsize(_MULTI_HDR_FMT)       # 6

_IMG_HDR_FMT    = ">HHBBB"                              # H,W,channels,bit_depth,ndim
_IMG_HDR_SIZE   = struct.calcsize(_IMG_HDR_FMT)         # 7
_DIGEST_SIZE    = 32
_NBYTES_FMT     = ">I"
_NBYTES_SIZE    = 4
# Total per-image overhead (excluding pixel data): 7+32+4 = 43 bytes
_IMG_OVERHEAD   = _IMG_HDR_SIZE + _DIGEST_SIZE + _NBYTES_SIZE


# ── serialization ──────────────────────────────────────────────────────────────

def _serialize_ciphers(pairs: list) -> bytes:
    """Pack list of (cipher ndarray, key_material dict) into one byte buffer."""
    parts = [struct.pack(_MULTI_HDR_FMT, _MULTI_MAGIC, len(pairs))]
    for cipher, km in pairs:
        c   = np.ascontiguousarray(cipher)
        H, W = c.shape[:2]
        ndim      = c.ndim
        channels  = 1 if ndim == 2 else c.shape[2]
        bit_depth = c.dtype.itemsize * 8
        digest    = km["digest"]
        raw       = c.tobytes()
        nbytes    = len(raw)
        img_hdr   = struct.pack(_IMG_HDR_FMT, H, W, channels, bit_depth, ndim)
        parts.append(img_hdr + digest + struct.pack(_NBYTES_FMT, nbytes) + raw)
    return b"".join(parts)


def _deserialize_ciphers(buf: bytes) -> list:
    """Return list of (cipher ndarray, digest bytes)."""
    magic, N = struct.unpack(_MULTI_HDR_FMT, buf[:_MULTI_HDR_SIZE])
    if magic != _MULTI_MAGIC:
        raise ValueError(f"Bad magic: {magic!r}  expected {_MULTI_MAGIC!r}")

    offset  = _MULTI_HDR_SIZE
    results = []
    for _ in range(N):
        H, W, channels, bit_depth, ndim = struct.unpack(
            _IMG_HDR_FMT, buf[offset: offset + _IMG_HDR_SIZE]
        )
        offset += _IMG_HDR_SIZE

        digest  = buf[offset: offset + _DIGEST_SIZE]
        offset += _DIGEST_SIZE

        nbytes  = struct.unpack(_NBYTES_FMT, buf[offset: offset + _NBYTES_SIZE])[0]
        offset += _NBYTES_SIZE

        dtype   = np.uint16 if bit_depth == 16 else np.uint8
        shape   = (H, W) if ndim == 2 else (H, W, channels)
        cipher  = np.frombuffer(buf[offset: offset + nbytes],
                                dtype=dtype).reshape(shape).copy()
        offset += nbytes
        results.append((cipher, digest))
    return results


def _payload_size(images: list) -> int:
    """Exact serialized payload byte count for a list of images (any shape/dtype)."""
    return _MULTI_HDR_SIZE + sum(
        _IMG_OVERHEAD + np.ascontiguousarray(img).nbytes
        for img in images
    )


# ── sender / receiver ──────────────────────────────────────────────────────────

def sender(images: list, audio: np.ndarray, passphrase: str,
           fast: bool = False, n_lsbs: int = 1):
    """
    Encrypt a list of images and embed into *audio*.

    Parameters
    ----------
    images    : list of ndarray — any shape (H,W) or (H,W,C), uint8 or uint16
    audio     : int16 1-D PCM array
    passphrase: encryption key string
    n_lsbs    : 1 or 2 bits per carrier sample

    Returns
    -------
    ciphers  : list of encrypted ndarrays
    stego    : int16 1-D array — carrier with hidden payload
    km_list  : list of key_material dicts (one per image)
    """
    pairs = [image_crypto.encrypt(img, passphrase, fast=fast) for img in images]

    payload = _serialize_ciphers(pairs)
    needed  = len(payload) * 8
    cap     = audio_stego.capacity_bits(audio, n_lsbs)
    if needed > cap:
        raise ValueError(
            f"Combined payload ({needed:,} bits) exceeds carrier capacity "
            f"({cap:,} bits). "
            f"Use longer audio{', or 2 LSBs' if n_lsbs == 1 else ''}, "
            f"or fewer/smaller images."
        )

    stego   = audio_stego.embed(audio, payload, n_lsbs=n_lsbs)
    ciphers = [c for c, _  in pairs]
    km_list = [km for _, km in pairs]
    return ciphers, stego, km_list


def receiver(stego: np.ndarray, passphrase: str,
             fast: bool = False) -> list:
    """
    Extract and decrypt all images from *stego* audio.

    Returns list of recovered ndarrays (same shape/dtype as originals).
    """
    payload = audio_stego.extract(stego)
    pairs   = _deserialize_ciphers(payload)
    results = []
    for cipher, digest in pairs:
        H, W      = cipher.shape[:2]
        channels  = 1 if cipher.ndim == 2 else cipher.shape[2]
        bit_depth = cipher.dtype.itemsize * 8
        salt = f"{H}x{W}x{channels}x{bit_depth}" + digest.hex()
        km   = {"passphrase": passphrase, "salt": salt, "digest": digest}
        results.append(image_crypto.decrypt(cipher, km, fast=fast))
    return results


# ── synthetic helpers (testing / demo only — not on production path) ───────────

def _synth_image(H: int = 64, W: int = 64, seed: int = 42,
                 modality: str = "brain") -> np.ndarray:
    rng   = np.random.default_rng(seed)
    img   = np.zeros((H, W, 3), dtype=np.uint8)
    y_lin = np.linspace(30, 200, H)
    x_lin = np.linspace(60, 180, W)
    base  = ((y_lin[:, None] + x_lin[None, :]) / 2).astype(np.uint8)
    Y, X  = np.ogrid[:H, :W]
    cy, cx = H // 2, W // 2

    if modality == "brain":
        img[:,:,0] = np.clip(base//2 + rng.integers(0,30,(H,W)), 0, 255)
        img[:,:,1] = np.clip(base//2 + rng.integers(0,25,(H,W)), 0, 255)
        img[:,:,2] = np.clip(base//3 + rng.integers(0,20,(H,W)), 0, 255)
        d = np.sqrt((Y-cy)**2 + (X-cx)**2)
        img[d < min(H,W)*0.45] = np.clip(img[d < min(H,W)*0.45].astype(int)+80, 0, 255)
        img[d < min(H,W)*0.15] = np.clip(img[d < min(H,W)*0.15].astype(int)+60, 0, 255)
    elif modality == "xray":
        gray = np.clip(base + rng.integers(0,20,(H,W)), 0, 255).astype(np.uint8)
        img[:,:,0] = img[:,:,1] = img[:,:,2] = gray
        step = max(1, H//6)
        for r0 in range(0, H, step):
            img[r0:r0+max(2,H//20),:,:] = np.clip(
                img[r0:r0+max(2,H//20),:,:].astype(int)+60, 0, 255)
    elif modality == "ct":
        img[:,:,0] = np.clip(base*2//3+rng.integers(0,35,(H,W)), 0, 255)
        img[:,:,1] = img[:,:,0]
        img[:,:,2] = np.clip(base//4+rng.integers(0,25,(H,W)), 0, 255)
        for dy,dx,rad,bright in [(-H//4,-W//4,H//8,70),(H//4,W//4,H//7,50)]:
            m = (Y-cy-dy)**2+(X-cx-dx)**2 < rad**2
            img[m] = np.clip(img[m].astype(int)+bright, 0, 255)
    elif modality == "bone":
        img[:,:,:] = 20
        img[:,:,0] = np.clip(20+rng.integers(0,15,(H,W)), 0, 255)
        for dy,dx in [(-H//4,0),(H//4,0),(0,-W//4),(0,W//4),(0,0)]:
            d = np.sqrt((Y-cy-dy)**2+(X-cx-dx)**2).clip(0, max(3,H//8))
            rad = max(3, H//8)
            hot = (180*(1-d/rad)+40).astype(int)
            m   = d < rad
            img[:,:,0] = np.where(m, np.clip(hot,0,255), img[:,:,0])
            img[:,:,1] = np.where(m, np.clip(hot//2,0,255), img[:,:,1])
            img[:,:,2] = np.where(m, np.clip(hot//4,0,255), img[:,:,2])
    return img.astype(np.uint8)


def _synth_audio(duration_s: float = 60.0, sr: int = 16000, seed: int = 7) -> np.ndarray:
    rng    = np.random.default_rng(seed)
    n      = int(duration_s * sr)
    t      = np.linspace(0, duration_s, n)
    signal = np.zeros(n)
    for i, f in enumerate([120, 240, 360, 500, 800]):
        amp    = 0.3 + 0.1 * i
        onset  = int((i / 5) * n)
        length = n // 6
        signal[onset: onset + length] += amp * np.sin(2 * np.pi * f * t[onset: onset + length])
    signal += 0.02 * rng.standard_normal(n)
    peak    = np.max(np.abs(signal)) + 1e-9
    return (signal / peak * 28000).astype(np.int16)


# ── demo CLI ───────────────────────────────────────────────────────────────────

def run(args):
    os.makedirs(args.outdir, exist_ok=True)

    if args.image:
        try:
            import image_io
            images     = [image_io.load(p)[0] for p in args.image]
            modalities = [os.path.splitext(os.path.basename(p))[0] for p in args.image]
        except Exception as e:
            print(f"Image load error: {e}")
            return
    else:
        modalities = ["brain", "xray", "ct", "bone"]
        images = [_synth_image(64, 64, seed=args.seed+i, modality=m)
                  for i, m in enumerate(modalities)]

    if args.audio:
        sr, audio = wavfile.read(args.audio)
        if audio.ndim > 1:
            audio = audio[:, 0]
        audio = audio.astype(np.int16)
    else:
        sr    = 16000
        audio = _synth_audio(60.0, sr, seed=args.seed)

    passphrase = args.key
    N          = len(images)
    print(f"=== CryptoStego Multi-Image Pipeline ===\n")
    print(f"  {N} image(s), {len(audio)/sr:.1f}s audio")

    for i, (img, m) in enumerate(zip(images, modalities)):
        Image.fromarray(
            img.astype(np.uint8) if img.dtype==np.uint16 else img
        ).save(os.path.join(args.outdir, f"01_original_{i:02d}_{m}.png"))

    ciphers, stego, km_list = sender(images, audio, passphrase, n_lsbs=1)

    for i, (c, m) in enumerate(zip(ciphers, modalities)):
        Image.fromarray(
            c.astype(np.uint8) if c.dtype==np.uint16 else c
        ).save(os.path.join(args.outdir, f"02_encrypted_{i:02d}_{m}.png"))
    wavfile.write(os.path.join(args.outdir, "03_stego_speech.wav"), sr, stego)

    recovered_list = receiver(stego, passphrase)
    for i, (rec, m) in enumerate(zip(recovered_list, modalities)):
        Image.fromarray(
            rec.astype(np.uint8) if rec.dtype==np.uint16 else rec
        ).save(os.path.join(args.outdir, f"04_recovered_{i:02d}_{m}.png"))

    print("\n--- Security Metrics ---")
    for i, (img, cipher, mod) in enumerate(zip(images, ciphers, modalities)):
        n_v = metrics.npcr(img, cipher)
        h_v = metrics.shannon_entropy(cipher)
        cc  = metrics.correlation_coefficients(cipher)
        print(f"  [{i}] {mod}: NPCR={n_v:.2f}% Entropy={h_v:.4f} "
              f"Corr(H/V/D)={cc['horizontal']:.3f}/{cc['vertical']:.3f}/{cc['diagonal']:.3f}")

    print("\n--- Recovery ---")
    for i, (img, rec, mod) in enumerate(zip(images, recovered_list, modalities)):
        m_v = metrics.mse(img, rec)
        print(f"  [{i}] {mod}: MSE={m_v}  {'PASS' if m_v==0 else 'FAIL'}")

    snr = metrics.snr(audio, stego)
    print(f"\n  Stego SNR: {snr:.1f} dB")
    print(f"\nOutputs: {os.path.abspath(args.outdir)}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="CryptoStego multi-image pipeline")
    parser.add_argument("--image",  nargs="+", default=None)
    parser.add_argument("--audio",  default=None)
    parser.add_argument("--key",    default="CryptoStegoKey2024!")
    parser.add_argument("--outdir", default=".")
    parser.add_argument("--seed",   type=int, default=42)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
