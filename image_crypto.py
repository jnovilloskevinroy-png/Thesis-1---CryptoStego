"""
image_crypto.py - Chaotic image encryption
===========================================
This module encrypts each image channel independently using chaotic maps.
It supports arrays shaped (H,W) or (H,W,C), with uint8 or uint16 pixels.

Architecture per channel: Confusion-1 → Diffusion → Confusion-2
  Confusion : argsort-based pixel permutation from Quantum Logistic Map
  Diffusion : chained additive-XOR cipher with Zaslavsky+Henon keystream

Key-parameter derivation: SHA-512(passphrase + salt + channel_index).
The salt includes H, W, channel count, bit depth, and SHA-256(plaintext),
so parameters depend on the image and channel; identical inputs are deterministic.
"""
import hashlib  # Supplies SHA-512 parameter derivation and the image SHA-256 digest.
import numpy as np  # Handles image arrays, channel operations, and byte views.
from chaotic_maps import (  # Chaotic generators used for permutation and diffusion.
        QuantumLogisticMap,
        ZaslavskyMap,
        Henon3DMap,
        to_keystream_bytes,
)


# ── key derivation ─────────────────────────────────────────────────────────────

def _derive_params(passphrase: str, salt: str, channel: int):
    # Hash the key, image-specific salt, and channel into deterministic map settings.
    digest = hashlib.sha512((passphrase + salt + str(channel)).encode()).digest()

    def b01(i):
        # Read two digest bytes as a fraction, avoiding exact parameter endpoints.
        v = (digest[2 * i] * 256 + digest[2 * i + 1]) / 65535.0
        return max(min(v, 1.0 - 1e-9), 1e-9)

    # Scale digest fractions into each map parameter's intended range.
    return (
        0.1  + 0.8  * b01(0),   # x0_qlm
        3.74 + 0.26 * b01(1),   # r_qlm  in [3.74, 4.0]
        6.0  + 4.0  * b01(2),   # beta   in [6, 10]
        0.05 + 0.9  * b01(3),   # x0_zas
        0.05 + 0.9  * b01(4),   # y0_zas
        -1.0 + 2.0  * b01(5),   # x0_hen
        -1.0 + 2.0  * b01(6),   # y0_hen
        -1.0 + 2.0  * b01(7),   # z0_hen
    )


def _build_maps(passphrase: str, salt: str, channel: int):
    p = _derive_params(passphrase, salt, channel)  # Derive repeatable settings for this channel.
    # Build the pixel permutation map and the two maps whose sequences form the keystream.
    return (
        QuantumLogisticMap(x0=p[0], r=p[1], beta=p[2]),
        ZaslavskyMap(x0=p[3], y0=p[4]),
        Henon3DMap(x0=p[5], y0=p[6], z0=p[7]),
    )


def _build_post_perm(passphrase: str, salt: str, channel: int) -> QuantumLogisticMap:
    # Prefix salt to derive a separate QLM sequence for the second permutation.
    p = _derive_params(passphrase, "P2_" + salt, channel)
    # Return an independent map so the two confusion passes use different settings.
    return QuantumLogisticMap(x0=p[0], r=p[1], beta=p[2])


# ── diffusion (reference loops, kept for test compatibility) ───────────────────

def _diffuse(flat: np.ndarray, ks: np.ndarray) -> np.ndarray:
    """cipher[i] = ((pixel[i]+ks[i]) & 0xFF) ^ cipher[i-1]"""
    n = len(flat)  # Number of input bytes; the keystream has the same length.
    cipher = np.empty(n, dtype=np.uint8)  # Allocate the encrypted byte sequence.
    prev = 0  # The first byte chains from zero; later bytes chain from prior ciphertext.
    for i in range(n):
        # Add the keystream modulo 256, then XOR with the previous cipher byte.
        cipher[i] = ((int(flat[i]) + int(ks[i])) & 0xFF) ^ prev
        prev = int(cipher[i])  # Carry this result into the next diffusion step.
    return cipher  # Return the chained, diffused bytes.


def _undiffuse(cipher: np.ndarray, ks: np.ndarray) -> np.ndarray:
    """Exact inverse of _diffuse."""
    n = len(cipher)  # Number of encrypted bytes to recover.
    plain = np.empty(n, dtype=np.uint8)  # Allocate the recovered byte sequence.
    prev = 0  # Mirrors the initial chain value used by _diffuse().
    for i in range(n):
        # Undo the previous-cipher XOR, subtract the keystream modulo 256.
        plain[i] = (int(cipher[i]) ^ prev) - int(ks[i]) & 0xFF
        prev = int(cipher[i])  # The inverse needs the preceding ciphertext byte.
    return plain  # Return bytes restored to their pre-diffusion values.


# Keep these equivalent implementations available for compatibility and testing.
def _diffuse_fast(flat: np.ndarray, ks: np.ndarray) -> np.ndarray:
    # Widen before adding so NumPy arithmetic can safely apply the modulo mask.
    added = ((flat.astype(np.int32) + ks.astype(np.int32)) & 0xFF).astype(np.uint8)
    cipher = np.empty_like(added)  # Allocate output with the same byte shape and type.
    prev = np.uint8(0)  # Initialize the ciphertext chaining value.
    for i in range(len(added)):
        cipher[i] = added[i] ^ prev  # Apply the same chained XOR as the reference loop.
        prev = cipher[i]  # Carry the current ciphertext byte forward.
    return cipher  # Return output expected to match _diffuse() exactly.


def _undiffuse_fast(cipher: np.ndarray, ks: np.ndarray) -> np.ndarray:
    plain = np.empty_like(cipher)  # Allocate a byte array for recovered plaintext.
    prev = np.uint8(0)  # Start with the same chain value as _undiffuse().
    for i in range(len(cipher)):
        # Reverse chained XOR and keystream addition using modulo-256 subtraction.
        plain[i] = (int(cipher[i] ^ prev) - int(ks[i])) & 0xFF
        prev = cipher[i]  # Keep the current cipher byte for the next inverse step.
    return plain  # Return the recovered bytes.


# ── per-channel encrypt / decrypt (any dtype) ──────────────────────────────────

def _encrypt_channel(ch: np.ndarray, passphrase: str, salt: str, chan_idx: int) -> np.ndarray:
    """
    Encrypt a 2-D channel array of dtype uint8 or uint16.

    The permutation operates at pixel granularity; the diffusion at byte
    granularity (each uint16 pixel contributes 2 bytes to the keystream).
    """
    ch = np.ascontiguousarray(ch)  # Ensure predictable memory layout for reshape and byte views.
    H, W = ch.shape  # Record channel height and width for the output shape.
    N = H * W  # Count pixels; permutations operate on pixels, not individual bytes.
    dtype = ch.dtype  # Preserve whether pixels are 8-bit or 16-bit.

    # Recreate this channel's deterministic map sequences from the key and salt.
    qlm, zas, hen = _build_maps(passphrase, salt, chan_idx)
    qlm2 = _build_post_perm(passphrase, salt, chan_idx)  # Separate map for the second permutation.

    # CONFUSION 1: reorder pixels using indices ranked by the QLM sequence.
    perm1 = qlm.permutation(N)
    flat = ch.ravel()[perm1]  # Flatten the channel and apply the first permutation.

    # DIFFUSION: reinterpret pixels as bytes; uint16 pixels contribute two bytes each.
    flat_bytes = np.ascontiguousarray(flat).view(np.uint8)
    n_bytes = len(flat_bytes)  # Generate one keystream byte for every data byte.
    # Add the map sequences modulo 1, then convert their combined values to bytes.
    ks = to_keystream_bytes(((zas.sequence(n_bytes) + hen.sequence(n_bytes)) % 1.0))
    diffed_bytes = _diffuse(flat_bytes, ks)  # Mix each data byte with the chained keystream.

    # CONFUSION 2: reinterpret diffused bytes as pixels and apply an independent permutation.
    perm2 = qlm2.permutation(N)
    diffed_pixels = diffed_bytes.view(dtype)  # Restore the original pixel dtype for reordering.
    cipher_flat = np.ascontiguousarray(diffed_pixels)[perm2]  # Shuffle pixels a second time.

    return cipher_flat.reshape(H, W)  # Restore the original 2-D channel dimensions.


def _decrypt_channel(ch: np.ndarray, passphrase: str, salt: str, chan_idx: int) -> np.ndarray:
    ch = np.ascontiguousarray(ch)  # Ensure the cipher channel has a contiguous memory layout.
    H, W = ch.shape  # Save dimensions so the recovered channel can be reshaped.
    N = H * W  # Number of pixels in this channel.
    dtype = ch.dtype  # Keep the original 8-bit or 16-bit pixel representation.

    # Recreate exactly the same deterministic maps used during encryption.
    qlm, zas, hen = _build_maps(passphrase, salt, chan_idx)
    qlm2 = _build_post_perm(passphrase, salt, chan_idx)  # Recreate the second-pass permutation map.

    # INVERSE CONFUSION 2: construct the inverse index map for the second permutation.
    perm2 = qlm2.permutation(N)
    inv_p2 = np.empty(N, dtype=np.int64)  # Allocate indices mapping output positions back to input.
    inv_p2[perm2] = np.arange(N)  # Reverse the permutation by assigning each original index its location.
    diffed_pixels = ch.ravel()[inv_p2]  # Restore the pre-permutation pixel order.
    diffed_bytes = np.ascontiguousarray(diffed_pixels).view(np.uint8)  # Return to byte-level data.

    # INVERSE DIFFUSION: reproduce the same keystream used during encryption.
    n_bytes = len(diffed_bytes)  # Keystream length must equal the number of cipher bytes.
    ks = to_keystream_bytes(((zas.sequence(n_bytes) + hen.sequence(n_bytes)) % 1.0))
    perm1_bytes = _undiffuse(diffed_bytes, ks)  # Remove the chained diffusion layer.

    # INVERSE CONFUSION 1: rebuild the inverse map for the first pixel permutation.
    perm1 = qlm.permutation(N)
    inv_p1 = np.empty(N, dtype=np.int64)  # Allocate indices for undoing the first permutation.
    inv_p1[perm1] = np.arange(N)  # Map each permuted location back to its original index.
    plain_pixels = perm1_bytes.view(dtype)  # Reinterpret recovered bytes as original pixel values.
    plain_flat = np.ascontiguousarray(plain_pixels)[inv_p1]  # Restore original pixel positions.

    return plain_flat.reshape(H, W)  # Reconstruct and return the original channel shape.


# ── public API ─────────────────────────────────────────────────────────────────

def encrypt(image: np.ndarray, passphrase: str, fast: bool = False):
    """
    Encrypt any image of shape (H,W) or (H,W,C), dtype uint8 or uint16.

    Returns
    -------
    cipher       : ndarray  same shape and dtype as *image*
    key_material : dict     {"passphrase", "salt", "digest"}
    """
    assert image.ndim in (2, 3), "Expected 2-D or 3-D array"  # Accept grayscale or channel-last images.
    image = np.ascontiguousarray(image)  # Normalize memory layout before hashing and processing.
    H, W = image.shape[:2]  # Read spatial dimensions shared by all channels.
    channels = 1 if image.ndim == 2 else image.shape[2]  # Determine grayscale versus channel count.
    bit_depth = image.dtype.itemsize * 8  # Record bits per pixel value (8 or 16).

    digest = hashlib.sha256(image.tobytes()).digest()  # Hash pixels to make map settings image-dependent.
    salt = f"{H}x{W}x{channels}x{bit_depth}" + digest.hex()  # Combine image metadata and digest deterministically.

    if image.ndim == 2:
        cipher = _encrypt_channel(image, passphrase, salt, 0)  # Encrypt grayscale as channel zero.
    else:
        # Encrypt each color channel with its own channel-index-derived map parameters.
        cipher = np.stack([
            _encrypt_channel(image[:, :, c], passphrase, salt, c)
            for c in range(channels)
        ], axis=2)

    # Return cipher pixels plus deterministic material needed to regenerate the maps.
    return cipher, {"passphrase": passphrase, "salt": salt, "digest": digest}


def decrypt(cipher: np.ndarray, key_material: dict, fast: bool = False) -> np.ndarray:
    """Decrypt a cipher image. Returns the original array."""
    passphrase = key_material["passphrase"]  # Recover the key input used to derive map settings.
    salt = key_material["salt"]  # Recover the image-specific salt used during encryption.

    if cipher.ndim == 2:
        return _decrypt_channel(cipher, passphrase, salt, 0)  # Reverse the sole grayscale channel.

    # Reverse each color channel with the same channel index used during encryption.
    return np.stack([
        _decrypt_channel(cipher[:, :, c], passphrase, salt, c)
        for c in range(cipher.shape[2])
    ], axis=2)
