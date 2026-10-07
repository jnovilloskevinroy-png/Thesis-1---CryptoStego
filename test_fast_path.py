"""Assert vectorized fast-path matches reference loop byte-for-byte."""
import numpy as np
import pytest
from image_crypto import _diffuse, _undiffuse, _diffuse_fast, _undiffuse_fast
import image_crypto
from pipeline import _synth_image


def _make_inputs(n: int = 1024, seed: int = 0):
    rng   = np.random.default_rng(seed)
    flat  = rng.integers(0, 256, n, dtype=np.uint8)
    ks    = rng.integers(0, 256, n, dtype=np.uint8)
    return flat, ks


def test_diffuse_fast_matches_reference():
    flat, ks = _make_inputs()
    ref  = _diffuse(flat, ks)
    fast = _diffuse_fast(flat, ks)
    assert np.array_equal(ref, fast), "Fast diffuse does NOT match reference!"


def test_undiffuse_fast_matches_reference():
    flat, ks  = _make_inputs()
    cipher    = _diffuse(flat, ks)
    ref_plain = _undiffuse(cipher, ks)
    fast_plain= _undiffuse_fast(cipher, ks)
    assert np.array_equal(ref_plain, fast_plain), "Fast undiffuse does NOT match reference!"


def test_encrypt_decrypt_roundtrip_ref():
    img = _synth_image(64, 64, seed=1)
    cipher, km = image_crypto.encrypt(img, "TestKey!", fast=False)
    recovered  = image_crypto.decrypt(cipher, km, fast=False)
    assert np.array_equal(img, recovered), "Reference round-trip failed!"


def test_encrypt_decrypt_roundtrip_fast():
    img = _synth_image(64, 64, seed=2)
    cipher, km = image_crypto.encrypt(img, "TestKey!", fast=True)
    recovered  = image_crypto.decrypt(cipher, km, fast=True)
    assert np.array_equal(img, recovered), "Fast round-trip failed!"


def test_fast_path_matches_ref_full_image():
    """Cipher produced by fast path must be identical to reference cipher."""
    img = _synth_image(32, 32, seed=3)

    cipher_ref,  km_ref  = image_crypto.encrypt(img, "SameKey!", fast=False)
    cipher_fast, km_fast = image_crypto.encrypt(img, "SameKey!", fast=True)

    assert np.array_equal(cipher_ref, cipher_fast), \
        "Fast-path cipher differs from reference cipher!"


if __name__ == "__main__":
    test_diffuse_fast_matches_reference()
    test_undiffuse_fast_matches_reference()
    test_encrypt_decrypt_roundtrip_ref()
    test_encrypt_decrypt_roundtrip_fast()
    test_fast_path_matches_ref_full_image()
    print("All tests passed.")
