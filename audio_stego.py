"""
audio_stego.py - LSB steganography in speech audio
====================================================
The pipeline passes encrypted image bytes here for hiding in speech audio.
Carrier frames are selected by KMeans on ZCR + log-energy so that
the high-energy (voiced) portion of speech carries the payload.

Supports 1 or 2 LSBs per sample.

Header format (version 2 — magic b"MSG2"):
  Offset  Size  Field
  ------  ----  -----
  0       4     magic: b"MSG2"
  4       1     n_lsbs  (uint8, 1 or 2)
  5       4     payload_length  (uint32 big-endian, bytes)
  Total:  9 bytes

Legacy version 1 (magic b"MSTG") is auto-detected on extract().
"""
import struct  # Packs and unpacks the binary payload header.
import numpy as np  # Splits byte payloads into bits and handles sample arrays.
from sklearn.cluster import KMeans  # Clusters audio frames to choose carriers.

FRAME_SIZE = 256  # Analyze and embed audio in fixed blocks of 256 samples.

# ── header constants ───────────────────────────────────────────────────────────
# v2 (current)
_MAGIC_V2      = b"MSG2"  # Identifies the current payload header format.
_HDR_FMT_V2    = ">4sBI"  # Big-endian magic, LSB count, and payload byte length.
_HDR_BYTES_V2  = struct.calcsize(_HDR_FMT_V2)  # Number of bytes in a v2 header.

# v1 (legacy, read-only backward compat)
_MAGIC_V1      = b"MSTG"  # Identifies payloads written by the legacy format.
_HDR_FMT_V1    = ">4sI"  # Legacy header contains magic and payload byte length.
_HDR_BYTES_V1  = struct.calcsize(_HDR_FMT_V1)  # Number of bytes in a v1 header.


# ── carrier frame selection ────────────────────────────────────────────────────

def _compute_features(audio: np.ndarray) -> np.ndarray:
    n_frames = len(audio) // FRAME_SIZE  # Count complete frames; ignore a partial tail.
    # Trim to complete frames, reshape, and use floats for feature calculations.
    frames = audio[: n_frames * FRAME_SIZE].reshape(n_frames, FRAME_SIZE).astype(np.float64)
    # Compute sign changes between neighboring samples as the frame's ZCR feature.
    zcr = np.sum(np.abs(np.diff(np.sign(frames), axis=1)), axis=1) / (FRAME_SIZE - 1)
    # Log frame energy so large amplitude differences are easier to cluster.
    energy = np.log(np.sum(frames ** 2, axis=1) + 1e-10)
    feats = np.stack([zcr, energy], axis=1)  # Make one [ZCR, energy] row per frame.
    # Min-max scale each feature so ZCR and energy have comparable clustering weight.
    feats = (feats - feats.min(0)) / ((feats.max(0) - feats.min(0)) + 1e-12)
    return feats  # Return the normalized feature matrix to K-means.


def _carrier_mask(audio: np.ndarray, seed: int = 0) -> np.ndarray:
    """Boolean mask (n_frames,) — True for carrier (high-energy) frames."""
    feats = _compute_features(audio)  # Describe each complete frame by ZCR and energy.
    # Split frames into two repeatable groups; multiple initializations improve fitting.
    km = KMeans(n_clusters=2, random_state=seed, n_init=10)
    labels = km.fit_predict(feats)  # Assign each frame to one of the two clusters.
    # Compare average normalized energy in each cluster to find the higher-energy one.
    mean_e = [feats[labels == k, 1].mean() for k in range(2)]
    # Return True only for frames in the high-energy cluster used as carriers.
    return labels == int(np.argmax(mean_e))


# ── public API ─────────────────────────────────────────────────────────────────

def capacity_bits(audio: np.ndarray, n_lsbs: int = 1) -> int:
    """Total embeddable bits; scales linearly with n_lsbs (1 or 2)."""
    # Limit each sample to one or two modified low bits.
    assert n_lsbs in (1, 2), "n_lsbs must be 1 or 2"
    mask = _carrier_mask(audio)  # Identify the complete frames available for data.
    # Each selected frame contributes 256 samples, with n_lsbs bits per sample.
    return int(mask.sum()) * FRAME_SIZE * n_lsbs


def embed(audio: np.ndarray, payload: bytes,
          n_lsbs: int = 1, seed: int = 0) -> np.ndarray:
    """
    Embed *payload* into *audio* using LSB steganography.

    Parameters
    ----------
    audio   : int16 1-D PCM array
    payload : raw bytes to embed
    n_lsbs  : 1 → 1 bit/sample (~90 dB SNR); 2 → 2 bits/sample (~84 dB SNR)
    seed    : KMeans seed for reproducible carrier selection

    Returns
    -------
    stego : int16 1-D array (same length as *audio*)
    """
    assert n_lsbs in (1, 2)  # Reject unsupported bits-per-sample settings.
    # Prefix payload with format marker, selected LSB depth, and byte length.
    header = struct.pack(_HDR_FMT_V2, _MAGIC_V2, n_lsbs, len(payload))
    message = header + payload  # Extraction needs the header before the payload.
    msg_bits = len(message) * 8  # Convert total message size from bytes to bits.

    cap = capacity_bits(audio, n_lsbs)  # Calculate carrier capacity before modifying audio.
    if msg_bits > cap:
        # Fail clearly rather than silently truncating the hidden data.
        raise ValueError(
            f"Payload too large: need {msg_bits:,} bits, carrier has {cap:,} bits. "
            f"Use a longer audio file{', or 2 LSBs' if n_lsbs == 1 else ''}, "
            f"or fewer/smaller images."
        )

    bits = np.unpackbits(np.frombuffer(message, dtype=np.uint8))  # Convert bytes to MSB-first bits.
    n_frames = len(audio) // FRAME_SIZE  # Only complete frames participate in embedding.
    mask = _carrier_mask(audio, seed)  # Recompute the deterministic carrier-frame selection.
    stego = audio.copy()  # Preserve the cover audio and modify only the copy.
    bit_ptr = 0  # Track the next message bit that has not been embedded.

    for f in range(n_frames):
        # Leave non-carrier frames untouched.
        if not mask[f]:
            continue
        start = f * FRAME_SIZE  # Find the first sample in this frame.
        for s in range(FRAME_SIZE):
            # Stop once every header and payload bit has been stored.
            if bit_ptr >= msg_bits:
                break
            sample = int(stego[start + s])  # Convert the PCM value for bitwise operations.
            if n_lsbs == 1:
                # Clear the lowest bit, then write one message bit into it.
                stego[start + s] = np.int16((sample & ~1) | int(bits[bit_ptr]))
                bit_ptr += 1  # One sample carried one message bit.
            else:
                # Read up to two bits; pad the final pair if the message length is odd.
                b0 = int(bits[bit_ptr])     if bit_ptr     < msg_bits else 0
                b1 = int(bits[bit_ptr + 1]) if bit_ptr + 1 < msg_bits else 0
                # Clear the two low bits, then place b0 in bit 1 and b1 in bit 0.
                stego[start + s] = np.int16((sample & ~3) | (b0 << 1) | b1)
                bit_ptr = min(bit_ptr + 2, msg_bits)  # Advance over the pair just embedded.
        # No later carrier frames are needed after the message is complete.
        if bit_ptr >= msg_bits:
            break

    return stego  # Return the modified carrier audio containing the hidden message.


def extract(stego: np.ndarray, seed: int = 0) -> bytes:
    """
    Extract and return the embedded payload.

    Auto-detects 1-LSB vs 2-LSB and v1 (MSTG) vs v2 (MSG2) headers by
    trying both LSB depths in order (1 then 2).
    """
    mask = _carrier_mask(stego, seed)  # Recreate the same carrier frames used by embed().

    for try_lsbs in (1, 2):
        # Read enough bits for a v2 header while trying each supported LSB depth.
        probe_needed = _HDR_BYTES_V2 * 8
        probe_bits = _collect_bits(stego, mask, probe_needed, n_lsbs=try_lsbs)
        # Without a complete header, the audio cannot identify a valid payload.
        if len(probe_bits) < probe_needed:
            raise ValueError("Carrier too short to contain a valid header.")

        probe_bytes = np.packbits(probe_bits[:probe_needed]).tobytes()  # Reassemble header bits as bytes.
        magic = probe_bytes[:4]  # The leading marker identifies the header version.

        if magic == _MAGIC_V2:
            # Parse the v2 marker, LSB depth, and payload length from the probe.
            _, n_lsbs_embed, payload_len = struct.unpack(
                _HDR_FMT_V2, probe_bytes[:_HDR_BYTES_V2]
            )
            # Accept this interpretation only if the stored LSB depth matches the probe.
            if n_lsbs_embed != try_lsbs:
                continue  # A mismatch means the other LSB-depth interpretation may work.
            hdr_bytes = _HDR_BYTES_V2  # Record the header size so payload bytes can be sliced.
            n_lsbs_embed = try_lsbs  # Use the verified LSB depth for full extraction.
        elif magic == _MAGIC_V1 and try_lsbs == 1:
            # Read legacy headers, which supported one LSB and omitted the depth field.
            _, payload_len = struct.unpack(_HDR_FMT_V1, probe_bytes[:_HDR_BYTES_V1])
            n_lsbs_embed = 1  # Legacy payload bits were stored one per sample.
            hdr_bytes = _HDR_BYTES_V1  # Use the shorter legacy header size.
        else:
            continue  # Unknown marker or incompatible depth; try the other interpretation.

        # Collect the complete header and payload using the now-known bit depth.
        total_bits = (hdr_bytes + payload_len) * 8  # Convert expected total bytes to bits.
        all_bits = _collect_bits(stego, mask, total_bits, n_lsbs=n_lsbs_embed)
        # Detect a truncated or damaged carrier before attempting byte conversion.
        if len(all_bits) < total_bits:
            raise ValueError(
                f"Stego audio is too short for the declared payload "
                f"({payload_len} bytes)."
            )
        # Pack the recovered bits back into a byte string.
        all_bytes = np.packbits(
            np.array(all_bits[:total_bits], dtype=np.uint8)
        ).tobytes()
        # Return only payload bytes, omitting the format header.
        return all_bytes[hdr_bytes: hdr_bytes + payload_len]

    # Neither supported LSB interpretation contained a recognized header marker.
    raise ValueError(
        "Unknown or corrupted header magic. "
        "This may not be a CryptoStego stego file."
    )


# ── internal helpers ───────────────────────────────────────────────────────────

def _collect_bits(stego: np.ndarray, mask: np.ndarray,
                  n_bits: int, n_lsbs: int = 1) -> list:
    """
    Collect up to *n_bits* bits from carrier frames.
    For n_lsbs=2 each sample contributes bit1=(s>>1)&1 then bit0=s&1.
    """
    bits = []  # Accumulate recovered message bits in their original order.
    n_frames = len(stego) // FRAME_SIZE  # Ignore any incomplete final audio frame.

    for f in range(n_frames):
        # Skip frames not selected as carriers by the feature clustering.
        if not mask[f]:
            continue
        start = f * FRAME_SIZE  # Compute the first sample index in this frame.
        for s in range(FRAME_SIZE):
            # Return as soon as the requested header or payload bits are collected.
            if len(bits) >= n_bits:
                return bits
            sample = int(stego[start + s])  # Read the sample as an integer for bit masking.
            if n_lsbs == 1:
                bits.append(sample & 1)  # Recover the single lowest-order bit.
            else:
                # Read bit 1 before bit 0 to match the order used by embed().
                bits.append((sample >> 1) & 1)
                bits.append(sample & 1)
        # Stop scanning frames once enough bits have been recovered.
        if len(bits) >= n_bits:
            break

    return bits  # Return fewer bits only when the carrier ended before n_bits.
