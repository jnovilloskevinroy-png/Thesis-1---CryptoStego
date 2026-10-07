"""
CryptoStego Streamlit Application
================================
Production-grade UI for secure, general-purpose digital image transmission.
All encryption / steganography logic is delegated to the backend modules
(image_crypto, audio_stego, pipeline) — no logic is duplicated here.

Launch:
    streamlit run app.py
"""
import hashlib
import io
import os
import secrets
import sys
import zipfile

import numpy as np
import streamlit as st
from PIL import Image
from scipy.io import wavfile

# ── Backend (same functions the CLI uses) ─────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audio_stego
import metrics
from pipeline import (
    _IMG_HDR_SIZE,
    _MULTI_HDR_SIZE,
    _deserialize_ciphers,
    _synth_audio,
    receiver as _receiver,
    sender as _sender,
)

# ── Optional DICOM ────────────────────────────────────────────────────────────
try:
    import pydicom
    _DICOM_OK = True
except ImportError:
    _DICOM_OK = False


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="CryptoStego - General-Purpose Secure Image Transmission",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
/* tab labels */
.stTabs [data-baseweb="tab-list"] { gap: 32px; }
.stTabs [data-baseweb="tab"]      { font-size: 1.05rem; font-weight: 600; }

/* status boxes */
.mbox { border-radius: 6px; padding: 12px 16px; margin: 8px 0;
        font-family: monospace; font-size: 0.92rem; line-height: 1.5; }
.mok  { background:#d4edda; color:#155724; border-left:4px solid #28a745; }
.merr { background:#f8d7da; color:#721c24; border-left:4px solid #dc3545; }
.minf { background:#d1ecf1; color:#0c5460; border-left:4px solid #17a2b8; }
.mwrn { background:#fff3cd; color:#856404; border-left:4px solid #ffc107; }
</style>
""", unsafe_allow_html=True)


def _box(css, html):
    st.markdown(f'<div class="mbox {css}">{html}</div>', unsafe_allow_html=True)

def ok_box(m):  _box("mok",  m)
def err_box(m): _box("merr", m)
def inf_box(m): _box("minf", m)
def wrn_box(m): _box("mwrn", m)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS  (pure functions — no Streamlit calls)
# ═══════════════════════════════════════════════════════════════════════════════

def _load_image_src(src) -> tuple:
    """
    Load an UploadedFile or a file-system path as a uint8 RGB ndarray.
    Returns (filename: str, image: ndarray).
    No metadata or PHI is retained — only pixel data.
    """
    if isinstance(src, str):
        name = os.path.basename(src)
        with open(src, "rb") as f:
            data = f.read()
    else:
        name = src.name
        src.seek(0)
        data = src.read()

    ext = os.path.splitext(name)[1].lower()

    if ext == ".dcm":
        if not _DICOM_OK:
            raise RuntimeError(
                "DICOM support requires pydicom. "
                "Install it with: pip install pydicom"
            )
        ds     = pydicom.dcmread(io.BytesIO(data))
        px     = ds.pixel_array.astype(np.float32)
        lo, hi = float(px.min()), float(px.max())
        px     = ((px - lo) / max(hi - lo, 1e-9) * 255).astype(np.uint8)
        if px.ndim == 2:
            px = np.stack([px] * 3, axis=-1)
        elif px.ndim == 3 and px.shape[-1] == 1:
            px = np.concatenate([px] * 3, axis=-1)
        return name, px

    img = Image.open(io.BytesIO(data)).convert("RGB")
    return name, np.array(img)


def _load_wav(src) -> tuple:
    """Load WAV from UploadedFile or path. Returns (sample_rate, int16 1-D array)."""
    if isinstance(src, str):
        sr, audio = wavfile.read(src)
    else:
        src.seek(0)
        sr, audio = wavfile.read(io.BytesIO(src.read()))
    if audio.ndim > 1:
        audio = audio[:, 0]
    return sr, audio.astype(np.int16)


def _to_wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    wavfile.write(buf, sr, audio)
    return buf.getvalue()


def _to_png_bytes(img: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(img.astype(np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def _make_zip(images: list, names: list) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for img, name in zip(images, names):
            stem = os.path.splitext(name)[0]
            zf.writestr(f"{stem}.png", _to_png_bytes(img))
    return buf.getvalue()


def _verify(recovered: np.ndarray, digest: bytes) -> bool:
    """Check SHA-256 of recovered pixel data against the stored nonce digest."""
    return hashlib.sha256(recovered.tobytes()).digest() == digest


def _payload_bytes(images: list) -> int:
    return _MULTI_HDR_SIZE + sum(
        _IMG_HDR_SIZE + 32 + img.shape[0] * img.shape[1] * 3
        for img in images
    )


# ═══════════════════════════════════════════════════════════════════════════════
# APP HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.title("CryptoStego - General-Purpose Secure Image Transmission")
st.caption(
    "Chaotic three-map encryption (Quantum Logistic + Zaslavsky + Henon) "
    "combined with speech-signal LSB steganography. "
    "No metadata or PHI is written to disk — only pixel data is processed."
)

tab_send, tab_recv = st.tabs([
    "  Sender  (Encrypt & Embed)  ",
    "  Receiver  (Extract & Decrypt)  ",
])


# ═══════════════════════════════════════════════════════════════════════════════
# SENDER TAB
# ═══════════════════════════════════════════════════════════════════════════════
with tab_send:
    st.header("Sender Mode")
    st.write(
        "Encrypt one or more patient images with the chaotic cipher and "
        "hide the entire cipher pack inside a single speech audio file. "
        "The receiver needs only the stego WAV and the passphrase."
    )

    # ── Step 1 — Images ──────────────────────────────────────────────────────
    with st.container(border=True):
        st.subheader("Step 1 - Patient Images")
        col_up, col_fld = st.columns([3, 2])
        with col_up:
            ufiles = st.file_uploader(
                "Upload images (PNG / JPG / DICOM .dcm) — multiple files allowed",
                type=["png", "jpg", "jpeg", "dcm"],
                accept_multiple_files=True,
                key="s_files",
            )
        with col_fld:
            st.write("Or load from a server folder path:")
            fld = st.text_input(
                "Folder path",
                placeholder="/data/patient_scans",
                key="s_folder",
            )
            if st.button("Load folder", key="btn_fld"):
                exts = {".png", ".jpg", ".jpeg", ".dcm"}
                try:
                    found = sorted([
                        os.path.join(fld, f)
                        for f in os.listdir(fld)
                        if os.path.splitext(f)[1].lower() in exts
                    ])
                    if found:
                        st.session_state["s_folder_paths"] = found
                        st.success(f"Found {len(found)} image(s).")
                    else:
                        st.warning("No supported images found in that folder.")
                except Exception as exc:
                    st.error(f"Cannot read folder: {exc}")

    # ── Step 2 — Cover Audio ─────────────────────────────────────────────────
    with st.container(border=True):
        st.subheader("Step 2 - Cover Speech Audio")
        col_aw, col_syn = st.columns([3, 2])
        with col_aw:
            audio_up = st.file_uploader(
                "Upload a cover WAV (mono 16-bit PCM preferred; stereo is auto-mixed to mono)",
                type=["wav"],
                key="s_audio",
            )
        with col_syn:
            use_syn = st.checkbox(
                "Use synthetic speech (demo / testing only)",
                value=True,
                key="s_use_syn",
            )
            syn_dur = st.slider(
                "Synthetic duration (seconds)",
                min_value=10, max_value=300, value=60,
                key="s_dur",
                disabled=not st.session_state.get("s_use_syn", True),
            )
            if use_syn:
                st.caption(
                    "Synthetic speech is suitable for demos. "
                    "For clinical use, provide a real cover audio file."
                )

    # ── Step 3 — Key ─────────────────────────────────────────────────────────
    with st.container(border=True):
        st.subheader("Step 3 - Encryption Key")

        col_k1, col_k2 = st.columns([4, 1])
        with col_k2:
            if st.button(
                "Generate secure key",
                key="btn_gk",
                use_container_width=True,
                help="Creates a 128-bit cryptographically random passphrase.",
            ):
                st.session_state["s_key"] = secrets.token_hex(16)

        with col_k1:
            passphrase = st.text_input(
                "Passphrase",
                key="s_key",
                help=(
                    "This key is required to decrypt the images. "
                    "Share it securely with the receiver (encrypted email, "
                    "hospital key-management system, or in-person). "
                    "NEVER transmit it alongside the stego file."
                ),
            )

        if passphrase:
            st.info(
                "Keep this key secret. The receiver must enter it exactly. "
                "It is NOT stored in the stego file.",
                icon=None,
            )

    # ── Encrypt & Embed ───────────────────────────────────────────────────────
    enc_btn = st.button(
        "Encrypt & embed",
        type="primary",
        key="btn_enc",
        use_container_width=True,
    )

    if enc_btn:
        sources = list(ufiles or []) + st.session_state.get("s_folder_paths", [])

        # --- input validation ---
        if not sources:
            err_box("No images selected. Please upload files or load a folder.")
            st.stop()
        if not passphrase.strip():
            err_box("Please enter or generate a passphrase before encrypting.")
            st.stop()

        with st.spinner("Loading, encrypting and embedding — please wait..."):

            # Load images
            images, names, skipped = [], [], []
            for src in sources:
                try:
                    n, img = _load_image_src(src)
                    images.append(img)
                    names.append(n)
                except Exception as exc:
                    skipped.append((getattr(src, "name", str(src)), str(exc)))

            for sname, reason in skipped:
                wrn_box(f"Skipped <b>{sname}</b>: {reason}")

            if not images:
                err_box("No valid images could be loaded. Check file formats.")
                st.stop()

            # Load audio
            try:
                if audio_up and not use_syn:
                    sr, audio = _load_wav(audio_up)
                else:
                    sr    = 16000
                    audio = _synth_audio(float(syn_dur), sr)
            except Exception as exc:
                err_box(f"Could not load audio: {exc}")
                st.stop()

            # Capacity pre-check
            payload_b = _payload_bytes(images)
            cap_bits  = audio_stego.capacity_bits(audio)
            needed_b  = payload_b * 8
            if needed_b > cap_bits:
                min_sec = needed_b / (cap_bits / (len(audio) / sr))
                err_box(
                    f"Audio too short: payload needs <b>{needed_b:,}</b> bits "
                    f"but the carrier holds only <b>{cap_bits:,}</b> bits.<br>"
                    f"Minimum audio duration for this image set: "
                    f"<b>{min_sec:.0f} s</b>. "
                    f"Use a longer cover file or reduce the number / size of images."
                )
                st.stop()

            # Run backend sender
            try:
                ciphers, stego, km_list = _sender(images, audio, passphrase.strip())
            except Exception as exc:
                err_box(f"Encryption error: {exc}")
                st.stop()

            snr_db  = metrics.snr(audio, stego)
            cap_pct = needed_b / cap_bits * 100

            st.session_state["s_result"] = {
                "images":    images,
                "names":     names,
                "ciphers":   ciphers,
                "stego_wav": _to_wav_bytes(stego, sr),
                "snr":       snr_db,
                "cap_pct":   cap_pct,
                "n":         len(images),
            }

    # ── Sender results ────────────────────────────────────────────────────────
    if "s_result" in st.session_state:
        res = st.session_state["s_result"]
        st.divider()
        st.subheader("Transmission Package Ready")

        ok_box(
            f"SUCCESS - {res['n']} image(s) encrypted and embedded. "
            f"Stego WAV is ready for transmission."
        )

        # Summary metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Images embedded", res["n"])
        m2.metric("Stego SNR", f"{res['snr']:.1f} dB",
                  help="Signal-to-noise ratio of the stego audio vs original. >50 dB is inaudible.")
        m3.metric("Carrier capacity used", f"{res['cap_pct']:.1f} %")

        # Download + player
        dl_col, pl_col = st.columns([1, 2])
        with dl_col:
            st.download_button(
                "Download stego WAV",
                data=res["stego_wav"],
                file_name="cryptostego_output.wav",
                mime="audio/wav",
                key="dl_stego",
                use_container_width=True,
            )
        with pl_col:
            st.audio(res["stego_wav"], format="audio/wav")

        # Per-image detail
        st.subheader("Per-image Encryption Report")
        for i, (img, cipher, name) in enumerate(
            zip(res["images"], res["ciphers"], res["names"])
        ):
            with st.expander(f"Image {i:02d} — {name}", expanded=(i == 0)):
                c_orig, c_ciph, c_met = st.columns([2, 2, 3])
                with c_orig:
                    st.image(img, caption="Original (plaintext)", use_container_width=True)
                with c_ciph:
                    st.image(cipher, caption="Cipher (noise)", use_container_width=True)
                with c_met:
                    nv = metrics.npcr(img, cipher)
                    uv = metrics.uaci(img, cipher)
                    hv = metrics.shannon_entropy(cipher)
                    cc = metrics.correlation_coefficients(cipher)
                    st.markdown(
                        f"| Metric | Value |\n|--------|-------|\n"
                        f"| NPCR | {nv:.4f} % |\n"
                        f"| UACI | {uv:.4f} % |\n"
                        f"| Entropy | {hv:.4f} bits |\n"
                        f"| Corr H | {cc['horizontal']:.4f} |\n"
                        f"| Corr V | {cc['vertical']:.4f} |\n"
                        f"| Corr D | {cc['diagonal']:.4f} |"
                    )
                    st.caption(
                        "NPCR ~99.6% and near-zero correlations confirm strong encryption. "
                        "UACI varies by image intensity distribution."
                    )


# ═══════════════════════════════════════════════════════════════════════════════
# RECEIVER TAB
# ═══════════════════════════════════════════════════════════════════════════════
with tab_recv:
    st.header("Receiver Mode")
    st.write(
        "Upload the stego WAV you received and enter the passphrase "
        "to extract and verify every embedded image."
    )

    # ── Step 1 — Stego WAV ───────────────────────────────────────────────────
    with st.container(border=True):
        st.subheader("Step 1 - Stego WAV")
        stego_up = st.file_uploader(
            "Upload the stego WAV received from sender",
            type=["wav"],
            key="r_stego",
        )

    # ── Step 2 — Key ─────────────────────────────────────────────────────────
    with st.container(border=True):
        st.subheader("Step 2 - Decryption Key")
        recv_key = st.text_input(
            "Passphrase (must match exactly what the sender used)",
            type="password",
            key="r_key",
        )

    # ── Extract & Decrypt ─────────────────────────────────────────────────────
    dec_btn = st.button(
        "Extract & decrypt",
        type="primary",
        key="btn_dec",
        use_container_width=True,
    )

    if dec_btn:
        if not stego_up:
            err_box("Please upload the stego WAV file.")
            st.stop()
        if not recv_key.strip():
            err_box("Please enter the decryption key.")
            st.stop()

        with st.spinner("Extracting and decrypting — please wait..."):

            # Load stego audio
            try:
                sr, stego = _load_wav(stego_up)
            except Exception as exc:
                err_box(f"Could not read WAV file: {exc}")
                st.stop()

            # Extract raw payload and parse image headers
            try:
                payload = audio_stego.extract(stego)
                pairs   = _deserialize_ciphers(payload)
            except Exception as exc:
                err_str = str(exc).lower()
                if "magic" in err_str:
                    err_box(
                        "Payload not found — this file may not be a CryptoStego stego WAV, "
                        "or it may be corrupted."
                    )
                else:
                    err_box(f"Extraction failed: {exc}")
                st.stop()

            # Decrypt
            try:
                recovered = _receiver(stego, recv_key.strip())
            except Exception as exc:
                err_box(f"Decryption error: {exc}")
                st.stop()

            # Integrity: SHA-256 of recovered pixels vs stored digest
            integrity = [
                _verify(rec, digest)
                for rec, (_, digest) in zip(recovered, pairs)
            ]

            st.session_state["r_result"] = {
                "recovered":  recovered,
                "integrity":  integrity,
                "n":          len(recovered),
            }

    # ── Receiver results ──────────────────────────────────────────────────────
    if "r_result" in st.session_state:
        res = st.session_state["r_result"]
        st.divider()
        st.subheader("Recovered Images")

        all_ok  = all(res["integrity"])
        n_fail  = sum(not ok for ok in res["integrity"])

        if all_ok:
            ok_box(
                f"All {res['n']} image(s) recovered successfully. "
                f"SHA-256 integrity verified for every image."
            )
        else:
            err_box(
                f"{n_fail} of {res['n']} image(s) failed the SHA-256 integrity check. "
                "The passphrase may be wrong or the stego file may be corrupted."
            )

        imgs_out, names_out = [], []
        for i, (img, ok) in enumerate(zip(res["recovered"], res["integrity"])):
            with st.expander(f"Image {i:02d}", expanded=True):
                c_img, c_info = st.columns([2, 3])
                with c_img:
                    st.image(img, caption=f"Recovered image {i:02d}", use_container_width=True)
                with c_info:
                    if ok:
                        ok_box(
                            f"Image {i:02d} &nbsp;&nbsp; "
                            f"MSE = 0 &nbsp;|&nbsp; SHA-256 integrity PASS"
                        )
                    else:
                        err_box(
                            f"Image {i:02d} &nbsp;&nbsp; integrity FAIL "
                            f"— wrong key or corrupted data"
                        )
                    fname = f"recovered_{i:02d}.png"
                    st.download_button(
                        f"Download {fname}",
                        data=_to_png_bytes(img),
                        file_name=fname,
                        mime="image/png",
                        key=f"dl_r_{i}",
                    )
            imgs_out.append(img)
            names_out.append(f"recovered_{i:02d}.png")

        if imgs_out:
            st.divider()
            st.download_button(
                "Download all images as ZIP",
                data=_make_zip(imgs_out, names_out),
                file_name="cryptostego_recovered.zip",
                mime="application/zip",
                key="dl_zip",
            )
