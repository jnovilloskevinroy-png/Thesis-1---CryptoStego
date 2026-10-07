# CryptoStego - General-Purpose Secure Image Transmission
## Chaotic Encryption + Speech Steganography

CryptoStego encrypts any digital image (color or grayscale, 8-bit or 16-bit, including DICOM) with a three-map chaotic cipher (confusion + diffusion + confusion) and hides the encrypted payload inside a speech audio file using LSB steganography guided by zero-crossing rate / energy clustering. Multiple images can be packed into a single stego WAV. The model is domain-agnostic — any digital image is protected the same way, since the cipher and embedding act purely on pixel values and sample amplitudes.

---

## Quick Start

```bash
pip install -r requirements.txt   # add: pip install pydicom   (optional, for .dcm support)

streamlit run app.py              # web UI      → http://localhost:8501
python app_gui.py                 # desktop app
python pipeline.py --image img.png --audio speech.wav --key "MySecret"   # CLI
```

---

## Architecture

| Module | Purpose |
|--------|---------|
| `chaotic_maps.py` | Quantum Logistic Map, Zaslavsky Map, Henon 3-D Map — deterministic chaos PRNGs |
| `image_crypto.py` | Per-channel Confusion1 → Diffusion → Confusion2; SHA-512 key derivation; per-image SHA-256 nonce |
| `audio_stego.py`  | 16-bit PCM LSB steganography; carrier frames selected by KMeans on ZCR + log-energy |
| `metrics.py`      | NPCR, UACI, MSE, PSNR, SNR, Shannon entropy, correlation coefficients |
| `pipeline.py`     | Multi-image sender / receiver; CLI with argparse |
| `app.py`          | Streamlit UI — Sender and Receiver tabs |
| `app_gui.py`      | PySide6 desktop application — Protect and Recover tabs |
| `watch_folder.py` | Unattended automation — monitors a directory, auto-encrypts new images, writes a JSON-Lines audit log |
| `analysis.py`     | Histogram, correlation scatter, and key-sensitivity visualisation panels |
| `evaluation.py`   | Full evaluation suite — metrics.csv + REPORT.md |
| `test_fast_path.py` | pytest: vectorized diffusion matches reference loop byte-for-byte |

---

## Usage

- **Streamlit (`app.py`)** — *Sender*: upload images, upload a cover WAV or use built-in synthetic audio, generate or type a key, click Encrypt & embed. *Receiver*: upload the stego WAV, enter the key, click Extract & decrypt.
- **Desktop (`app_gui.py`)** — *Protect*: add images, optionally load your own cover WAV (auto-generated if skipped), pick 1 or 2 LSBs, click Protect Images. *Recover*: open the stego WAV, enter the key.
- **CLI (`pipeline.py`)** — flags: `--image PATH...`, `--audio PATH`, `--key STR`, `--outdir PATH`, `--seed INT`. Omit `--image`/`--audio` to run on synthetic demo data.
- **Automation (`watch_folder.py`)** — `python watch_folder.py --input-dir ... --output-dir ... --audio ... --key ...` (or `--config file.json`); watches a folder and auto-encrypts new images as they arrive.

All interfaces produce bit-exact recoveries (MSE = 0), verified via SHA-256 digest.

---

## Tests

```bash
pytest test_fast_path.py -v
```

---

## Verified Metrics (4 x 64x64 synthetic images, 60 s audio)

| Metric | Result | Target |
|--------|--------|--------|
| NPCR (per image) | ~99.60 % | ~99.6 % |
| Cipher entropy | ~7.95 bits* | >7.99 bits |
| H / V / D correlations | ~0.00 each | ≈ 0 |
| MSE (recovery) | **0.0** | 0 exact |
| Stego SNR | **88.5 dB** | >50 dB |
| Wrong-key MSE | ~7 000–15 000 | non-zero |

*Expected for 64×64 demo images (small histogram); larger real-world images consistently exceed 7.99 bits.

---

## Security Design

| Layer | Mechanism |
|-------|-----------|
| **Key derivation** | SHA-512(passphrase + salt + channel); salt = HxW + SHA-256(plaintext) — unique per image |
| **Confusion 1** | Pixels permuted by argsort of Quantum Logistic Map orbit |
| **Diffusion** | `cipher[i] = ((pixel[i]+ks[i]) & 0xFF) ^ cipher[i-1]`; hybrid keystream from Zaslavsky + Henon maps |
| **Confusion 2** | Second independent permutation scatters diffusion-chain neighbours → near-zero 2-D correlations |
| **Integrity** | SHA-256 digest of plaintext pixels stored per-image in payload; verified at receiver |
| **Stego** | LSB of high-energy speech frames; carrier selection is public but payload is opaque without the key |

DICOM metadata/PHI is never retained — only pixel arrays are processed in memory.

---

## Dependencies

```
numpy  scipy  Pillow  scikit-learn  matplotlib  streamlit  watchdog  pytest
pydicom  (optional, for DICOM files)
```
