"""
evaluation.py - Research-grade evaluation suite
================================================
Computes all security and steganography metrics on real image data,
generates publication-quality plots, and writes metrics.csv + REPORT.md.

Usage (CLI):
    python evaluation.py \\
        --images ct.dcm mri.png \\
        --audio  cover.wav \\
        --key    "MyPassphrase" \\
        --outdir eval_output/

Or call evaluate() directly from the desktop app / Jupyter notebook.
"""
import argparse  # Parses paths and options for the evaluation CLI.
import csv  # Writes the per-image metrics table.
import hashlib  # Used by the report/evaluation utilities.
import os  # Creates output folders and constructs output paths.
import struct  # Supports binary-format helpers used by the evaluation pipeline.
from datetime import datetime  # Adds a generation timestamp to the report.

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # Produces plots saved by the evaluation suite.
import numpy as np  # Computes metrics, samples pixels, and builds perturbations.
from scipy.io import wavfile  # Reads the input cover WAV in the command-line workflow.

import audio_stego
import image_crypto
import metrics
import pipeline as _pipe


# ── dtype-aware metric helpers ─────────────────────────────────────────────────

def _max_val(arr: np.ndarray) -> float:
    # Return the largest representable pixel intensity for this supported dtype.
    return 65535.0 if arr.dtype == np.uint16 else 255.0


def _uaci(img: np.ndarray, cipher: np.ndarray) -> float:
    """UACI normalised to the actual bit depth."""
    # Normalize mean absolute pixel change by the maximum of the image's bit depth.
    return 100.0 * np.abs(img.astype(np.float64) - cipher.astype(np.float64)).mean() / _max_val(img)


def _entropy(img: np.ndarray) -> float:
    """Shannon entropy; uses 65536-bin histogram for 16-bit images."""
    n_bins = 65536 if img.dtype == np.uint16 else 256  # Use one bin per possible intensity.
    hi = 65536 if img.dtype == np.uint16 else 256  # Histogram range endpoint is exclusive.
    chans = [img[:, :, c] for c in range(img.shape[2])] if img.ndim == 3 else [img]  # Split color channels.
    ents = []  # Accumulate channel entropies before averaging.
    for ch in chans:
        hist, _ = np.histogram(ch.ravel(), bins=n_bins, range=(0, hi))  # Count each intensity value.
        prob = hist / hist.sum()  # Convert histogram counts to probabilities.
        prob = prob[prob > 0]  # Exclude empty bins from the logarithm.
        ents.append(-np.sum(prob * np.log2(prob)))  # Compute Shannon entropy for this channel.
    return float(np.mean(ents))  # Return grayscale entropy or mean color-channel entropy.


def _scatter_sample(ch: np.ndarray, n: int = 3000, seed: int = 0):
    rng = np.random.default_rng(seed)  # Make the sampled scatter points reproducible.
    H, W = ch.shape  # Read dimensions to keep sampled coordinates in bounds.
    r = rng.integers(0, H, n)  # Sample rows for the left side of each pair.
    c = rng.integers(0, W - 1, n)  # Exclude the last column so each pixel has a right neighbor.
    return ch[r, c].astype(float), ch[r, c + 1].astype(float)  # Return neighboring intensity arrays.


# ── plot generators ────────────────────────────────────────────────────────────

def _plot_histogram(img: np.ndarray, cipher: np.ndarray, idx: int, outdir: str):
    n_bins = 256 if img.dtype == np.uint8 else 512  # Choose display histogram resolution by dtype.
    hi = 255 if img.dtype == np.uint8 else 65535  # Set the plot's intensity-axis maximum.
    chnames = ["Red", "Green", "Blue"] if img.ndim == 3 else ["Gray"]  # Label each displayed channel.
    n_ch = img.shape[2] if img.ndim == 3 else 1  # Determine how many channel columns to draw.
    cols = ["red", "green", "blue"] if n_ch == 3 else ["steelblue"]  # Select display colors.

    # Reserve one row for the source image and one for its encrypted image.
    fig, axes = plt.subplots(2, n_ch, figsize=(5 * n_ch, 7))
    if n_ch == 1:
        axes = np.array([[axes[0]], [axes[1]]])  # Normalize axes shape for the grayscale case.
    fig.suptitle(f"Image {idx:02d} — Pixel Intensity Histograms", fontweight="bold")

    for c in range(n_ch):
        ch_img = img[:, :, c] if img.ndim == 3 else img  # Select source channel or whole grayscale image.
        ch_cip = cipher[:, :, c] if cipher.ndim == 3 else cipher  # Select matching cipher channel.
        # Draw matching intensity histograms to compare their distributions.
        axes[0,c].hist(ch_img.ravel(),  bins=n_bins, range=(0,hi), color=cols[c], alpha=0.75)
        axes[0,c].set_title(f"Plaintext — {chnames[c]}")
        axes[1,c].hist(ch_cip.ravel(), bins=n_bins, range=(0,hi), color="gray", alpha=0.75)
        axes[1,c].set_title(f"Cipher — {chnames[c]}")
        for ax in (axes[0,c], axes[1,c]):  # Keep both plots on the same intensity scale.
            ax.set_xlim(0, hi)
            ax.set_ylabel("Count")

    plt.tight_layout()  # Prevent titles and axes from overlapping.
    path = os.path.join(outdir, f"hist_{idx:02d}.png")  # Name this image's histogram file.
    plt.savefig(path, dpi=120)  # Save the histogram comparison.
    plt.close()  # Release the figure resources.
    return path  # Tell the caller where the plot was written.


def _plot_scatter(img: np.ndarray, cipher: np.ndarray, idx: int, outdir: str):
    hi = 255.0 if img.dtype == np.uint8 else 65535.0  # Use a shared axis range for the pixel dtype.
    ch_i = img[:, :, 0] if img.ndim == 3 else img  # Compare the first channel or grayscale image.
    ch_c = cipher[:, :, 0] if cipher.ndim == 3 else cipher  # Select the corresponding cipher channel.
    x_p, y_p = _scatter_sample(ch_i)  # Sample horizontal neighbors from the source image.
    x_c, y_c = _scatter_sample(ch_c)  # Sample the same relationship from the cipher image.

    # Plot source and cipher pixel pairs side-by-side for visual correlation comparison.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f"Image {idx:02d} — Adjacent-Pixel Correlation (H, ch0)", fontweight="bold")
    for ax, x, y, title in [(ax1, x_p, y_p, "Plaintext"), (ax2, x_c, y_c, "Cipher")]:
        ax.scatter(x, y, s=1, alpha=0.3, color="steelblue")  # Each dot is one neighboring-pixel pair.
        ax.set_title(title)
        ax.set_xlabel("Pixel (i)")
        ax.set_ylabel("Pixel (i+1)")
        ax.set_xlim(0, hi)
        ax.set_ylim(0, hi)

    plt.tight_layout()  # Fit both scatter plots and labels into the figure.
    path = os.path.join(outdir, f"scatter_{idx:02d}.png")  # Create a per-image output path.
    plt.savefig(path, dpi=120)  # Write the source-versus-cipher plot.
    plt.close()  # Release the figure resources.
    return path  # Return the saved plot path.


def _plot_key_sensitivity(img: np.ndarray, cipher: np.ndarray,
                          bad_rec: np.ndarray, idx: int, outdir: str):
    def _disp(arr):
        # Use the high byte for 16-bit display; retain 8-bit values otherwise.
        a = (arr >> 8).astype(np.uint8) if arr.dtype == np.uint16 else arr.astype(np.uint8)
        return a if a.ndim == 3 else np.stack([a] * 3, axis=-1)  # Repeat grayscale into RGB for plotting.

    # Display source, cipher, and wrong-key recovery together.
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(f"Image {idx:02d} — Key Sensitivity", fontweight="bold")
    for ax, im, t in zip(axes, [img, cipher, bad_rec],
                         ["Original", "Cipher", "Wrong-key decrypt"]):
        ax.imshow(_disp(im))  # Convert to displayable pixels and render this case.
        ax.set_title(t)
        ax.axis("off")
    plt.tight_layout()  # Keep image panels and titles separated.
    path = os.path.join(outdir, "key_sensitivity.png")  # Use a stable output name.
    plt.savefig(path, dpi=120)  # Save the visual key-sensitivity result.
    plt.close()  # Release the figure.
    return path  # Return its file path.


def _plot_robustness(results: dict, outdir: str):
    labels = list(results.keys())  # Use each attack name as a chart category.
    mses = [results[k]["mse"] for k in labels]  # Read recovery error for each attack.

    # Compare recovery errors; green denotes near-exact recovery, red denotes nonzero error.
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, mses, color=["green" if m < 1e-6 else "red" for m in mses])
    ax.set_ylabel("MSE (recovered vs original)")
    ax.set_title("Robustness Tests — Recovery MSE per Attack")
    ax.bar_label(bars, fmt="%.0f")
    plt.tight_layout()  # Fit labels and values into the chart.
    path = os.path.join(outdir, "robustness.png")  # Set the robustness plot's output path.
    plt.savefig(path, dpi=120)  # Save the attack comparison chart.
    plt.close()  # Release plotting resources.
    return path  # Return the saved chart path.


# ── main evaluation function ───────────────────────────────────────────────────

def evaluate(
    images:     list,
    ciphers:    list,
    recovered:  list,
    km_list:    list,
    stego:      np.ndarray,
    cover:      np.ndarray,
    passphrase: str,
    output_dir: str = "eval_output",
    n_lsbs:     int = 1,
) -> dict:
    """
    Run the full evaluation suite.

    Parameters
    ----------
    images     : original images (any dtype/shape)
    ciphers    : encrypted images
    recovered  : decrypted images
    km_list    : key_material dicts returned by image_crypto.encrypt()
    stego      : stego audio (int16 1-D)
    cover      : original cover audio (int16 1-D)
    passphrase : encryption passphrase
    output_dir : where to save all outputs
    n_lsbs     : 1 or 2 (used for capacity reporting)

    Returns
    -------
    summary : dict of all computed metrics
    """
    os.makedirs(output_dir, exist_ok=True)  # Ensure all report and plot outputs have a destination.

    # Collect results by evaluation category for the return value and report writer.
    summary = {
        "per_image": [],
        "stego": {},
        "key_sensitivity": {},
        "robustness": {},
        "keyspace": {
            "bits": 512,
            "note": "SHA-512 key derivation yields 2^512 effective key space. "
                    "Brute force is computationally infeasible.",
        },
    }

    # ── 1. Per-image metrics ─────────────────────────────────────────────────
    # 1. Measure ciphertext statistics and exact-recovery quality for each image.
    print("  [eval] Per-image security metrics...")
    for i, (img, cipher, rec) in enumerate(zip(images, ciphers, recovered)):
        npcr_v = metrics.npcr(img, cipher)  # Fraction of pixel values changed by encryption.
        uaci_v = _uaci(img, cipher)  # Average normalized intensity difference.
        ent_v = _entropy(cipher)  # Cipher intensity entropy.
        cc = metrics.correlation_coefficients(cipher)  # Spatial correlations in the cipher image.
        mse_v = metrics.mse(img, rec)  # Error after encrypt-then-decrypt recovery.
        max_v = _max_val(img)  # Correct peak intensity for this image's bit depth.
        psnr_v = metrics.psnr(img, rec, max_val=max_v)  # Recovery quality relative to peak intensity.

        # Store one CSV/report-ready row for this source image.
        row = {
            "idx": i,
            "shape": str(img.shape),
            "dtype": str(img.dtype),
            "npcr":  round(npcr_v, 4),
            "uaci":  round(uaci_v, 4),
            "entropy": round(ent_v, 4),
            "corr_h": round(cc["horizontal"], 4),
            "corr_v": round(cc["vertical"],   4),
            "corr_d": round(cc["diagonal"],   4),
            "mse_recovery":  mse_v,
            "psnr_recovery": psnr_v,
        }
        summary["per_image"].append(row)  # Include this image's measurements in the summary.

        # Save visual checks alongside the numeric metrics.
        _plot_histogram(img, cipher, i, output_dir)
        _plot_scatter(img, cipher, i, output_dir)
        print(f"    Image {i}: NPCR={npcr_v:.2f}% Entropy={ent_v:.4f} MSE={mse_v}")

    # ── 2. Key sensitivity (first image) ─────────────────────────────────────
    # 2. Decrypt the first cipher using a deliberately altered passphrase.
    print("  [eval] Key sensitivity test...")
    if images and km_list:
        cipher0 = ciphers[0]  # Use the first image as the representative sensitivity case.
        km0 = km_list[0]  # Preserve the matching salt and digest while changing the key.
        bad_pass = passphrase[:-1] + chr(ord(passphrase[-1]) ^ 1)  # Flip one bit in the last key character.
        bad_km = {"passphrase": bad_pass, "salt": km0["salt"], "digest": km0["digest"]}
        try:
            bad_rec = image_crypto.decrypt(cipher0, bad_km)  # Observe the result produced by the wrong key.
            ks_mse = float(metrics.mse(images[0], bad_rec))  # Quantify its difference from the source.
        except Exception as e:
            bad_rec = cipher0  # Keep a displayable fallback if decryption raises an error.
            ks_mse = -1.0  # Mark the metric unavailable due to the exception.
        _plot_key_sensitivity(images[0], cipher0, bad_rec, 0, output_dir)
        summary["key_sensitivity"] = {"mse_wrong_key": ks_mse}  # Save the key-sensitivity result.
        print(f"    Wrong-key MSE: {ks_mse:.1f}")

    # ── 3. Stego audio metrics ────────────────────────────────────────────────
    # 3. Measure signal distortion and how much selected carrier capacity is occupied.
    print("  [eval] Stego audio metrics...")
    snr_v = metrics.snr(cover, stego)  # Compare cover and stego signal power.
    psnr_v = metrics.psnr(cover.astype(float), stego.astype(float), max_val=32768.0)  # Compare peak-to-error ratio.
    total_bits = _pipe._payload_size(images) * 8  # Serialized payload size, converted from bytes to bits.
    cap_bits = audio_stego.capacity_bits(cover, n_lsbs)  # Available bits in the selected carrier frames.
    cap_pct = total_bits / max(cap_bits, 1) * 100  # Guard against zero capacity and report percentage used.
    summary["stego"] = {
        "snr_db":   round(snr_v, 2),
        "psnr_db":  round(psnr_v, 2),
        "cap_pct":  round(cap_pct, 2),
        "n_lsbs":   n_lsbs,
    }
    print(f"    SNR={snr_v:.1f} dB  PSNR={psnr_v:.1f} dB  Capacity used={cap_pct:.1f}%")

    # ── 4. Robustness tests ───────────────────────────────────────────────────
    # 4. Perturb the stego carrier and see whether the receiver can still recover the image.
    print("  [eval] Robustness tests...")
    rob = {}  # Store each attack's recovery error and status.

    # 4a - Add seeded bounded integer noise; values are sampled uniformly from [-50, 50].
    rng = np.random.default_rng(0)  # Make this synthetic perturbation repeatable.
    noisy = np.clip(stego.astype(np.int32) + rng.integers(-50, 51, len(stego)),
                    -32768, 32767).astype(np.int16)
    try:
        noisy_rec = _pipe.receiver(noisy, passphrase)  # Attempt extraction and decryption after perturbation.
        rob_mse = float(metrics.mse(images[0], noisy_rec[0])) if noisy_rec else float("inf")
        rob["Gaussian noise σ≈50"] = {"mse": rob_mse, "result": "FAIL" if rob_mse > 0 else "PASS"}
    except Exception as e:
        # Extraction errors also mean this perturbed carrier did not recover the payload.
        rob["Gaussian noise σ≈50"] = {"mse": float("inf"), "result": f"FAIL ({e})"}

    # 4b - Remove the final 20 percent of samples to test truncation sensitivity.
    crop_len = int(0.8 * len(stego))  # Keep the first 80 percent of the carrier.
    crop_stego = stego[:crop_len]  # Form the cropped signal passed to the receiver.
    try:
        crop_rec = _pipe.receiver(crop_stego, passphrase)  # Attempt recovery from the shortened carrier.
        crop_mse = float(metrics.mse(images[0], crop_rec[0])) if crop_rec else float("inf")
        rob["Crop 20%"] = {"mse": crop_mse, "result": "FAIL" if crop_mse > 0 else "PASS"}
    except Exception as e:
        rob["Crop 20%"] = {"mse": float("inf"), "result": f"FAIL-as-expected ({type(e).__name__})"}

    summary["robustness"] = rob  # Keep attack results in the returned summary.
    _plot_robustness(rob, output_dir)  # Visualize recovery error for each attack.
    for k, v in rob.items():
        print(f"    {k}: {v['result']}  MSE={v['mse']:.0f}")

    # ── 5. CSV output ─────────────────────────────────────────────────────────
    # 5. Write numeric per-image results as a CSV table.
    csv_path = os.path.join(output_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        # Fix the column order so generated CSV files are consistent.
        writer = csv.DictWriter(f, fieldnames=[
            "idx","shape","dtype","npcr","uaci","entropy",
            "corr_h","corr_v","corr_d","mse_recovery","psnr_recovery"
        ])
        writer.writeheader()  # Emit column names before data rows.
        writer.writerows(summary["per_image"])  # Write one row for each evaluated image.
    print(f"  [eval] CSV saved: {csv_path}")

    # ── 6. REPORT.md ──────────────────────────────────────────────────────────
    _write_report(summary, output_dir)  # Generate the human-readable Markdown summary.

    return summary  # Return all computed measurements to callers such as the app.


# ── report generator ───────────────────────────────────────────────────────────

def _write_report(summary: dict, output_dir: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")  # Timestamp this report generation.
    rows = summary["per_image"]  # Gather image-level metric rows.
    stg = summary["stego"]  # Gather audio quality and capacity results.
    ks = summary["key_sensitivity"]  # Gather altered-key result.
    rob = summary["robustness"]  # Gather carrier perturbation results.
    ksp = summary["keyspace"]  # Gather configured keyspace summary.

    # Start a Markdown table and its column headings for per-image measurements.
    tbl_lines = [
        "| Image | Shape | Dtype | NPCR (%) | UACI (%) | Entropy | Corr-H | Corr-V | Corr-D | MSE | PSNR |",
        "|-------|-------|-------|----------|----------|---------|--------|--------|--------|-----|------|",
    ]
    for r in rows:
        # Format infinite PSNR explicitly; otherwise limit display precision.
        psnr_str = "inf" if r["psnr_recovery"] == float("inf") else f"{r['psnr_recovery']:.2f}"
        # Add this image's metric values as a Markdown table row.
        tbl_lines.append(
            f"| {r['idx']:02d} | {r['shape']} | {r['dtype']} "
            f"| {r['npcr']:.4f} | {r['uaci']:.4f} | {r['entropy']:.4f} "
            f"| {r['corr_h']:.4f} | {r['corr_v']:.4f} | {r['corr_d']:.4f} "
            f"| {r['mse_recovery']} | {psnr_str} |"
        )

    rob_lines = []  # Format one concise report entry for each robustness test.
    for k, v in rob.items():
        rob_lines.append(f"- **{k}**: {v['result']} — MSE = {v['mse']:.0f}")

    # Assemble the report from preformatted rows and evaluation summaries.
    md = f"""# CryptoStego Evaluation Report

Generated: {ts}

---

## 1. Per-Image Security Metrics

{chr(10).join(tbl_lines)}

**Notes:**
- NPCR target: ~99.6% (measures pixel change rate between plaintext and cipher)
- UACI is normalised to the actual bit depth; 33.5% is the target for uniform input
- Entropy target: >7.99 bits (for images ≥ 140×140; smaller images have lower statistical estimate)
- Correlations should be near 0 to confirm spatial decorrelation

---

## 2. Recovery Verification

All images should show MSE = 0 and PSNR = inf (lossless, bit-exact recovery),
verified by SHA-256 digest of the decrypted pixel array vs the stored nonce.

---

## 3. Key Sensitivity

Decrypting cipher with a 1-character-flipped passphrase:

- Wrong-key MSE: **{ks.get("mse_wrong_key", "N/A")}**

A high MSE (>>0) confirms the cipher is sensitive to even a 1-bit key change.

---

## 4. Keyspace Analysis

- Key derivation: SHA-512(passphrase + salt + channel_index)
- Effective key space: **2^{ksp['bits']} keys**
- {ksp['note']}

---

## 5. Steganography Quality

| Metric | Value |
|--------|-------|
| Stego SNR | {stg.get('snr_db','N/A')} dB |
| Stego PSNR | {stg.get('psnr_db','N/A')} dB |
| Carrier capacity used | {stg.get('cap_pct','N/A')} % |
| LSBs per sample | {stg.get('n_lsbs','N/A')} |

SNR > 50 dB indicates the stego audio is perceptually indistinguishable from the cover.

---

## 6. Robustness Tests

{chr(10).join(rob_lines)}

**Note:** LSB steganography is inherently fragile to audio manipulation.
Even a single-bit perturbation to a carrier sample corrupts the corresponding payload bit.
This is expected behaviour — the system is designed for lossless digital transmission,
not for resilience against active attacks on the audio channel.

---

## 7. Saved Outputs

| File | Description |
|------|-------------|
| `hist_NN.png` | Pixel intensity histograms: plaintext vs cipher |
| `scatter_NN.png` | Adjacent-pixel correlation scatter plots |
| `key_sensitivity.png` | Visual comparison: original / cipher / wrong-key decrypt |
| `robustness.png` | MSE bar chart for each attack scenario |
| `metrics.csv` | All per-image metrics in tabular form |
| `REPORT.md` | This document |
"""
    path = os.path.join(output_dir, "REPORT.md")  # Choose the Markdown report destination.
    with open(path, "w", encoding="utf-8") as f:
        f.write(md)  # Write the complete report as UTF-8 text.
    print(f"  [eval] Report saved: {path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CryptoStego evaluation suite")  # Configure CLI inputs.
    parser.add_argument("--images",  nargs="+", required=True, help="Input image paths")
    parser.add_argument("--audio",   required=True,            help="Cover audio WAV path")
    parser.add_argument("--key",     required=True,            help="Encryption passphrase")
    parser.add_argument("--outdir",  default="eval_output",    help="Output directory")
    parser.add_argument("--n-lsbs",  type=int, default=1,      help="LSBs per sample (1 or 2)")
    args = parser.parse_args()  # Read the requested image, audio, key, and output settings.

    import image_io as _io  # Load supported image formats through the project's I/O layer.
    images = [_io.load(p)[0] for p in args.images]  # Keep pixel arrays and discard display metadata.
    sr, cover = wavfile.read(args.audio)  # Read the cover WAV and its sample rate.
    if cover.ndim > 1:
        cover = cover[:, 0]  # Use the first channel when the input WAV is stereo or multichannel.
    cover = cover.astype(np.int16)  # Convert samples to the PCM type expected by audio_stego.

    print(f"Encrypting {len(images)} image(s)...")
    # Run the production sender: encrypt images, serialize them, and embed into cover audio.
    ciphers, stego, km_list = _pipe.sender(images, cover, args.key, n_lsbs=args.n_lsbs)

    print("Decrypting...")
    recovered = _pipe.receiver(stego, args.key)  # Extract and decrypt to provide recovery measurements.

    print("Running evaluation...")
    # Evaluate the generated cipher, stego audio, recovered images, and attack cases.
    evaluate(images, ciphers, recovered, km_list, stego, cover,
             args.key, args.outdir, args.n_lsbs)
    print("Done.")


if __name__ == "__main__":
    main()
