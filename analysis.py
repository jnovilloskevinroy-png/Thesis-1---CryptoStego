"""Visual analysis plots for image statistics and encryption key sensitivity."""
import os  # Creates output directories and joins output file paths.
import numpy as np  # Samples image pixels and prepares arrays for plotting.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import image_crypto
import metrics


def _sample_pairs(ch: np.ndarray, n: int = 3000, seed: int = 0):
    """Sample n random adjacent pixel pairs (horizontal) from a channel."""
    rng = np.random.default_rng(seed)  # Seed sampling so plots are reproducible.
    H, W = ch.shape  # Read channel dimensions to choose valid neighboring coordinates.
    rows = rng.integers(0, H, n)  # Randomly select row indices.
    cols = rng.integers(0, W - 1, n)  # Stop before the last column so a right neighbor exists.
    x = ch[rows, cols].astype(np.float32)  # Sample the left pixel from each pair.
    y = ch[rows, cols + 1].astype(np.float32)  # Sample its immediate right neighbor.
    return x, y  # Return corresponding values for a scatter plot.


def plot_histograms(plain: np.ndarray, cipher: np.ndarray,
                   outpath: str = "analysis_histograms.png"):
    """RGB histograms of plaintext vs cipher image."""
    channel_names = ["Red", "Green", "Blue"]  # Labels for the three image channels.
    channel_colors = ["red", "green", "blue"]  # Use matching colors for plaintext histograms.

    # Create rows for plaintext and cipher, and columns for RGB channels.
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    fig.suptitle("Pixel Intensity Histograms", fontsize=14, fontweight="bold")

    for c in range(3):
        # Plot the original intensity distribution in the top row.
        ax = axes[0, c]
        ax.hist(plain[:, :, c].ravel(), bins=256, range=(0, 255),
                color=channel_colors[c], alpha=0.75)
        ax.set_title(f"Plaintext – {channel_names[c]}")
        ax.set_xlim(0, 255)
        ax.set_ylabel("Count")

        # Plot the encrypted intensity distribution in the bottom row.
        ax = axes[1, c]
        ax.hist(cipher[:, :, c].ravel(), bins=256, range=(0, 255),
                color="gray", alpha=0.75)
        ax.set_title(f"Cipher – {channel_names[c]}")
        ax.set_xlim(0, 255)
        ax.set_ylabel("Count")

    plt.tight_layout()  # Adjust margins to keep labels and plots from overlapping.
    plt.savefig(outpath, dpi=120)  # Save the comparison image at readable resolution.
    plt.close()  # Release the figure instead of retaining it in memory.
    print(f"Saved: {outpath}")


def plot_correlations(plain: np.ndarray, cipher: np.ndarray,
                      outpath: str = "analysis_correlations.png"):
    """Adjacent-pixel correlation scatter plots (horizontal, first channel only)."""
    # Compare original and cipher pixel-neighbor relationships side by side.
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Adjacent-Pixel Correlation (Red channel, horizontal)",
                 fontsize=13, fontweight="bold")

    for ax, img, title in zip(axes, [plain, cipher], ["Plaintext", "Cipher"]):
        x, y = _sample_pairs(img[:, :, 0])  # Sample horizontal neighbors from the first channel.
        ax.scatter(x, y, s=1, alpha=0.3, color="steelblue")  # Plot each adjacent pair as a point.
        ax.set_title(title)
        ax.set_xlabel("Pixel (i)")
        ax.set_ylabel("Pixel (i+1)")
        ax.set_xlim(0, 255)
        ax.set_ylim(0, 255)

    plt.tight_layout()  # Fit plot labels within the figure.
    plt.savefig(outpath, dpi=120)  # Write the scatter comparison to disk.
    plt.close()  # Release plotting resources.
    print(f"Saved: {outpath}")


def plot_key_sensitivity(plain: np.ndarray, passphrase: str,
                         outpath: str = "analysis_key_sensitivity.png"):
    """Decrypt with correct key, 1-char-wrong key, and 2-char-wrong key."""
    cipher, km = image_crypto.encrypt(plain, passphrase)  # Encrypt once for all key comparisons.

    # Create a wrong passphrase by changing its last character.
    bad1 = passphrase[:-1] + ("X" if passphrase[-1] != "X" else "Y")
    # Create a second wrong passphrase by changing its first character.
    bad2 = ("A" if passphrase[0] != "A" else "B") + passphrase[1:]

    correct_rec = image_crypto.decrypt(cipher, km)  # Baseline: decrypt using the returned correct key material.
    # Also alter the salt in this case to test sensitivity to image-specific parameters.
    bad1_rec = image_crypto.decrypt(
        cipher, {**km, "salt": km["salt"][:-1] + ("0" if km["salt"][-1] != "0" else "1"),
                 "passphrase": bad1}
    )
    # Keep the salt unchanged here and vary only the passphrase.
    bad2_rec = image_crypto.decrypt(
        cipher, {**km, "passphrase": bad2}
    )

    images = [plain, cipher, correct_rec, bad1_rec, bad2_rec]  # Arrange all cases for the comparison panel.
    titles = [
        "Plaintext",
        "Cipher",
        f"Correct key\n(MSE={metrics.mse(plain, correct_rec):.1f})",
        f"1-char wrong key\n(MSE={metrics.mse(plain, bad1_rec):.1f})",
        f"2-char wrong key\n(MSE={metrics.mse(plain, bad2_rec):.1f})",
    ]

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))  # Give each key case its own panel.
    fig.suptitle("Key-Sensitivity Analysis", fontsize=13, fontweight="bold")

    for ax, img_arr, title in zip(axes, images, titles):
        ax.imshow(img_arr.astype(np.uint8))  # Display each input or decryption result as an image.
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    plt.tight_layout()  # Arrange the five panels without overlapping titles.
    plt.savefig(outpath, dpi=120)  # Save the key-sensitivity visualization.
    plt.close()  # Release the completed figure.
    print(f"Saved: {outpath}")


def run_all(plain: np.ndarray, passphrase: str = "CryptoStegoKey2024!", outdir: str = "."):
    os.makedirs(outdir, exist_ok=True)  # Ensure the destination directory exists.
    cipher, km = image_crypto.encrypt(plain, passphrase)  # Produce the cipher and matching decryption material.

    # Generate the three complementary diagnostics for this image and key.
    plot_histograms(
        plain, cipher,
        outpath=os.path.join(outdir, "analysis_histograms.png")
    )
    plot_correlations(
        plain, cipher,
        outpath=os.path.join(outdir, "analysis_correlations.png")
    )
    plot_key_sensitivity(
        plain, passphrase,
        outpath=os.path.join(outdir, "analysis_key_sensitivity.png")
    )


if __name__ == "__main__":
    import argparse  # Parse optional image, key, output, and seed settings.
    from pipeline import _synth_image  # Supply a demo image when no path is provided.

    parser = argparse.ArgumentParser(description="CryptoStego analysis plots")  # Configure the analysis CLI.
    parser.add_argument("--image",  default=None)
    parser.add_argument("--key",    default="CryptoStegoKey2024!")
    parser.add_argument("--outdir", default=".")
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()  # Read command-line values.

    if args.image:
        img = np.array(Image.open(args.image).convert("RGB"))  # Load the requested image as RGB.
    else:
        img = _synth_image(64, 64, seed=args.seed)  # Make a repeatable sample image for a demo run.

    run_all(img, passphrase=args.key, outdir=args.outdir)  # Create the analysis plots.
