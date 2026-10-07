"""Image-quality, cipher-statistics, and audio-distortion metric formulas."""
import numpy as np  # Array operations used to calculate each metric.


def npcr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Number of Pixels Change Rate (%) between two images."""
    # Pairwise pixel comparisons require arrays with matching dimensions.
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same shape.")
    changed = np.sum(img1 != img2)  # Count locations whose pixel values differ.
    total = img1.size  # Count all scalar values, including channels.
    return 100.0 * changed / total  # Express changed values as a percentage.


def uaci(img1: np.ndarray, img2: np.ndarray) -> float:
    """Unified Average Changing Intensity (%)."""
    # Compare corresponding pixels only when image dimensions agree.
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same shape.")
    diff = np.abs(img1.astype(np.float64) - img2.astype(np.float64))  # Absolute intensity change at each value.
    max_v = 65535.0 if img1.dtype == np.uint16 else 255.0  # Normalize by the dtype's maximum value.
    return 100.0 * diff.mean() / max_v  # Report mean normalized change as a percentage.


def mse(original: np.ndarray, recovered: np.ndarray) -> float:
    """Mean squared error between two images."""
    # Convert before subtraction so unsigned integer arithmetic cannot wrap around.
    return float(np.mean((original.astype(np.float64) - recovered.astype(np.float64)) ** 2))


def psnr(original: np.ndarray, recovered: np.ndarray, max_val: float = 255.0) -> float:
    """Peak Signal-to-Noise Ratio (dB)."""
    m = mse(original, recovered)  # Reuse MSE as the reconstruction error power.
    if m == 0:
        return float("inf")  # Exact recovery has zero error and infinite PSNR.
    return 10.0 * np.log10(max_val ** 2 / m)  # Convert peak-to-error ratio to decibels.


def snr(signal: np.ndarray, noisy: np.ndarray) -> float:
    """Signal-to-Noise Ratio (dB) — suitable for audio."""
    # Mean squared amplitudes estimate signal and distortion power.
    sig_power = np.mean(signal.astype(np.float64) ** 2)
    noise_power = np.mean((signal.astype(np.float64) - noisy.astype(np.float64)) ** 2)
    if noise_power == 0:
        return float("inf")  # Identical clean and stego signals have no measured distortion.
    return 10.0 * np.log10(sig_power / noise_power)  # Express signal-to-distortion power ratio in dB.


def shannon_entropy(img: np.ndarray) -> float:
    """Shannon entropy (bits per pixel, averaged over channels)."""
    entropies = []  # Store one entropy value per image channel.
    n_bins = 65536 if img.dtype == np.uint16 else 256  # Use one histogram bin per possible intensity.
    hi = 65536 if img.dtype == np.uint16 else 256  # Histogram range upper bound is exclusive.
    if img.ndim == 3:
        for c in range(img.shape[2]):
            # Build an intensity histogram independently for this channel.
            hist, _ = np.histogram(img[:, :, c], bins=n_bins, range=(0, hi))
            prob = hist / hist.sum()  # Convert counts to empirical intensity probabilities.
            prob = prob[prob > 0]  # Ignore empty bins because log2(0) is undefined.
            entropies.append(-np.sum(prob * np.log2(prob)))  # Shannon entropy for this channel.
    else:
        # Grayscale input has only one intensity histogram.
        hist, _ = np.histogram(img, bins=n_bins, range=(0, hi))
        prob = hist / hist.sum()  # Convert counts to empirical probabilities.
        prob = prob[prob > 0]  # Exclude intensity levels absent from the image.
        entropies.append(-np.sum(prob * np.log2(prob)))  # Compute grayscale Shannon entropy.
    return float(np.mean(entropies))  # Average channel entropy for color images.


def correlation_coefficients(img: np.ndarray) -> dict:
    """
    Adjacent-pixel correlation coefficients (H, V, D) per channel, averaged.
    Returns dict with keys 'horizontal', 'vertical', 'diagonal'.
    """
    if img.ndim == 2:
        channels = [img]  # Treat grayscale as a one-channel image.
    else:
        channels = [img[:, :, c] for c in range(img.shape[2])]  # Analyze each color channel separately.

    results = {"horizontal": [], "vertical": [], "diagonal": []}  # Collect coefficients per direction.

    for ch in channels:
        ch = ch.astype(np.float64)  # Use floating-point values for covariance calculations.
        H, W = ch.shape  # Read dimensions to form neighboring pixel pairs.

        # Pair every pixel with its right-hand neighbor.
        x_h = ch[:, :-1].ravel()
        y_h = ch[:, 1:].ravel()
        results["horizontal"].append(_corr(x_h, y_h))  # Measure horizontal dependence.

        # Pair every pixel with the pixel directly below it.
        x_v = ch[:-1, :].ravel()
        y_v = ch[1:, :].ravel()
        results["vertical"].append(_corr(x_v, y_v))  # Measure vertical dependence.

        # Pair every pixel with its down-right diagonal neighbor.
        x_d = ch[:-1, :-1].ravel()
        y_d = ch[1:, 1:].ravel()
        results["diagonal"].append(_corr(x_d, y_d))  # Measure diagonal dependence.

    # Return channel-averaged correlations for each spatial direction.
    return {k: float(np.mean(v)) for k, v in results.items()}


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation coefficient."""
    xm = x - x.mean()  # Center the first set of pixel values around its mean.
    ym = y - y.mean()  # Center the neighboring values around their mean.
    denom = np.sqrt(np.sum(xm ** 2) * np.sum(ym ** 2))  # Normalize covariance by both standard deviations.
    if denom == 0:
        return 0.0  # A constant input has undefined correlation; report neutral zero.
    return float(np.sum(xm * ym) / denom)  # Pearson correlation coefficient in [-1, 1].
