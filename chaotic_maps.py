"""Deterministic chaotic sequence generators used by the image cipher.

The Quantum Logistic Map supplies pixel permutations. The Zaslavsky and
Henon-like maps supply values combined into the diffusion keystream by
image_crypto.py. Initial transient values are discarded before use.
"""
import numpy as np  # NumPy stores the generated sequences and converts them to bytes.


class QuantumLogisticMap:
    """Generate values for the first and second image-pixel permutations."""

    def __init__(self, x0: float = 0.3, r: float = 3.99, beta: float = 6.0, transient: int = 500):
        # Keep r in the configured chaotic range used by this map.
        assert 3.74 <= r <= 4.0, "r must be in [3.74, 4.0]"
        # Require beta to meet the map's configured lower bound.
        assert beta >= 6, "beta must be >= 6"
        self.x0 = x0  # Starting state for each generated sequence.
        self.r = r  # Growth parameter in the recurrence.
        self.beta = beta  # Strength of the additional nonlinear term.
        self.transient = transient  # Number of initial iterations to skip.

    def _iterate(self, x: float, n: int) -> np.ndarray:
        out = np.empty(n)  # Allocate space for the requested sequence.
        for i in range(n):
            # Apply the quantum logistic recurrence to produce the next state.
            x = self.r * x * (1 - x) + self.beta * self.r * x**2 * (1 - x)**2
            # Fold values outside (0, 1) back into the unit interval.
            x = x - np.floor(x) if x >= 1.0 or x <= 0.0 else x
            out[i] = x  # Save this state as the next sequence value.
        return out  # Return all generated values to the caller.

    def sequence(self, n: int) -> np.ndarray:
        """Return n values in (0, 1) after discarding the transient."""
        x = self.x0  # Restart each sequence from the configured initial state.
        # Let initial behavior die out so callers use the later map orbit.
        for _ in range(self.transient):
            # Advance one state using the same recurrence as _iterate().
            x = self.r * x * (1 - x) + self.beta * self.r * x**2 * (1 - x)**2
            # Correct out-of-range states and avoid landing exactly on zero.
            if x >= 1.0 or x <= 0.0:
                x = abs(x - np.floor(x)) + 1e-9
        return self._iterate(x, n)  # Generate the requested post-transient values.

    def permutation(self, n: int) -> np.ndarray:
        """Return an argsort-based pixel permutation of length n."""
        seq = self.sequence(n)  # Generate one chaotic value for each pixel index.
        return np.argsort(seq)  # Order indices by those values to shuffle pixels.


class ZaslavskyMap:
    """Generate one component of the image-diffusion keystream."""

    def __init__(self, x0: float = 0.1, y0: float = 0.2,
                 nu: float = 0.5, eps: float = 0.9, gamma: float = 0.5,
                 transient: int = 500):
        self.x0 = x0  # Initial x state of the two-dimensional map.
        self.y0 = y0  # Initial y state of the two-dimensional map.
        self.nu = nu  # Controls the map's base x movement.
        self.eps = eps  # Controls the cosine perturbation strength.
        self.gamma = gamma  # Controls exponential damping in the y update.
        self.mu = (1 - np.exp(-gamma)) / gamma  # Derived coefficient used in the x update.
        self.transient = transient  # Number of initial state updates to skip.

    def _step(self, x: float, y: float):
        # Copy parameters locally for the two recurrence equations below.
        nu, eps, mu, gamma = self.nu, self.eps, self.mu, self.gamma
        # Advance x with drift and a periodic cosine perturbation; modulo 1 wraps it.
        xn = (x + nu * (1 + mu * y) + eps * nu * mu * np.cos(2 * np.pi * x)) % 1.0
        # Advance y with a cosine perturbation and exponential damping.
        yn = np.exp(-gamma) * (y + eps * np.cos(2 * np.pi * x))
        return xn, yn  # Return the next state for the next iteration.

    def sequence(self, n: int) -> np.ndarray:
        """Return n x-values (mod 1) after discarding transient."""
        x, y = self.x0, self.y0  # Start from the configured two-variable state.
        # Skip startup behavior before collecting values for encryption.
        for _ in range(self.transient):
            x, y = self._step(x, y)  # Advance both state variables once.
        out = np.empty(n)  # Allocate the x-value sequence returned to the cipher.
        for i in range(n):
            x, y = self._step(x, y)  # Generate the next point on the map orbit.
            out[i] = x  # Keep x; image_crypto combines it with Henon values.
        return out  # Return values already wrapped into [0, 1).


class Henon3DMap:
    """Generate the second component of the image-diffusion keystream."""

    def __init__(self, x0: float = 0.1, y0: float = 0.0, z0: float = 0.0,
                 a: float = 1.76, b: float = 0.1, transient: int = 500):
        self.x0 = x0  # Initial x state of the three-dimensional map.
        self.y0 = y0  # Initial y state of the three-dimensional map.
        self.z0 = z0  # Initial z state of the three-dimensional map.
        self.a = a  # Constant offset in the x recurrence.
        self.b = b  # Weight applied to z in the x recurrence.
        self.transient = transient  # Number of initial state updates to skip.

    def _step(self, x: float, y: float, z: float):
        # Nonlinear x update; y squared provides the nonlinear term.
        xn = self.a - y**2 - self.b * z
        # Shift the previous x state into the next y state.
        yn = x
        # Shift the previous y state into the next z state.
        zn = y
        # Wrap each component into [-2, 2] to keep the numerical state bounded.
        xn = ((xn + 2) % 4) - 2
        yn = ((yn + 2) % 4) - 2
        zn = ((zn + 2) % 4) - 2
        return xn, yn, zn  # Return the next three-dimensional state.

    def sequence(self, n: int) -> np.ndarray:
        """Return n x-values mapped to (0,1) after discarding transient."""
        x, y, z = self.x0, self.y0, self.z0  # Start from the configured initial state.
        # Skip startup behavior before collecting values for encryption.
        for _ in range(self.transient):
            x, y, z = self._step(x, y, z)  # Advance the complete map state once.
        out = np.empty(n)  # Allocate the normalized x-value sequence.
        for i in range(n):
            x, y, z = self._step(x, y, z)  # Generate the next point on the map orbit.
            out[i] = (x + 2) / 4.0  # Scale x from [-2, 2] into [0, 1].
        return out  # Return values used with the Zaslavsky sequence.


def to_keystream_bytes(seq: np.ndarray) -> np.ndarray:
    """Convert float sequence in [0,1) to uint8 keystream bytes."""
    scaled = seq * 256  # Expand each fractional value to the 0-256 byte range.
    floored = np.floor(scaled)  # Keep the integer bucket for each value.
    bounded = np.clip(floored, 0, 255)  # Ensure every value fits in one byte.
    return bounded.astype(np.uint8)  # Produce bytes consumed by image diffusion.
