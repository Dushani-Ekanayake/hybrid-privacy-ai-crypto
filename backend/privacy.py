"""
privacy.py — Differential Privacy layer

We use the Laplace mechanism here. The idea is simple:
after the model makes a prediction, we add a tiny amount of
random noise calibrated to (sensitivity / epsilon).

- epsilon: privacy budget. smaller = more private, less accurate.
- sensitivity: how much one person's data can shift the output.
"""

import numpy as np


class DifferentialPrivacy:
    def __init__(self, epsilon: float = 1.0, sensitivity: float = 1.0):
        if epsilon <= 0:
            raise ValueError("Epsilon must be positive.")
        self.epsilon = epsilon
        self.sensitivity = sensitivity

    def laplace_noise(self, shape=None):
        """Draw noise from Laplace distribution."""
        scale = self.sensitivity / self.epsilon
        return np.random.laplace(loc=0.0, scale=scale, size=shape)

    def privatize(self, value: float) -> float:
        """Add Laplace noise to a single prediction value."""
        noise = self.laplace_noise()
        noisy = value + noise
        # Clip to valid probability range
        return float(np.clip(noisy, 0.0, 1.0))

    def privatize_batch(self, values: list) -> list:
        """Add noise to a list of predictions."""
        return [self.privatize(v) for v in values]

    def info(self) -> dict:
        return {
            "mechanism": "Laplace",
            "epsilon": self.epsilon,
            "sensitivity": self.sensitivity,
            "noise_scale": self.sensitivity / self.epsilon,
        }
