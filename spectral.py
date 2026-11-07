"""Synthetic Beer–Lambert spectra and unconstrained concentration recovery."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np


WAVELENGTHS_NM = np.arange(350.0, 701.0)
TRUE_CONCENTRATIONS_M = np.array([22.0, 14.0, 9.0]) * 1e-6


@dataclass(frozen=True)
class Fit:
    concentrations_m: np.ndarray
    contributions: np.ndarray
    reconstructed: np.ndarray
    rank: int
    condition_number: float
    relative_concentration_error: float
    residual_rmse: float
    solver_seconds: float


def molar_absorptivity(spacing_nm: float) -> np.ndarray:
    """Return the three reference bands in inverse molarity per centimeter."""
    centers_nm = 520.0 + spacing_nm * np.array([-1.0, 0.0, 1.0])
    offsets = (WAVELENGTHS_NM[:, None] - centers_nm) / 25.0
    return 20_000.0 * np.exp(-0.5 * offsets**2)


def fit_mixture(matrix: np.ndarray, measured: np.ndarray) -> Fit:
    """Fit concentrations in mol/L for the demo mixture with known ground truth."""
    start = perf_counter()
    concentrations, _, rank, singular_values = np.linalg.lstsq(matrix, measured, rcond=None)
    solver_seconds = perf_counter() - start

    contributions = matrix * concentrations
    reconstructed = contributions.sum(axis=1)
    return Fit(
        concentrations_m=concentrations,
        contributions=contributions,
        reconstructed=reconstructed,
        rank=int(rank),
        condition_number=(
            float(singular_values[0] / singular_values[-1])
            if rank == matrix.shape[1]
            else float("inf")
        ),
        relative_concentration_error=float(
            np.linalg.norm(concentrations - TRUE_CONCENTRATIONS_M)
            / np.linalg.norm(TRUE_CONCENTRATIONS_M)
        ),
        residual_rmse=float(np.sqrt(np.mean((measured - reconstructed) ** 2))),
        solver_seconds=solver_seconds,
    )
