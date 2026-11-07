"""Run the synthetic UV–Vis concentration recovery experiment."""

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from plots import plot_concentrations, plot_conditioning, plot_spectra
from spectral import TRUE_CONCENTRATIONS_M, WAVELENGTHS_NM, fit_mixture, molar_absorptivity


SWEEP_SPACINGS_NM = (60, 30, 15, 8, 4)
STRESS_SPACING_NM = 4
PATH_LENGTH_CM = 1.0
METRIC_FIELDS = (
    "spacing_nm", "trial", "noise_std_au", "rank", "condition_number",
    "relative_concentration_error", "residual_rmse_au", "solver_seconds",
)


def simulate(bands, noise):
    matrix = PATH_LENGTH_CM * bands
    measured = matrix @ TRUE_CONCENTRATIONS_M + noise
    return measured, fit_mixture(matrix, measured)


def metric_rows(spacings_nm, noise_std_au, noise_samples):
    for spacing_nm in spacings_nm:
        bands = molar_absorptivity(spacing_nm)
        for trial, noise in enumerate(noise_samples):
            _, fit = simulate(bands, noise)
            yield {
                "spacing_nm": spacing_nm,
                "trial": trial,
                "noise_std_au": noise_std_au,
                "rank": fit.rank,
                "condition_number": fit.condition_number,
                "relative_concentration_error": fit.relative_concentration_error,
                "residual_rmse_au": fit.residual_rmse,
                "solver_seconds": fit.solver_seconds,
            }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spacing", type=float, default=30.0, help="Demo peak spacing in nm")
    parser.add_argument("--noise", type=float, default=0.003, help="Noise standard deviation in AU")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--trials", type=int, default=256, help="Noise draws per spacing")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    if not math.isfinite(args.spacing) or not 0 <= args.spacing <= 170:
        parser.error("--spacing must be between 0 and 170 nm to keep all peaks on the grid")
    if not math.isfinite(args.noise) or args.noise < 0:
        parser.error("--noise must be finite and nonnegative")
    if args.seed < 0 or args.trials < 1:
        parser.error("--seed must be nonnegative and --trials must be positive")

    rng = np.random.default_rng(args.seed)
    noise_samples = rng.normal(0, args.noise, size=(args.trials, WAVELENGTHS_NM.size))
    spacings = dict.fromkeys((*SWEEP_SPACINGS_NM, 0, args.spacing))
    rows = list(metric_rows(spacings, args.noise, noise_samples))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    bands = molar_absorptivity(args.spacing)
    measured, fit = simulate(bands, noise_samples[0])
    cases = [(args.spacing, fit)]
    if args.spacing != STRESS_SPACING_NM:
        _, stress_fit = simulate(molar_absorptivity(STRESS_SPACING_NM), noise_samples[0])
        cases.append((STRESS_SPACING_NM, stress_fit))

    plot_spectra(args.output_dir / "spectra.svg", args.spacing, bands, measured, fit)
    plot_concentrations(args.output_dir / "concentrations.svg", cases)
    plot_conditioning(args.output_dir / "conditioning.svg", sorted(SWEEP_SPACINGS_NM), rows, args.noise)

    print(f"Demo spacing {args.spacing:g} nm, noise SD {args.noise:g} AU")
    print("True concentrations (uM):", np.round(TRUE_CONCENTRATIONS_M * 1e6, 3))
    print("Fitted concentrations (uM):", np.round(fit.concentrations_m * 1e6, 3))
    print(f"Rank {fit.rank}/3, condition number {fit.condition_number:.3g}")
    print(f"Relative error {fit.relative_concentration_error:.2%}, residual RMSE {fit.residual_rmse:.4g} AU")
    if fit.rank < TRUE_CONCENTRATIONS_M.size:
        print("Individual concentrations are not identifiable at this spacing.")
    print(f"Wrote results to {args.output_dir}")


if __name__ == "__main__":
    main()
