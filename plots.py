"""SVG reports for the synthetic unmixing experiment."""

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from spectral import TRUE_CONCENTRATIONS_M, WAVELENGTHS_NM


COMPONENTS = ("A", "B", "C")
COLORS = ("#0072B2", "#D55E00", "#009E73")


def save_figure(figure, path):
    figure.savefig(path)
    plt.close(figure)


def plot_spectra(path, spacing_nm, bands, measured, fit):
    figure, (pure, mixture) = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    figure.suptitle(f"Synthetic UV–Vis spectra · {spacing_nm:g} nm peak spacing")

    for index, (component, color) in enumerate(zip(COMPONENTS, COLORS)):
        pure.plot(WAVELENGTHS_NM, bands[:, index], color=color, label=component)
        mixture.plot(WAVELENGTHS_NM, fit.contributions[:, index], color=color, label=component)
    mixture.scatter(WAVELENGTHS_NM, measured, s=6, alpha=0.45, color="gray", label="Measured")
    mixture.plot(WAVELENGTHS_NM, fit.reconstructed, color="black", label="Fitted total")

    pure.set(title="Pure-component reference bands", ylabel="Molar absorptivity (M⁻¹ cm⁻¹)")
    mixture.set(title="Mixture and fitted contributions", ylabel="Absorbance (AU)")
    for axis in (pure, mixture):
        axis.set(xlabel="Wavelength (nm)")
        axis.legend()
    save_figure(figure, path)


def plot_concentrations(path, cases):
    figure, axes = plt.subplots(
        1, len(cases), figsize=(4.6 * len(cases), 4), sharey=True,
        squeeze=False, layout="constrained",
    )
    positions = np.arange(len(COMPONENTS))
    for axis, (spacing_nm, fit) in zip(axes[0], cases):
        axis.bar(positions - 0.18, TRUE_CONCENTRATIONS_M * 1e6, 0.36,
                 color="white", edgecolor="black", label="True")
        axis.bar(positions + 0.18, fit.concentrations_m * 1e6, 0.36,
                 color=COLORS, label="Fitted")
        axis.axhline(0, color="black", linewidth=0.7)
        axis.set(xticks=positions, xticklabels=COMPONENTS, xlabel="Component",
                 title=f"{spacing_nm:g} nm spacing · rank {fit.rank}/3\n"
                       f"{fit.relative_concentration_error:.1%} relative error")
    axes[0, 0].set_ylabel("Concentration (µM)")
    axes[0, 0].legend()
    save_figure(figure, path)


def plot_conditioning(path, spacings_nm, rows, noise_std_au):
    groups = [[row for row in rows if row["spacing_nm"] == spacing] for spacing in spacings_nm]
    figure, axes = plt.subplots(3, 1, figsize=(7, 7), sharex=True, layout="constrained")
    figure.suptitle(f"Peak overlap and recovery · noise SD {noise_std_au:g} AU · {len(groups[0])} trials")

    condition, error, residual = axes
    condition.plot(spacings_nm, [group[0]["condition_number"] for group in groups], "o-")
    condition.set(ylabel="Condition number", yscale="log")
    condition.text(0.98, 0.94, "Coincident peaks (0 nm): rank 1", transform=condition.transAxes,
                   ha="right", va="top")

    for axis, field, scale, label in (
        (error, "relative_concentration_error", 100, "Concentration error (%)"),
        (residual, "residual_rmse_au", 1, "Residual RMSE (AU)"),
    ):
        low, median, high = np.array([
            np.percentile([row[field] for row in group], [10, 50, 90]) for group in groups
        ]).T * scale
        axis.fill_between(spacings_nm, low, high, alpha=0.2, label="10th–90th percentile")
        axis.plot(spacings_nm, median, "o-", label="Median")
        axis.set_ylabel(label)
    error.legend()
    residual.set(xlabel="Adjacent peak spacing (nm)", xticks=spacings_nm)
    for axis in axes:
        axis.grid(alpha=0.2)
    save_figure(figure, path)
