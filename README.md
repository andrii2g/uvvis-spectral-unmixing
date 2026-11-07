# UV-Vis Spectral Unmixing

Recover the concentrations of three synthetic substances from a single mixture spectrum. The experiment shows how overlapping absorption bands can make individual concentrations sensitive to noise even when the fitted spectrum closely matches the measurements.

## Run the demo

```sh
python -m pip install -r requirements.txt
python unmix.py --spacing 30 --noise 0.003 --seed 42 --trials 256 --output-dir results
```

The default run generates four files in `results/`.

| File | Contents |
| --- | --- |
| `spectra.svg` | Pure-component reference bands, noisy mixture measurements, fitted total, and each component's fitted contribution |
| `concentrations.svg` | True and fitted concentrations for the selected spacing and the 4 nm case using the same noise draw |
| `conditioning.svg` | Condition number and median errors with 10th–90th percentile bands across five peak spacings |
| `metrics.csv` | One row per spacing and trial, with noise standard deviation, numerical rank, condition number, relative concentration error, residual RMSE, and solve time in seconds |

## Model

The Beer–Lambert design matrix is $E_{ij}=l\,\varepsilon_j(\lambda_i)$. The program samples from 350 to 700 nm at 1 nm intervals. It uses a 1 cm optical path and three Gaussian bands centered at $520-d$, $520$, and $520+d$ nm, where $d$ is the adjacent peak spacing. Each band has a standard deviation of 25 nm and a peak molar absorptivity of 20,000 M⁻¹ cm⁻¹. The true concentrations are 22, 14, and 9 µM. [IUPAC Beer–Lambert law](https://goldbook.iupac.org/terms/view/B00626)

The program adds independent zero-mean Gaussian noise to the mixture absorbance at each wavelength. Each trial reuses its noise draw across spacings. NumPy least squares fits the concentrations in mol/L. The fit retains signed estimates and reports numerical rank. Relative concentration error is the Euclidean norm of the concentration error divided by the norm of the true concentrations. Residual RMSE measures the difference between measured and fitted absorbance. Solver time covers only the least-squares call. [NumPy least squares](https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html)

## Example results

With the default seed and 256 trials, decreasing the adjacent peak spacing from 30 nm to 4 nm raises the median concentration error by about 35 times. The median absorbance residual is about 0.003 AU in both cases.

| Peak spacing | Condition number | Median concentration error | Median residual RMSE |
| --- | ---: | ---: | ---: |
| 30 nm | 4.12 | 0.19% | 0.0030 AU |
| 4 nm | 233.89 | 6.72% | 0.0030 AU |

Try `python unmix.py --spacing 0 --noise 0 --output-dir results/coincident` for the rank-one case. The three reference bands coincide, so the data determine only the total concentration. The solver returns the minimum-norm split of 15 µM per component, which sums to the true 45 µM. A noiseless run at 30 nm recovers all three concentrations to floating-point precision.
