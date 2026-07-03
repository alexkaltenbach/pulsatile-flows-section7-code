# pNavierStokesMixed - Section 7.3

This folder contains the FEniCS implementation for Section 7.3 of the article:
the comparison between the reduced 1D approximation and a direct 2D
Taylor-Hood approximation on a finite strip.

## Mathematical Setup

The code uses the paper parameters

- `xmax = 20`
- `Sigma = (-1/2, 1/2)`
- `I = (0, 2*pi)`
- `alpha(t) = cos(t)`
- `p(x) = 2 + x` on the cross-section
- comparison region `omega = (5,15) x (-1/2,1/2)`

The full 2D problem is solved with quadratic velocity elements and linear
pressure elements. The reduced 1D comparison data are generated with the solver
in `../pLaplaceMixed`, using the stress factor `2^{-p/2}` described in
Section 7.3.

## Run Section 7.3

From this folder:

```bash
cd ~/Documents/Fenics/pNavierStokesMixed
python3 run_section_73.py --stage all --nmax 9
```

This executes

1. `ErrorDecayExperiment1D.py`: reduced 1D comparison data in `Comparison1D/`
2. `ErrorDecayExperiment.py`: full 2D data in `Comparison/`
3. `ErrorDecayComparison.py`: errors in `errors.pkl`
4. `ErrorPlot.py`: Section 7.3 convergence and approximation plot
5. `PlotSolution.py`: pressure and velocity surface plots

To compute without opening plot windows:

```bash
python3 run_section_73.py --stage all --nmax 9 --skip-plots
```

To run individual stages:

```bash
python3 run_section_73.py --stage one-d
python3 run_section_73.py --stage two-d
python3 run_section_73.py --stage compare
python3 run_section_73.py --stage plot
python3 run_section_73.py --stage surface
```

To save the Section 7.3 figures directly after the snapshots have been
generated:

```bash
python3 ErrorPlot.py --save-dir plots --format pdf png --no-show
python3 PlotSolution.py --save-dir plots --format pdf png --no-show
```

This writes `plots/Figure11.{pdf,png}` and separate Figure 12 files
`plots/Figure12_pressure.{pdf,png}` and
`plots/Figure12_velocity.{pdf,png}`.

The pressure plot shows
`\vert\overline{\pi_h^\tau-\Gamma^\tau x_1}^{\,\omega}\vert`, i.e. the
current mean-free pressure-potential difference on `omega`. The velocity plot
shows `\vert\mathbf{v}_h^\tau-v_h^\tau\mathbf{e}_1\vert`.

The `compare` stage checks `Comparison1D/iterations.pkl` and
`Comparison/iterations.pkl`. If you request `--nmax N`, both data stages must
contain all levels `n=1,...,N-1`; otherwise the comparison stops with a clear
data-consistency error.

## Output Files

- `Comparison1D/v*/` and `Comparison1D/Gamma*/`: reduced 1D snapshots
- `Comparison/v*/` and `Comparison/pi*/`: full 2D snapshots
- `errors.pkl`: Section-7.3 error data and EOCs. The comparison script stores
  six main quantities:
  `Omega_L2`, `Omega_H1`, `Omega_Gamma`, `omega_L2`, `omega_H1`,
  and `omega_Gamma`. Here `Gamma` denotes the square root of the modular
  pressure-potential error. On each domain the pressure difference is made
  mean-free on that same domain before the modular is evaluated.
- For backwards-compatible plotting, `L2`, `H1`, and `Gamma` are aliases for
  `omega_L2`, `omega_H1`, and `omega_Gamma`.

For reproducible figures, regenerate `errors.pkl` and the XML snapshot data
with the commands above.

## Notes

- The outer periodic Picard iteration uses `tol_stop=1e-12` and `Kmax=100`, as
  stated in Section 7.
- The inner semi-implicit nonlinear iteration uses absolute tolerance `1e-8`.
- The plotted pressure/Gamma error is `omega_Gamma`, i.e. the square root of
  the modular applied to the pressure-potential difference after subtracting
  its mean on `omega`.
- In the plot legend, the notation
  `\overline{q}^{\,\omega}` denotes the mean-free part
  `q - \langle q\rangle_\omega`.
- Section 7.3 is computationally much heavier than Sections 7.1 and 7.2.
