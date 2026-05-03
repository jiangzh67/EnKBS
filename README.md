# EnKBS — Code

MATLAB code accompanying the paper:

> *A Continuous-Time Ensemble Kalman–Bucy Smoother for Causal Inference and Model Discovery in Partially Observed Systems*
> Zhang Jiang, Marios Andreou, Sebastian Reich，Nan Chen.


The scripts reproduce the three numerical experiments reported in the paper:

1. Assimilative causal inference on the noisy nonlinear dyad model.
2. Filtering and smoothing with localization and inflation on the 40-dimensional, noisy and partially observed Lorenz-96 system.
3. Causality-based model discovery on the Lorenz-84 model with one
   hidden component.

Together these scripts reproduce all numerical results and figures reported in the paper.

## Requirements

- MATLAB R2021a or later. No toolboxes are required.
  `L84.mlx` is a Live Script and needs the desktop MATLAB.

## File map and figures

### `dyad/` — assimilative causal inference (ACI / CIR)

The two scripts `utov.m` and `vtou.m` correspond to the two causal directions
in the dyad model:

- `utov.m` the $u \rightarrow v$ direction: $v$ is observed,
  $u$ is hidden, produces the assimilative causal inference metric
(ACI) and the objective causal influence range (CIR) using the ensemble Kalman–Bucy smoother (EnKBS).
- `vtou.m` the $v \rightarrow u$ direction: $u$ is observed,
  $v$ is hidden. It runs the analytic CGNS filter/smoother as a reference
  and the EnKBF/EnKBS on the same path, then reproduces the same
  ACI, and CIR diagnostics for both methods.

Output figures: truth time series, CIR curve, and ACI curve over
time. These are the panels combined into the dyad ACI/CIR figure in the
paper.

- `rmse.m` — Generates the truth path, runs the analytic
  CGNS filter/smoother once for the optimal-RMSE baseline, then loops
  `run_enkf_enks_dyad` over the ensemble sizes
  `m = 4, 5, 6, 8, 10, 15, 20, 50, 100, 200, 500, 1000, 2000`. Produces
  the RMSE-vs-ensemble-size figure.

Helpers:

- `run_enkf_enks_dyad.m` — single-run helper that returns the filter and
  smoother RMSE for one ensemble size. It is called by `rmse.m` to scan
  ensemble size.

- `simps.m` — Simpson numerical integration. Used inside `utov.m` and
  `vtou.m` in the CIR definition.


### `enkbs_lorenz96.m` — Lorenz-96 experiment

Standalone script. 

The script:

1. Integrates the reference paths of Lorenz-96.
2. Plots the spatiotemporal heatmap of the truth on the last 25 time units.
3. Runs the EnKBF forward and the EnKBS backward with localization (radius `r0 = 3`) and multiplicative inflation (`delta = sqrt(1.005)`). 
4. Plots the truth, filter mean with $\pm 2\sigma$ band, and smoother mean
   with $\pm 2\sigma$ band for $x_1$ on the window $t \in [75, 100]$.
5. Plots the filter and smoother standard deviations of $x_1$ on
   $t \in [90, 100]$.
6. Reports the spatially and temporally averaged RMSE on $t \in [20, 100]$.



### `model discovery/` — Lorenz-84 model discovery

Driver script: `L84.mlx`. Helpers in this folder:

- `L84_EnKBS.m` — EnKBF + EnKBS for the hidden $x$ component, with one
  trajectory drawn from the smoother as the conditional sample.
- `L84_CGNS_SmootherSampling.m` — analytic conditional Gaussian
  filter/smoother + one conditional sample. Used as the EnKBS-free
  baseline.
- `L84_theta_from_truth.m` — encodes the true Lorenz-84 system in the
  36-dimensional library coefficient vector `Theta_truth`. Used to grade
  the discovered model.
- `L84_sim_theta_same_noise.m` — re-simulates the Lorenz-84 system from
  any candidate `Theta` using the same Brownian path as the truth, so
  that trajectories, PDFs, and ACFs are directly comparable.
- `l84_causation_entropy_mask.m` — causation-entropy computation across
  the 11 non-constant library terms for each of the three equations,
  thresholded at `thrCE` to produce the structural mask.
- `l84_estimate_theta_sigma.m` — constrained weighted least squares for
  the active library coefficients and observation noise variances
  $\sigma_y, \sigma_z$, given a smoothed/sampled hidden path. Linear
  equality constraints `H * Theta = g` enforce the physical structure
  reflected in the Lorenz-84 system.
- `acf_formula21.m` — autocorrelation helper used in the diagnostics
  figure when MATLAB's `autocorr` is not available.

#### Switching between EnKBS sampling and CGNS sampling

Inside `L84.mlx`, the iterative model-discovery loop calls one of the two
hidden-state samplers. The default is the EnKBS:

```matlab
% in the iteration loop
[~, ~, ~, ~, xhat] = L84_EnKBS( ...
     dt, N, m_ens, sigma_x^2, sigma_y^2, sigma_z^2, y_obs, z_obs, Theta_k);
% [~, ~, ~, ~, xhat] = L84_CGNS_SmootherSampling( ...
%      dt, N, m_ens, sigma_x^2, sigma_y^2, sigma_z^2, y_obs, z_obs, Theta_k);
```

To switch to the CGNS sampler, comment the `L84_EnKBS` call and uncomment
the `L84_CGNS_SmootherSampling` call.

The thresholds and the iteration budget are set near the top of the loop
section: `thrCE` is the causation-entropy cutoff used by
`l84_causation_entropy_mask`, `maxOuterIter` is the outer-loop budget,
and `m_ens` is the ensemble size used by the EnKBS sampler. The constraint
matrix `H` and right-hand side `g` are built in the cell preceding the
loop and encode the physical structure of the Lorenz-84 library.

#### Figures produced by `L84.mlx`

Run the Live Script top to bottom. The figure cells produce, in order:

1. Truth time series of $x$, $y$, $z$ on $t \in [0, 50]$.
2. Hidden-state estimate $\hat x(t)$ at selected outer iterations, the Frobenius distance between the
   current and stable causation matrices, and the parameter $b$ as afunction of iteration.
3. Truth-vs-identified trajectories, marginal PDFs, and
   autocorrelation functions for each of $x$, $y$, $z$.

A printed comparison table of true-vs-identified library coefficients
appears in the script output.


Random seeds are fixed in each script for reproducibility.

## Citation

If you use this code, please cite the paper above.

## License

Released under the MIT License. See `LICENSE`.
