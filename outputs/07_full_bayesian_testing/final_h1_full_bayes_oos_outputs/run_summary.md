# h=1 factor-specific RBF-ARD 7D — full Bayes hyperparameter integration

- **Horizon:** h=1 only.
- **DNS:** fixed from `dns_fitted_params_labeled.csv`; **not** re-estimated.
- **Broken residual file** `dns_beta_residuals.csv`: **not** used.
- **Targets:** one-step **yield-implied** beta correction (W=I, ridge=1e-8). **No** Kalman beta residuals.
- **Kernel:** RBF-ARD only; **7D** inputs; **separate** scalar GPs for L, S, C.
- **Prior:** Moderate-long-time (explicit normal priors on logs; Step 8 specification).
- **Shrinkage:** lambda_corr = 0.5.
- **Train:** target_month ≤ 2003-12 (scaling, MAP init, MCMC use **training rows only**).
- **Test:** forecast origins 2004-01 .. 2023-12 (same convention as other fixed-DNS runs).
- **Sampler:** adaptive random-walk Metropolis–Hastings in log-hyperparameter space (initialized at MAP).
- **MCMC:** chains=2, warmup=400, draws=400, thin=1.

## Pooled OOS (bp)

| maturity_set | DNS RMSE | Full Bayes RMSE | Δ RMSE | beats DNS RMSE |
|---|---:|---:|---:|:---:|
| neural_13 | 28.1074 | 26.7672 | 1.3402 | True |
| project_17 | 27.4562 | 26.2848 | 1.1715 | True |

## Runtime

- elapsed_s: 930.3
