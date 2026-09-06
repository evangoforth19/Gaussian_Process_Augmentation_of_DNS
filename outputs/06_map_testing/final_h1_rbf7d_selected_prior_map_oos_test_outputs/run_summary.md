# Final OOS test (2004–2023): MAP RBF-7D GP correction

- **Final** one-month-ahead MAP OOS evaluation; **no** validation or selection in this script.
- **Selected Stage 2:** RBF, 7D, `conservative_amp_noise`, `long_time`, `moderate_dims`, `lambda_corr=0.25`.
- **Test window:** target months **2004-01** … **2023-12**; **train** targets **≤ 2003-12** only for GP fit and scalers.
- **DNS** parameters fixed; **not** refit; `dns_beta_residuals.csv` **not** used.
- **One-step yield-implied** beta correction targets; vector GP \(g:\mathbb{R}^7\to\mathbb{R}^3\).

## Pooled test metrics (neural_13)

- DNS: RMSE **28.0566** bp, MSE **787.1745** bp², MAE **20.4621** bp.
- GP: RMSE **28.6878** bp, MSE **822.9921** bp², MAE **20.9755** bp.
- Improvements vs DNS (RMSE / MSE / MAE): **-0.6312** / **-35.8176** / **-0.5133**.

## Pooled test metrics (project_17)

- DNS: RMSE **27.4146** bp; GP: RMSE **28.1584** bp.

## Baselines (neural_13 RMSE bp)

- CONST: **27.9655**; AR1: **27.9801**.

## Diebold–Mariano (GP vs DNS, neural_13, HAC lag 0)

- Squared loss: mean diff **-35.817614**, DM stat **-4.3786**, one-sided *p* **1.0000**.
- Absolute loss: DM stat **-4.4554**, one-sided *p* **1.0000**.

## Class-paper interpretation

Positive RMSE/MSE improvement and small one-sided *p*-values support **GP beating DNS** on average one-month-ahead yield error at the pooled maturity sets, subject to the usual caveats (stationarity, subperiod stability, and the fixed-DNS + MAP simplification).
