# Gaussian Process Augmentation of DNS

Private research repository: **plain Dynamic Nelson–Siegel (Kalman) + Gaussian Process residual augmentation** for Treasury yield forecasting.

This tree is the curated **main spine** from the DNS GP Project `Deliverables Folder` (stages 01–07).

# DNS GP Project — Deliverables (trimmed)

Curated scripts and outputs for the **fixed plain-DNS → GP augmentation** pipeline. Macro-DNS fits, exploratory MAP runs, rolling-DNS pilots, and duplicate Kalman bundles are **not** included.

**Layout:** numbered `scripts/` and `outputs/` subfolders follow the same stage order.

---

## Pipeline overview

| Stage | `scripts/` | `outputs/` |
|-------|------------|------------|
| 01 — WRDS ZCB collection | `01_data_collection_wrds_zcb/` | Rebuilt DNS panels; WRDS raw exports (symlinks) |
| 02 — Macro panel | `02_macro_panel/` | `master_macro_dns_panel.csv` (yields + macro for DNS/GP) |
| 03 — Kalman **plain DNS only** | `03_kalman_filter_plain_dns/` | Fixed DNS parameters used by GP (`plain_dns_kalman_fit/`) |
| 04 — Stage 1 selection | `04_stage1_kernel_input_selection/` | Kernel / input / λ_corr selection |
| 05 — Stage 2 selection | `05_stage2_hyperprior_shrinkage/` | Hyperprior regime + λ_corr selection |
| 06 — MAP testing (final config only) | `06_map_testing/` | 2004–2023 MAP OOS with Stage 1+2 winners |
| 07 — Full Bayesian testing | `07_full_bayesian_testing/` | Factor-specific full Bayes OOS |

---

## Selected model configuration

**Stage 1** (`outputs/04_stage1_kernel_input_selection/stage1_outputs/stage1_selected_structure.csv`):

| Setting | Value |
|---------|--------|
| Kernel | RBF |
| Inputs | 7D |
| λ_corr (validation) | 0.5 |

**Stage 2** (`outputs/05_stage2_hyperprior_shrinkage/stage2_outputs/stage2_selected_prior.csv`):

| Setting | Value |
|---------|--------|
| Amplitude / noise prior | conservative_amp_noise |
| Time prior | long_time |
| Dimension prior | moderate_dims |
| λ_corr (final MAP) | **0.25** |

**Final MAP OOS** (Stage 2 config): `outputs/06_map_testing/final_h1_rbf7d_selected_prior_map_oos_test_outputs/`

Scripts: `final_h1_rbf7d_selected_prior_map_oos_test.py` plus shared helpers `evaluate_fixed_dns_gp.py`, `plain_dns_gp_correction.py`.

---

## Plain DNS used by GP augmentation

The GP stack reads fixed parameters from:

`outputs/03_kalman_filter_plain_dns/plain_dns_kalman_fit/dns_fitted_params_labeled.csv`

Produced by `scripts/03_kalman_filter_plain_dns/fit_dns_and_macro_dns_updated.py` on `master_macro_dns_panel.csv` (train window 1972–2003). That script *can* also fit Macro-DNS when macro columns are present; **only plain DNS outputs** are packaged here.

At run time, Stage 1/2/MAP scripts forward-filter yields with these fixed Φ, Q, H (yield-implied β targets; `dns_beta_residuals.csv` is not used).

---

## WRDS raw data (symlinks)

Large CRSP Treasury exports are **symlinks** in `outputs/01_data_collection_wrds_zcb/` (if present):

- `wrds_crsp_treasury_daily_export.csv` → `Yield/qz4vtnncykho8sgm.csv`
- `wrds_crsp_treasury_monthly_export.csv` → `Yield/h3ktxzxmldfnwnbs.csv`

Re-link or copy if you move this folder off the original machine.

---

## Dependencies

- **Kalman DNS:** `Dynamic_Nelson_Siegel_Svensson_Kalman_Filter` ([GitHub](https://github.com/werleycordeiro/Dynamic_Nelson_Siegel_Svensson_Kalman_Filter))
- **GP stack:** NumPy, SciPy, pandas

Scripts use paths relative to the main project layout (`Kalman Filter/Original Macro + DNS Filter/…`, `Macro/master_macro_dns_panel.csv`). To re-run from this deliverables folder alone, point those paths at the copies under `outputs/02_macro_panel/` and `outputs/03_kalman_filter_plain_dns/plain_dns_kalman_fit/`.

---

## Removed from this package (available in main repo)

- Macro-DNS Kalman scripts and all `macro_dns_*` outputs
- Kalman test bundles, DNS comparison runs, `dns_macro_dns_outputs/`
- Rolling DNS core (`evaluate_rolling_dns_core_for_gp.py`) and beta caches
- Exploratory MAP: pass1, kernel comparisons, prior sweeps, transition pipeline, plain-DNS GP legacy
- Older full Bayes bundle (`full_bayes_dns_gp_posterior_predictive.py` / `full_bayes_dns_gp_outputs/`)
- Stage 1/2 resume checkpoint pickles (`_checkpoints/*.pkl`)

---

## Canonical artifacts

| Artifact | Location |
|----------|----------|
| Master yield + macro panel | `outputs/02_macro_panel/master_macro_dns_panel.csv` |
| Fixed plain DNS parameters (GP input) | `outputs/03_kalman_filter_plain_dns/plain_dns_kalman_fit/dns_fitted_params_labeled.csv` |
| Model-ready ZCB panel (percent) | `outputs/01_data_collection_wrds_zcb/dns_rebuild_output/dns_panel_model_ready_percent_1972_present.csv` |
| Stage 1 winner | `outputs/04_stage1_kernel_input_selection/stage1_outputs/stage1_selected_structure.csv` |
| Stage 2 winner | `outputs/05_stage2_hyperprior_shrinkage/stage2_outputs/stage2_selected_prior.csv` |
| Final MAP OOS | `outputs/06_map_testing/final_h1_rbf7d_selected_prior_map_oos_test_outputs/` |
| Full Bayes OOS | `outputs/07_full_bayesian_testing/final_h1_full_bayes_oos_outputs/` |

**Note:** Full Bayes OOS (`final_h1_factor_specific_rbf7d_full_bayes_oos.py`) uses λ_corr = 0.50 in its saved run; final MAP uses λ_corr = 0.25 from Stage 2.
