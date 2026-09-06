# Plain DNS vs GP residual corrections — pooled yield metrics

**Generated:** consolidated from existing `pooled_yield_metrics_bp.csv` outputs under `Gaussian Processes/`.

**Metric definitions**

- **Pooled RMSE / MAE (bp):** same definitions as the neural-network DNS paper style runs — errors are yield prediction minus actual in **decimal**, then scaled to **basis points** (multiply by 10,000); pooling is over all test forecast origins and maturities in the named set.
- **Plain DNS:** no residual correction; \(\hat{y} = \Lambda(\lambda)\,\beta^{\text{DNS}}_{t+h|t}\).
- **DNS reference columns** below match **`fixed_dns_gp_pass1_outputs/pooled_yield_metrics_bp.csv`** (identical DNS rows across runs on this panel).

**Maturity sets**

- **neural_13:** \(\{3,6,9,12,24,36,48,60,72,84,96,108,120\}\) months.
- **project_17:** \(\{3,6,9,12,15,18,21,24,30,36,48,60,72,84,96,108,120\}\) months.

**Horizons:** \(h \in \{1,3,6,12\}\) months (fixed pre-2004 DNS; train targets through 2003-12; test origins 2004-01–2023-12 per each evaluation script).

---

## neural_13 — pooled RMSE (bp)

| horizon | DNS | GP_MAP | GP_BAYES | MAP_GP_SIMPLE | GP_PASS1_V2 | GP_RBF_ARD_10D | GP_RBF_ARD_19D | GP_RBF_ARD_19D_SHRINK05 |
|---|---|---|---|---|---|---|---|---|
| 1 | 28.11 | 31.11 | 29.95 | 31.29 | 31.34 | 27.87 | 28.27 | 28.19 |
| 3 | 54.02 | 52.17 | 52.40 | 52.16 | 135.55 | 53.02 | 54.48 | 54.25 |
| 6 | 84.12 | 84.30 | 84.81 | 84.30 | 178.42 | 86.07 | 86.05 | 85.02 |
| 12 | 132.24 | 136.54 | 136.62 | 136.49 | 559.16 | 136.37 | 136.03 | 134.08 |

## neural_13 — pooled MAE (bp)

| horizon | DNS | GP_MAP | GP_BAYES | MAP_GP_SIMPLE | GP_PASS1_V2 | GP_RBF_ARD_10D | GP_RBF_ARD_19D | GP_RBF_ARD_19D_SHRINK05 |
|---|---|---|---|---|---|---|---|---|
| 1 | 20.52 | 23.13 | 22.30 | 23.24 | 24.26 | 20.24 | 20.73 | 20.63 |
| 3 | 41.03 | 39.09 | 39.23 | 39.08 | 103.40 | 39.57 | 41.64 | 41.33 |
| 6 | 65.62 | 67.23 | 67.83 | 67.20 | 153.29 | 67.95 | 67.90 | 66.72 |
| 12 | 107.49 | 112.88 | 112.97 | 112.77 | 535.63 | 111.76 | 111.31 | 109.35 |

## project_17 — pooled RMSE (bp)

| horizon | DNS | GP_MAP | GP_BAYES | MAP_GP_SIMPLE | GP_PASS1_V2 | GP_RBF_ARD_10D | GP_RBF_ARD_19D | GP_RBF_ARD_19D_SHRINK05 |
|---|---|---|---|---|---|---|---|---|
| 1 | 27.46 | 31.48 | 30.05 | 31.66 | 31.68 | 27.21 | 27.59 | 27.52 |
| 3 | 54.06 | 51.44 | 51.67 | 51.44 | 141.07 | 53.08 | 54.50 | 54.27 |
| 6 | 85.16 | 84.54 | 84.98 | 84.54 | 181.09 | 87.12 | 87.14 | 86.08 |
| 12 | 134.84 | 138.31 | 138.38 | 138.27 | 560.12 | 139.05 | 138.67 | 136.70 |

## project_17 — pooled MAE (bp)

| horizon | DNS | GP_MAP | GP_BAYES | MAP_GP_SIMPLE | GP_PASS1_V2 | GP_RBF_ARD_10D | GP_RBF_ARD_19D | GP_RBF_ARD_19D_SHRINK05 |
|---|---|---|---|---|---|---|---|---|
| 1 | 19.74 | 23.43 | 22.31 | 23.53 | 24.81 | 19.46 | 19.93 | 19.84 |
| 3 | 40.91 | 37.82 | 37.94 | 37.81 | 108.91 | 39.07 | 41.52 | 41.21 |
| 6 | 66.40 | 67.07 | 67.57 | 67.03 | 155.55 | 68.78 | 68.85 | 67.58 |
| 12 | 110.16 | 114.62 | 114.72 | 114.50 | 535.84 | 114.56 | 114.05 | 112.06 |

*(Values rounded to two decimals for display; underlying CSVs carry full precision.)*

---

## GP configuration reference

| ID (column) | Reported model name in CSV | Source output folder | What it is |
|-------------|----------------------------|----------------------|------------|
| **GP_MAP** | `DNS+GP_MAP` | `fixed_dns_gp_clean_outputs/` | **MAP** fit of `JointPlainDNSGP` in `plain_dns_gp_correction.py`: **Matérn 5/2 ARD** on **3D** input \(\beta_t = (L,S,C)\) only; joint multi-output GP with shared amplitude / per-factor lengthscales and noise; residuals are \(h\)-step DNS factor residuals (same protocol as other rows). |
| **GP_BAYES** | `DNS+GP_BAYES` | `fixed_dns_gp_clean_outputs/` | Same kernel / feature map as **GP_MAP**, but **adaptive Metropolis** approximate Bayesian GP (when enabled in that run); correction is **posterior predictive mean** of residuals, not a single MAP point. |
| **MAP_GP_SIMPLE** | `DNS+MAP_GP_SIMPLE` | `fixed_dns_gp_pass1_outputs/` | Same **`JointPlainDNSGP`** class as **GP_MAP** (Matérn 5/2 ARD on **\(\beta_t\)** only), evaluated inside **`evaluate_fixed_dns_gp_pass1.py`** so it shares **exactly** the Pass 1 train/test split and \(h\)-step residual targets. (Other scripts also embed this baseline; numbers here are from the Pass 1 bundle.) |
| **GP_PASS1_V2** | `DNS+GP_PASS1_V2` | `fixed_dns_gp_pass1_outputs/` | **Pass 1 “v2”**: **19D** standardized memory features (levels, lags, diffs, second diffs, time); **three independent single-output GPs**; kernel = **linear** (on 18 “shape” dims) **+** **Matérn 3/2** block-ARD over six 3-blocks **+** **separate Matérn 3/2 on calendar time**; **L-BFGS-B MAP** with analytic gradient. |
| **GP_RBF_ARD_10D** | `DNS+GP_RBF_ARD_10D` | `fixed_dns_gp_rbf_ard_10d_outputs/` | **10D** standardized block: \(\beta_t-\mu\), \(\beta_{t-1}-\mu\), \(\beta_t-\beta_{t-1}\), and **time**; **single RBF-ARD** kernel (Gaussian) with one lengthscale per dimension; MAP (script `evaluate_fixed_dns_gp_rbf_ard_10d.py`). |
| **GP_RBF_ARD_19D** | `DNS+GP_RBF_ARD_19D` | `fixed_dns_gp_rbf_ard_19d_outputs/` | **19D** standardized block (full memory vector: three levels, diffs, second diffs, time) with **time inside the joint RBF-ARD** (no separate additive time kernel); **single RBF-ARD** per factor; conservative Normal priors on logs; MAP (`evaluate_fixed_dns_gp_rbf_ard_19d.py`). |
| **GP_RBF_ARD_19D_SHRINK05** | `DNS+GP_RBF_ARD_19D_SHRINK05` | `fixed_dns_gp_rbf_ard_19d_outputs/` | Same fitted GP mean correction as **GP_RBF_ARD_19D**, but applied with factor **0.5** to the residual correction (\(\lambda_{\text{corr}}=0.5\)) before mapping to yields — a conservative sensitivity run. |

**Shared experimental backbone (all rows):** fixed DNS from `Kalman Filter/Original Macro + DNS Filter/dns_fitted_params_labeled.csv`; Kalman **filtered** \(\beta_t\) in decimal; **no** `dns_beta_residuals.csv`; DNS **not** refit inside these scripts.

---

## Tidy long format (TSV)

*Copy from the code block below into a `.tsv` file if you want to pivot or join in Excel/R.*

```text
config_id	metrics_source	maturity_set	horizon	DNS_rmse_bp	DNS_mae_bp	GP_rmse_bp	GP_mae_bp
GP_BAYES	fixed_dns_gp_clean_outputs	neural_13	1	28.107373670472917	20.520949627170396	29.95494040721886	22.298990771111452
GP_BAYES	fixed_dns_gp_clean_outputs	neural_13	3	54.02237295031276	41.03440397018366	52.401821778394904	39.22869432746234
GP_BAYES	fixed_dns_gp_clean_outputs	neural_13	6	84.11703930159551	65.61803361640013	84.80540128327948	67.82678583273079
GP_BAYES	fixed_dns_gp_clean_outputs	neural_13	12	132.23666309844813	107.48800237924614	136.61862369008003	112.96937123504897
GP_MAP	fixed_dns_gp_clean_outputs	neural_13	1	28.107373670472917	20.520949627170396	31.112998386252233	23.13135411425098
GP_MAP	fixed_dns_gp_clean_outputs	neural_13	3	54.02237295031276	41.03440397018366	52.16560966094568	39.089435980856926
GP_MAP	fixed_dns_gp_clean_outputs	neural_13	6	84.11703930159551	65.61803361640013	84.30159711166492	67.2263105381172
GP_MAP	fixed_dns_gp_clean_outputs	neural_13	12	132.23666309844813	107.48800237924614	136.53586244036111	112.8810842858682
GP_PASS1_V2	fixed_dns_gp_pass1_outputs	neural_13	1	28.107373670472917	20.520949627170396	31.335043008025522	24.259880712423023
GP_PASS1_V2	fixed_dns_gp_pass1_outputs	neural_13	3	54.02237295031276	41.03440397018366	135.54840634147592	103.39953988612136
GP_PASS1_V2	fixed_dns_gp_pass1_outputs	neural_13	6	84.11703930159551	65.61803361640013	178.41664058658327	153.28660729898448
GP_PASS1_V2	fixed_dns_gp_pass1_outputs	neural_13	12	132.23666309844813	107.48800237924614	559.1593415296807	535.6277075985462
GP_RBF_ARD_10D	fixed_dns_gp_rbf_ard_10d_outputs	neural_13	1	28.107373670472917	20.520949627170396	27.868087641091083	20.243044120861722
GP_RBF_ARD_10D	fixed_dns_gp_rbf_ard_10d_outputs	neural_13	3	54.02237295031276	41.03440397018366	53.02162881170955	39.57172203961729
GP_RBF_ARD_10D	fixed_dns_gp_rbf_ard_10d_outputs	neural_13	6	84.11703930159551	65.61803361640013	86.07435367122233	67.94907859094585
GP_RBF_ARD_10D	fixed_dns_gp_rbf_ard_10d_outputs	neural_13	12	132.23666309844813	107.48800237924614	136.36743113807174	111.7553101499969
GP_RBF_ARD_19D	fixed_dns_gp_rbf_ard_19d_outputs	neural_13	1	28.107373670472917	20.520949627170396	28.265888115853553	20.732492598442594
GP_RBF_ARD_19D	fixed_dns_gp_rbf_ard_19d_outputs	neural_13	3	54.02237295031276	41.03440397018366	54.480838631814976	41.63729494793008
GP_RBF_ARD_19D	fixed_dns_gp_rbf_ard_19d_outputs	neural_13	6	84.11703930159551	65.61803361640013	86.0536782022625	67.90273367702271
GP_RBF_ARD_19D	fixed_dns_gp_rbf_ard_19d_outputs	neural_13	12	132.23666309844813	107.48800237924614	136.02730209326648	111.31237226843314
GP_RBF_ARD_19D_SHRINK05	fixed_dns_gp_rbf_ard_19d_outputs	neural_13	1	28.107373670472917	20.520949627170396	28.185014889728215	20.625325913326307
GP_RBF_ARD_19D_SHRINK05	fixed_dns_gp_rbf_ard_19d_outputs	neural_13	3	54.02237295031276	41.03440397018366	54.24585865921629	41.329617992240856
GP_RBF_ARD_19D_SHRINK05	fixed_dns_gp_rbf_ard_19d_outputs	neural_13	6	84.11703930159551	65.61803361640013	85.0194389677335	66.72204746206388
GP_RBF_ARD_19D_SHRINK05	fixed_dns_gp_rbf_ard_19d_outputs	neural_13	12	132.23666309844813	107.48800237924614	134.08295997118884	109.35401228258398
MAP_GP_SIMPLE	fixed_dns_gp_pass1_outputs	neural_13	1	28.107373670472917	20.520949627170396	31.29412488881471	23.235539109108625
MAP_GP_SIMPLE	fixed_dns_gp_pass1_outputs	neural_13	3	54.02237295031276	41.03440397018366	52.164122328332766	39.08050291552929
MAP_GP_SIMPLE	fixed_dns_gp_pass1_outputs	neural_13	6	84.11703930159551	65.61803361640013	84.30292067005004	67.20311304615012
MAP_GP_SIMPLE	fixed_dns_gp_pass1_outputs	neural_13	12	132.23666309844813	107.48800237924614	136.49248389315125	112.77275406238812
GP_BAYES	fixed_dns_gp_clean_outputs	project_17	1	27.456211711754605	19.743924975959853	30.054376376236018	22.31094255782405
GP_BAYES	fixed_dns_gp_clean_outputs	project_17	3	54.061083184513166	40.9056170803028	51.67391566385085	37.94218131182699
GP_BAYES	fixed_dns_gp_clean_outputs	project_17	6	85.15943489725461	66.39977850048368	84.97515478463038	67.56935100657736
GP_BAYES	fixed_dns_gp_clean_outputs	project_17	12	134.83905236131943	110.16313789882047	138.38104374513352	114.72466360091931
GP_MAP	fixed_dns_gp_clean_outputs	project_17	1	27.456211711754605	19.743924975959853	31.482476176949262	23.432655888323193
GP_MAP	fixed_dns_gp_clean_outputs	project_17	3	54.061083184513166	40.9056170803028	51.44023425369155	37.824346231309555
GP_MAP	fixed_dns_gp_clean_outputs	project_17	6	85.15943489725461	66.39977850048368	84.53947050660278	67.07190416487724
GP_MAP	fixed_dns_gp_clean_outputs	project_17	12	134.83905236131943	110.16313789882047	138.3142944658345	114.62475463135755
GP_PASS1_V2	fixed_dns_gp_pass1_outputs	project_17	1	27.456211711754605	19.743924975959853	31.682556555890624	24.81205169359786
GP_PASS1_V2	fixed_dns_gp_pass1_outputs	project_17	3	54.061083184513166	40.9056170803028	141.07208507624227	108.91271288930037
GP_PASS1_V2	fixed_dns_gp_pass1_outputs	project_17	6	85.15943489725461	66.39977850048368	181.09128245105043	155.55362791155616
GP_PASS1_V2	fixed_dns_gp_pass1_outputs	project_17	12	134.83905236131943	110.16313789882047	560.1244178965758	535.838096937686
GP_RBF_ARD_10D	fixed_dns_gp_rbf_ard_10d_outputs	project_17	1	27.456211711754605	19.743924975959853	27.207739397795255	19.456383474934412
GP_RBF_ARD_10D	fixed_dns_gp_rbf_ard_10d_outputs	project_17	3	54.061083184513166	40.9056170803028	53.07887885336667	39.066601531114934
GP_RBF_ARD_10D	fixed_dns_gp_rbf_ard_10d_outputs	project_17	6	85.15943489725461	66.39977850048368	87.11513050678818	68.77839051797625
GP_RBF_ARD_10D	fixed_dns_gp_rbf_ard_10d_outputs	project_17	12	134.83905236131943	110.16313789882047	139.04891644454574	114.5642902342347
GP_RBF_ARD_19D	fixed_dns_gp_rbf_ard_19d_outputs	project_17	1	27.456211711754605	19.743924975959853	27.591771958330252	19.93362262959766
GP_RBF_ARD_19D	fixed_dns_gp_rbf_ard_19d_outputs	project_17	3	54.061083184513166	40.9056170803028	54.495003538180356	41.52027950321851
GP_RBF_ARD_19D	fixed_dns_gp_rbf_ard_19d_outputs	project_17	6	85.15943489725461	66.39977850048368	87.14383773441865	68.8453029547483
GP_RBF_ARD_19D	fixed_dns_gp_rbf_ard_19d_outputs	project_17	12	134.83905236131943	110.16313789882047	138.66532812993415	114.05171639214305
GP_RBF_ARD_19D_SHRINK05	fixed_dns_gp_rbf_ard_19d_outputs	project_17	1	27.456211711754605	19.743924975959853	27.522201041201235	19.83688937217433
GP_RBF_ARD_19D_SHRINK05	fixed_dns_gp_rbf_ard_19d_outputs	project_17	3	54.061083184513166	40.9056170803028	54.27211261925755	41.20750872217908
GP_RBF_ARD_19D_SHRINK05	fixed_dns_gp_rbf_ard_19d_outputs	project_17	6	85.15943489725461	66.39977850048368	86.08356259107026	67.58403169771364
GP_RBF_ARD_19D_SHRINK05	fixed_dns_gp_rbf_ard_19d_outputs	project_17	12	134.83905236131943	110.16313789882047	136.70181512149668	112.05909633963188
MAP_GP_SIMPLE	fixed_dns_gp_pass1_outputs	project_17	1	27.456211711754605	19.743924975959853	31.66451543912582	23.529046245210154
MAP_GP_SIMPLE	fixed_dns_gp_pass1_outputs	project_17	3	54.061083184513166	40.9056170803028	51.44172277551865	37.80667303952384
MAP_GP_SIMPLE	fixed_dns_gp_pass1_outputs	project_17	6	85.15943489725461	66.39977850048368	84.53804395042454	67.03410245677732
MAP_GP_SIMPLE	fixed_dns_gp_pass1_outputs	project_17	12	134.83905236131943	110.16313789882047	138.26561375557773	114.50370959908822
```
