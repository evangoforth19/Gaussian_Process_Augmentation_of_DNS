# Stage 1: single GP kernel / input selection (h=1)

- **Stage 1 structural validation only** (pre-2004 rolling folds). **No 2004–2023 test** data used.
- **One-step yield-implied beta correction targets** only; **no Kalman beta residuals**; `dns_beta_residuals.csv` **not read**.
- **Single vector-valued GP** \(g:\mathbb{R}^D\to\mathbb{R}^3\): shared input kernel (ARD lengthscales), **joint MAP** over shared \(\ell\) (+ RQ shape if RQ) and output-specific \(\alpha_j,\sigma_j\).
- **Kernels**: RBF-ARD, RQ-ARD. **Inputs**: 4D and 7D only (**10D not implemented**).
- **Fixed prior**: Moderate-long-time (see script constants).
- **Shrinkage grid**: `lambda_corr` in {0.50, 1.00} (post hoc on the same validation correction vector; **no** extra joint MAP per λ).
- **Jitter search**: default escalates `1e-6 → …` only when optimization/Cholesky pathologies yield a non-finite or penalty-level objective; `STAGE1_EXHAUSTIVE_JITTER=1` runs the full ladder every restart.
- **Checkpointing**: `_checkpoints/fold{…}_{KERNEL}_{input}.pkl` after each (fold, kernel, input); skipped on rerun unless `STAGE1_FORCE_RECOMPUTE=1`.
- **Folds**: (1) val targets 1990–01..1994–12; (2) 1995–01..1999–12; (3) 2000–01..2003–12; training uses all one-step rows with target ≤ train cut.
- **Selection**: minimize **pooled** validation RMSE on **neural_13** with \(S_{\mathrm{val}}=\sqrt{\sum e^2/\sum n}\) across folds and pooled errors.

## Selected structure

- Kernel: **RBF**, input: **7D** (D=7), `lambda_corr`=0.5.
- Pooled neural_13 RMSE \(S_{\mathrm{val}}\): **29.3372** bp; mean fold MAE (neural_13): **22.5221** bp.
- Project_17 (selected config, RMS of fold RMSEs): **29.7975** bp; mean MAE: **22.9315** bp.

## GP validation table (excerpt)

    fold_id kernel input_set  input_dim  lambda_corr maturity_set  val_rmse_bp  val_mae_bp  n_val_origins  n_maturities  n_errors  optimizer_success_all_outputs pathology_flag  final_jitter_max
0         1    RBF        4D          4          0.5    neural_13    31.449687   25.332642             60            13       780                           True           none          0.000001
1         1    RBF        4D          4          0.5   project_17    31.999729   26.090251             60            17      1020                           True           none          0.000001
2         1    RBF        4D          4          1.0    neural_13    31.864037   25.592993             60            13       780                           True           none          0.000001
3         1    RBF        4D          4          1.0   project_17    32.444907   26.385867             60            17      1020                           True           none          0.000001
4         1    RBF        7D          7          0.5    neural_13    31.413042   25.289681             60            13       780                           True           none          0.000001
5         1    RBF        7D          7          0.5   project_17    31.954458   26.038724             60            17      1020                           True           none          0.000001
6         1    RBF        7D          7          1.0    neural_13    31.815455   25.525219             60            13       780                           True           none          0.000001
7         1    RBF        7D          7          1.0   project_17    32.383956   26.305023             60            17      1020                           True           none          0.000001
8         1     RQ        4D          4          0.5    neural_13    31.451634   25.333745             60            13       780                           True           none          0.000001
9         1     RQ        4D          4          0.5   project_17    32.000610   26.089285             60            17      1020                           True           none          0.000001
10        1     RQ        4D          4          1.0    neural_13    31.869082   25.595831             60            13       780                           True           none          0.000001
11        1     RQ        4D          4          1.0   project_17    32.448332   26.385835             60            17      1020                           True           none          0.000001
12        1     RQ        7D          7          0.5    neural_13    31.416873   25.293291             60            13       780                           True           none          0.000001
13        1     RQ        7D          7          0.5   project_17    31.957643   26.040916             60            17      1020                           True           none          0.000001
14        1     RQ        7D          7          1.0    neural_13    31.822435   25.531438             60            13       780                           True           none          0.000001
15        1     RQ        7D          7          1.0   project_17    32.390025   26.309658             60            17      1020                           True           none          0.000001
16        2    RBF        4D          4          0.5    neural_13    24.836921   18.126742             60            13       780                           True           none          0.000001
17        2    RBF        4D          4          0.5   project_17    24.906747   18.315739             60            17      1020                           True           none          0.000001
18        2    RBF        4D          4          1.0    neural_13    24.974811   18.265832             60            13       780                           True           none          0.000001
19        2    RBF        4D          4          1.0   project_17    25.004361   18.402507             60            17      1020                           True           none          0.000001
20        2    RBF        7D          7          0.5    neural_13    24.687605   18.038909             60            13       780                           True           none          0.000001
21        2    RBF        7D          7          0.5   project_17    24.748946   18.237759             60            17      1020                           True           none          0.000001
22        2    RBF        7D          7          1.0    neural_13    24.678058   18.082573             60            13       780                           True           none          0.000001
23        2    RBF        7D          7          1.0   project_17    24.689233   18.236077             60            17      1020                           True           none          0.000001
24        2     RQ        4D          4          0.5    neural_13    24.828993   18.115954             60            13       780                           True           none          0.000001
25        2     RQ        4D          4          0.5   project_17    24.902398   18.310571             60            17      1020                           True           none          0.000001
26        2     RQ        4D          4          1.0    neural_13    24.953568   18.237757             60            13       780                           True           none          0.000001
27        2     RQ        4D          4          1.0   project_17    24.989857   18.385476             60            17      1020                           True           none          0.000001
28        2     RQ        7D          7          0.5    neural_13    24.728513   18.064944             60            13       780                           True           none          0.000001
29        2     RQ        7D          7          0.5   project_17    24.793850   18.264456             60            17      1020                           True           none          0.000001
30        2     RQ        7D          7          1.0    neural_13    24.755891   18.129561             60            13       780                           True           none          0.000001
31        2     RQ        7D          7          1.0   project_17    24.775527   18.287540             60            17      1020                           True           none          0.000001
32        3    RBF        4D          4          0.5    neural_13    32.108023   24.342728             48            13       624                           True           none          0.000001
33        3    RBF        4D          4          0.5   project_17    32.346091   24.566489             48            17       816                           True           none          0.000001
34        3    RBF        4D          4          1.0    neural_13    32.080972   24.333361             48            13       624                           True           none          0.000001
35        3    RBF        4D          4          1.0   project_17    32.355295   24.599210             48            17       816                           True           none          0.000001
36        3    RBF        7D          7          0.5    neural_13    31.890839   24.237561             48            13       624                           True           none          0.000001
37        3    RBF        7D          7          0.5   project_17    32.094780   24.518150             48            17       816                           True           none          0.000001
38        3    RBF        7D          7          1.0    neural_13    31.797563   24.246109             48            13       624                           True           none          0.000001
39        3    RBF        7D          7          1.0   project_17    32.036441   24.629881             48            17       816                           True           none          0.000001
40        3     RQ        4D          4          0.5    neural_13    32.182984   24.413451             48            13       624                           True           none          0.000001
41        3     RQ        4D          4          0.5   project_17    32.434788   24.643769             48            17       816                           True           none          0.000001
42        3     RQ        4D          4          1.0    neural_13    32.228531   24.471718             48            13       624                           True           none          0.000001
43        3     RQ        4D          4          1.0   project_17    32.531049   24.751409             48            17       816                           True           none          0.000001
44        3     RQ        7D          7          0.5    neural_13    31.984698   24.299950             48            13       624                           True           none          0.000001
45        3     RQ        7D          7          0.5   project_17    32.207672   24.568928             48            17       816                           True           none          0.000001
46        3     RQ        7D          7          1.0    neural_13    31.941122   24.344403             48            13       624                           True           none          0.000001
47        3     RQ        7D          7          1.0   project_17    32.210080   24.710760             48            17       816                           True           none          0.000001

## Baseline validation (DNS / CONST / AR1)

    fold_id               model maturity_set  val_rmse_bp  val_mae_bp  n_val_origins  n_maturities  n_errors
0         1                 DNS    neural_13    31.143857   25.132494             60            13       780
1         1                 DNS   project_17    31.703474   25.883659             60            17      1020
2         1  DNS+CONST_ONE_STEP    neural_13    31.648993   25.408533             60            13       780
3         1  DNS+CONST_ONE_STEP   project_17    32.210088   26.176175             60            17      1020
4         1    DNS+AR1_ONE_STEP    neural_13    32.070385   25.839776             60            13       780
5         1    DNS+AR1_ONE_STEP   project_17    32.660365   26.609965             60            17      1020
6         1                 DNS    neural_13    31.143857   25.132494             60            13       780
7         1                 DNS   project_17    31.703474   25.883659             60            17      1020
8         1  DNS+CONST_ONE_STEP    neural_13    31.648993   25.408533             60            13       780
9         1  DNS+CONST_ONE_STEP   project_17    32.210088   26.176175             60            17      1020
10        1    DNS+AR1_ONE_STEP    neural_13    32.070385   25.839776             60            13       780
11        1    DNS+AR1_ONE_STEP   project_17    32.660365   26.609965             60            17      1020
12        1                 DNS    neural_13    31.143857   25.132494             60            13       780
13        1                 DNS   project_17    31.703474   25.883659             60            17      1020
14        1  DNS+CONST_ONE_STEP    neural_13    31.648993   25.408533             60            13       780
15        1  DNS+CONST_ONE_STEP   project_17    32.210088   26.176175             60            17      1020
16        1    DNS+AR1_ONE_STEP    neural_13    32.070385   25.839776             60            13       780
17        1    DNS+AR1_ONE_STEP   project_17    32.660365   26.609965             60            17      1020
18        1                 DNS    neural_13    31.143857   25.132494             60            13       780
19        1                 DNS   project_17    31.703474   25.883659             60            17      1020
20        1  DNS+CONST_ONE_STEP    neural_13    31.648993   25.408533             60            13       780
21        1  DNS+CONST_ONE_STEP   project_17    32.210088   26.176175             60            17      1020
22        1    DNS+AR1_ONE_STEP    neural_13    32.070385   25.839776             60            13       780
23        1    DNS+AR1_ONE_STEP   project_17    32.660365   26.609965             60            17      1020
24        2                 DNS    neural_13    24.747862   18.030246             60            13       780
25        2                 DNS   project_17    24.866748   18.286547             60            17      1020
26        2  DNS+CONST_ONE_STEP    neural_13    24.988177   18.260209             60            13       780
27        2  DNS+CONST_ONE_STEP   project_17    25.018416   18.403044             60            17      1020
28        2    DNS+AR1_ONE_STEP    neural_13    24.983226   18.324245             60            13       780
29        2    DNS+AR1_ONE_STEP   project_17    24.979899   18.377488             60            17      1020
30        2                 DNS    neural_13    24.747862   18.030246             60            13       780
31        2                 DNS   project_17    24.866748   18.286547             60            17      1020
32        2  DNS+CONST_ONE_STEP    neural_13    24.988177   18.260209             60            13       780
33        2  DNS+CONST_ONE_STEP   project_17    25.018416   18.403044             60            17      1020
34        2    DNS+AR1_ONE_STEP    neural_13    24.983226   18.324245             60            13       780
35        2    DNS+AR1_ONE_STEP   project_17    24.979899   18.377488             60            17      1020
36        2                 DNS    neural_13    24.747862   18.030246             60            13       780
37        2                 DNS   project_17    24.866748   18.286547             60            17      1020
38        2  DNS+CONST_ONE_STEP    neural_13    24.988177   18.260209             60            13       780
39        2  DNS+CONST_ONE_STEP   project_17    25.018416   18.403044             60            17      1020
40        2    DNS+AR1_ONE_STEP    neural_13    24.983226   18.324245             60            13       780
41        2    DNS+AR1_ONE_STEP   project_17    24.979899   18.377488             60            17      1020
42        2                 DNS    neural_13    24.747862   18.030246             60            13       780
43        2                 DNS   project_17    24.866748   18.286547             60            17      1020
44        2  DNS+CONST_ONE_STEP    neural_13    24.988177   18.260209             60            13       780
45        2  DNS+CONST_ONE_STEP   project_17    25.018416   18.403044             60            17      1020
46        2    DNS+AR1_ONE_STEP    neural_13    24.983226   18.324245             60            13       780
47        2    DNS+AR1_ONE_STEP   project_17    24.979899   18.377488             60            17      1020
48        3                 DNS    neural_13    32.149563   24.367080             48            13       624
49        3                 DNS   project_17    32.352170   24.548826             48            17       816
50        3  DNS+CONST_ONE_STEP    neural_13    32.673553   24.901298             48            13       624
51        3  DNS+CONST_ONE_STEP   project_17    33.062500   25.223222             48            17       816
52        3    DNS+AR1_ONE_STEP    neural_13    32.449327   24.771030             48            13       624
53        3    DNS+AR1_ONE_STEP   project_17    32.740327   25.035988             48            17       816
54        3                 DNS    neural_13    32.149563   24.367080             48            13       624
55        3                 DNS   project_17    32.352170   24.548826             48            17       816
56        3  DNS+CONST_ONE_STEP    neural_13    32.673553   24.901298             48            13       624
57        3  DNS+CONST_ONE_STEP   project_17    33.062500   25.223222             48            17       816
58        3    DNS+AR1_ONE_STEP    neural_13    32.449327   24.771030             48            13       624
59        3    DNS+AR1_ONE_STEP   project_17    32.740327   25.035988             48            17       816
60        3                 DNS    neural_13    32.149563   24.367080             48            13       624
61        3                 DNS   project_17    32.352170   24.548826             48            17       816
62        3  DNS+CONST_ONE_STEP    neural_13    32.673553   24.901298             48            13       624
63        3  DNS+CONST_ONE_STEP   project_17    33.062500   25.223222             48            17       816
64        3    DNS+AR1_ONE_STEP    neural_13    32.449327   24.771030             48            13       624
65        3    DNS+AR1_ONE_STEP   project_17    32.740327   25.035988             48            17       816
66        3                 DNS    neural_13    32.149563   24.367080             48            13       624
67        3                 DNS   project_17    32.352170   24.548826             48            17       816
68        3  DNS+CONST_ONE_STEP    neural_13    32.673553   24.901298             48            13       624
69        3  DNS+CONST_ONE_STEP   project_17    33.062500   25.223222             48            17       816
70        3    DNS+AR1_ONE_STEP    neural_13    32.449327   24.771030             48            13       624
71        3    DNS+AR1_ONE_STEP   project_17    32.740327   25.035988             48            17       816

## RBF vs RQ / 4D vs 7D (pooled neural_13)

- Best pooled RMSE — RBF: 29.3372 bp; RQ: 29.3801 bp.
- Best pooled RMSE — 4D: 29.4636 bp; 7D: 29.3372 bp.

## Stage 2 recommendation

Tune priors and optional `lambda_corr` densification on the **selected** kernel/input using the same pre-2004 protocol.
