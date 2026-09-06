#!/usr/bin/env python3
"""
Fit the linear yields-only Dynamic Nelson-Siegel (DNS) benchmark and the
linear Macro-DNS benchmark using the fitting methodology from the notebook,
but using the training window implied by the GP framework / readme.

Key choices
-----------
1. Fitting methodology follows the notebook:
   - convert decimal yields to percent
   - use cross-sectional OLS DNS betas for initialization
   - fit a VAR(1) on the beta path for starting values
   - optimize the Kalman package likelihood with scipy.optimize.minimize
   - reconstruct Q correctly from the package's upper-triangular root
2. Training window follows the uploaded framework, not the notebook:
   - default train start: 1972-01-01
   - default train end  : 2003-12-31
   because the framework sets the outer test period to start in 2004-01.
3. Macro-DNS uses the framework's linear benchmark:
      s_t = c + Phi s_{t-1} + u_t,
      s_t = [beta_t, macro_t],
   where beta_t are the filtered DNS factors and macro_t = [FFR, CU, PI].
4. Beta residuals are saved for both models:
   - DNS:        beta_t - E[beta_t | beta_{t-1}]
   - Macro-DNS:  beta_t - E[beta_t | beta_{t-1}, macro_{t-1}]
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import optimize

warnings.filterwarnings("ignore")

try:
    from Dynamic_Nelson_Siegel_Svensson_Kalman_Filter import kalman
except Exception:
    kalman = None

YIELD_COLS: List[str] = [
    "y3", "y6", "y9", "y12", "y15", "y18", "y21", "y24",
    "y30", "y36", "y48", "y60", "y72", "y84", "y96", "y108", "y120",
]

MATURITIES = np.array(
    [3, 6, 9, 12, 15, 18, 21, 24, 30, 36, 48, 60, 72, 84, 96, 108, 120],
    dtype=float,
)

DEFAULT_TRAIN_START = "1972-01-01"
DEFAULT_TRAIN_END = "2003-12-31"
DEFAULT_OUTPUT_DIR = "dns_macro_dns_outputs"
# Project root = parent of this file's directory (Kalman Filter -> DNS GP Project)
_DNS_GP_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MAIN_PANEL = str(_DNS_GP_PROJECT_ROOT / "Macro" / "master_macro_dns_panel.csv")

MODEL = "NS"
AHEAD = 0
FRCT = False
LIK = True

MACRO_ALIASES = {
    "FFR": ["FFR", "ffr", "FFR_lag", "ffr_lag", "fedfunds", "FEDFUNDS"],
    "CU": ["CU", "cu", "CUg", "cug", "capacity_utilization", "TCU"],
    "PI": ["PI", "pi", "PIf", "pif", "inflation", "CPIAUCSL", "cpi_inflation"],
}

DATE_CANDIDATES = [
    "Month",
    "date",
    "panel_month",
    "calendar_month_start",
    "actual_quote_date",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit DNS and linear Macro-DNS benchmarks.")
    parser.add_argument(
        "--data-csv",
        type=str,
        default=DEFAULT_MAIN_PANEL,
        help="Main panel CSV containing yields and, ideally, FFR/CU/PI macro columns.",
    )
    parser.add_argument(
        "--macro-csv",
        type=str,
        default=None,
        help="Optional separate CSV containing date + FFR/CU/PI columns to merge in.",
    )
    parser.add_argument(
        "--train-start",
        type=str,
        default=DEFAULT_TRAIN_START,
        help="Training window start date (inclusive).",
    )
    parser.add_argument(
        "--train-end",
        type=str,
        default=DEFAULT_TRAIN_END,
        help="Training window end date (inclusive). Defaults to pre-2004 training per the framework.",
    )
    parser.add_argument(
        "--lambda0",
        type=float,
        default=0.0609,
        help="Starting value for Nelson-Siegel lambda in the two-step initializer.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where outputs are saved.",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=2000,
        help="Maximum optimizer iterations for the DNS fit.",
    )
    parser.add_argument(
        "--maxfun",
        type=int,
        default=50000,
        help="Maximum function evaluations for the DNS fit.",
    )
    return parser.parse_args()


def require_kalman_package() -> None:
    if kalman is None:
        raise ImportError(
            "The package 'Dynamic_Nelson_Siegel_Svensson_Kalman_Filter' is not installed.\n"
            "Install one of the following before running this script:\n"
            "  pip install Dynamic-Nelson-Siegel-Svensson-Kalman-Filter\n"
            "or\n"
            "  pip install git+https://github.com/werleycordeiro/Dynamic_Nelson_Siegel_Svensson_Kalman_Filter.git"
        )


def detect_date_column(df: pd.DataFrame) -> str:
    for col in DATE_CANDIDATES:
        if col in df.columns:
            return col
    raise ValueError(f"Could not find a date column. Tried: {DATE_CANDIDATES}")


def coerce_monthly_date(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    monthly_mask = s.str.fullmatch(r"\d{4}-\d{2}")
    s = s.where(~monthly_mask, s + "-01")
    return pd.to_datetime(s, errors="raise")


def read_panel(csv_path: str) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Could not find CSV: {path}")
    df = pd.read_csv(path)
    date_col = detect_date_column(df)
    df["Month"] = coerce_monthly_date(df[date_col])
    df = df.sort_values("Month").drop_duplicates(subset=["Month"]).reset_index(drop=True)
    return df


def resolve_macro_columns(df: pd.DataFrame) -> Dict[str, str]:
    out: Dict[str, str] = {}
    lower_map = {c.lower(): c for c in df.columns}
    for target, aliases in MACRO_ALIASES.items():
        found = None
        for alias in aliases:
            if alias in df.columns:
                found = alias
                break
            alias_lower = alias.lower()
            if alias_lower in lower_map:
                found = lower_map[alias_lower]
                break
        if found is not None:
            out[target] = found
    return out


def prepare_sample(
    data_csv: str,
    macro_csv: Optional[str],
    train_start: str,
    train_end: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], Dict[str, str]]:
    main = read_panel(data_csv)

    missing_yields = [c for c in YIELD_COLS if c not in main.columns]
    if missing_yields:
        raise ValueError(f"Missing yield columns in main panel: {missing_yields}")

    if macro_csv is not None:
        macro = read_panel(macro_csv)
        macro_cols = resolve_macro_columns(macro)
        if set(macro_cols.keys()) != {"FFR", "CU", "PI"}:
            raise ValueError(
                "Macro CSV must contain date plus FFR, CU, and PI columns. "
                f"Detected only: {macro_cols}"
            )
        macro_keep = ["Month"] + [macro_cols[k] for k in ["FFR", "CU", "PI"]]
        macro = macro[macro_keep].rename(columns={v: k for k, v in macro_cols.items()})
        main = main.merge(macro, on="Month", how="left", validate="one_to_one")

    sample = main.loc[
        (main["Month"] >= pd.Timestamp(train_start))
        & (main["Month"] <= pd.Timestamp(train_end))
    ].copy()
    if sample.empty:
        raise ValueError(f"Training sample is empty for window {train_start} to {train_end}.")

    sample = sample.sort_values("Month").reset_index(drop=True)

    # Notebook methodology: convert decimal yields to percent.
    Y = sample[YIELD_COLS].astype(float) * 100.0

    macro_cols = resolve_macro_columns(sample)
    macro_df: Optional[pd.DataFrame] = None
    if set(macro_cols.keys()) == {"FFR", "CU", "PI"}:
        macro_df = sample[[macro_cols[k] for k in ["FFR", "CU", "PI"]]].astype(float).copy()
        macro_df.columns = ["FFR", "CU", "PI"]
        macro_df.index = sample["Month"]
    else:
        macro_cols = {}

    return sample, Y, macro_df, macro_cols


def ns_loadings(lambda_: float, maturities: np.ndarray) -> np.ndarray:
    x = lambda_ * maturities
    c1 = np.ones_like(maturities, dtype=float)
    c2 = (1.0 - np.exp(-x)) / x
    c3 = c2 - np.exp(-x)
    return np.column_stack([c1, c2, c3])


def estimate_betas_ols(y_df: pd.DataFrame, maturities: np.ndarray, lambda_: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    X = ns_loadings(lambda_, maturities)
    XtX_inv = np.linalg.inv(X.T @ X)

    betas = []
    residuals = []
    for t in range(len(y_df)):
        y_t = y_df.iloc[t].to_numpy(dtype=float)
        beta_t = XtX_inv @ X.T @ y_t
        resid_t = y_t - X @ beta_t
        betas.append(beta_t)
        residuals.append(resid_t)

    betas_df = pd.DataFrame(betas, columns=["L", "S", "C"], index=y_df.index)
    residuals_df = pd.DataFrame(residuals, columns=y_df.columns, index=y_df.index)
    return betas_df, residuals_df


def fit_var1(beta_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    B = beta_df.to_numpy(dtype=float)
    Y_t = B[1:, :]
    X_t = np.column_stack([np.ones(len(B) - 1), B[:-1, :]])
    coef, *_ = np.linalg.lstsq(X_t, Y_t, rcond=None)
    c = coef[0, :]
    Phi = coef[1:, :].T
    mu = np.linalg.solve(np.eye(beta_df.shape[1]) - Phi, c)
    eps = Y_t - X_t @ coef
    Sigma = np.cov(eps.T, bias=False)
    return Phi, mu, Sigma, eps


def fit_var1_general(state_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = state_df.to_numpy(dtype=float)
    Y_t = X[1:, :]
    Z_t = np.column_stack([np.ones(len(X) - 1), X[:-1, :]])
    coef, *_ = np.linalg.lstsq(Z_t, Y_t, rcond=None)
    c = coef[0, :]
    Phi = coef[1:, :].T

    n = state_df.shape[1]
    try:
        mu = np.linalg.solve(np.eye(n) - Phi, c)
    except np.linalg.LinAlgError:
        mu = np.linalg.pinv(np.eye(n) - Phi) @ c

    pred = Z_t @ coef
    eps = Y_t - pred
    Sigma = np.cov(eps.T, bias=False)
    return Phi, mu, Sigma, eps, pred


def upper_root_params_from_cov(cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    s11, s12, s13 = cov[0, 0], cov[0, 1], cov[0, 2]
    s22, s23 = cov[1, 1], cov[1, 2]
    s33 = cov[2, 2]

    u22 = np.sqrt(max(s33, 1e-12))
    u12 = s23 / u22
    u02 = s13 / u22

    rem22 = s22 - u12**2
    if rem22 <= 0:
        rem22 = 1e-12
    u11 = np.sqrt(rem22)

    u01 = (s12 - u02 * u12) / u11

    rem11 = s11 - u01**2 - u02**2
    if rem11 <= 0:
        rem11 = 1e-12
    u00 = np.sqrt(rem11)

    U = np.array([
        [u00, u01, u02],
        [0.0, u11, u12],
        [0.0, 0.0, u22],
    ])
    return np.array([u00, u01, u02, u11, u12, u22]), U


def build_initial_param_vector(y_df: pd.DataFrame, maturities: np.ndarray, lambda0: float = 0.0609) -> Dict[str, np.ndarray]:
    betas, resid = estimate_betas_ols(y_df, maturities, lambda0)
    Phi, mu, Sigma, eps = fit_var1(betas)
    H_std = resid.std(axis=0, ddof=1).to_numpy(dtype=float)
    H_std = np.clip(H_std, 1e-6, None)
    q_params, U = upper_root_params_from_cov(Sigma)

    param0 = np.concatenate([
        np.array([np.log(lambda0)]),
        H_std,
        Phi.reshape(-1),
        mu,
        q_params,
    ])
    return {
        "param0": param0,
        "betas_ols": betas,
        "resid_ols": resid,
        "Phi0": Phi,
        "mu0": mu,
        "Sigma0": Sigma,
        "Q_root0": U,
        "eps0": eps,
    }


def unpack_dns_params(param: np.ndarray, n_yields: int = 17) -> Dict[str, np.ndarray]:
    lam = float(np.exp(param[0]))
    h_std = np.asarray(param[1:1 + n_yields], dtype=float)
    H = np.diag(h_std**2)

    idx = 1 + n_yields
    Phi = np.asarray(param[idx:idx + 9], dtype=float).reshape(3, 3)
    idx += 9

    mu = np.asarray(param[idx:idx + 3], dtype=float)
    idx += 3

    q = np.asarray(param[idx:idx + 6], dtype=float)
    U = np.array([
        [q[0], q[1], q[2]],
        [0.0, q[3], q[4]],
        [0.0, 0.0, q[5]],
    ])
    Q = U @ U.T
    c = (np.eye(3) - Phi) @ mu

    return {
        "lambda": lam,
        "H_std": h_std,
        "H": H,
        "Phi": Phi,
        "mu": mu,
        "c": c,
        "Q_root": U,
        "Q": Q,
    }


def dns_param_names(yield_cols: List[str]) -> List[str]:
    names = ["log_lambda"]
    names.extend([f"H_std_{col}" for col in yield_cols])
    names.extend(
        [
            "Phi_L_L",
            "Phi_L_S",
            "Phi_L_C",
            "Phi_S_L",
            "Phi_S_S",
            "Phi_S_C",
            "Phi_C_L",
            "Phi_C_S",
            "Phi_C_C",
        ]
    )
    names.extend(["mu_L", "mu_S", "mu_C"])
    names.extend(["Q_root_00", "Q_root_01", "Q_root_02", "Q_root_11", "Q_root_12", "Q_root_22"])
    return names


def macro_dns_param_vector_and_names(macro_dns: Dict[str, object], cols: List[str]) -> Tuple[np.ndarray, List[str]]:
    phi = np.asarray(macro_dns["Phi"], dtype=float)
    mu = np.asarray(macro_dns["mu"], dtype=float)
    sigma = np.asarray(macro_dns["Sigma"], dtype=float)
    q_params, _ = upper_root_params_from_cov(sigma)

    names: List[str] = []
    for row in cols:
        for col in cols:
            names.append(f"Phi_{row}_{col}")
    names.extend([f"mu_{col}" for col in cols])

    q_names: List[str] = []
    for i in range(len(cols)):
        for j in range(i, len(cols)):
            q_names.append(f"Sigma_root_{i}{j}")
    names.extend(q_names)

    vector = np.concatenate([phi.reshape(-1), mu, q_params])
    return vector, names


def measurement_error_summary(actual_y: pd.DataFrame, fitted_y: pd.DataFrame, maturities: np.ndarray) -> pd.DataFrame:
    resid = actual_y.to_numpy(dtype=float) - fitted_y.to_numpy(dtype=float)
    return pd.DataFrame({
        "maturity_months": np.asarray(maturities, dtype=int),
        "mean_error_bp": resid.mean(axis=0) * 100.0,
        "std_error_bp": resid.std(axis=0, ddof=1) * 100.0,
        "rmse_bp": np.sqrt((resid**2).mean(axis=0)) * 100.0,
    })


def compute_dns_beta_residuals(factors: pd.DataFrame, Phi: np.ndarray, mu: np.ndarray) -> pd.DataFrame:
    c = (np.eye(3) - Phi) @ mu
    prev = factors.iloc[:-1].copy()
    curr = factors.iloc[1:].copy()
    pred = (prev.to_numpy(dtype=float) @ Phi.T) + c
    resid = curr.to_numpy(dtype=float) - pred

    return pd.DataFrame({
        "Month": curr.index,
        "L_actual": curr.iloc[:, 0].to_numpy(),
        "S_actual": curr.iloc[:, 1].to_numpy(),
        "C_actual": curr.iloc[:, 2].to_numpy(),
        "L_pred": pred[:, 0],
        "S_pred": pred[:, 1],
        "C_pred": pred[:, 2],
        "L_resid": resid[:, 0],
        "S_resid": resid[:, 1],
        "C_resid": resid[:, 2],
    })


def fit_dns(
    y_df: pd.DataFrame,
    sample_dates: pd.Series,
    lambda0: float,
    maxiter: int,
    maxfun: int,
) -> Dict[str, object]:
    require_kalman_package()

    init = build_initial_param_vector(y_df, MATURITIES, lambda0=lambda0)
    param0 = init["param0"]

    result = optimize.minimize(
        fun=kalman,
        x0=param0,
        args=(y_df, LIK, FRCT, AHEAD, MATURITIES, MODEL),
        method="L-BFGS-B",
        options={"disp": True, "maxiter": maxiter, "maxfun": maxfun},
    )

    if not result.success:
        raise RuntimeError(f"DNS optimization failed: {result.message}")

    est = unpack_dns_params(result.x.copy(), n_yields=len(YIELD_COLS))
    a_tt, a_t, P_tt, P_t, v2, v1, Yf = kalman(
        param=result.x.copy(),
        Y=y_df,
        lik=False,
        frct=False,
        ahead=0,
        mty=MATURITIES,
        model=MODEL,
    )

    fitted_yields = pd.DataFrame(v1, columns=YIELD_COLS, index=sample_dates)
    factors = pd.DataFrame(a_tt, columns=["L", "S", "C"], index=sample_dates)
    meas_summary = measurement_error_summary(y_df.set_index(sample_dates), fitted_yields, MATURITIES)
    beta_resids = compute_dns_beta_residuals(factors, est["Phi"], est["mu"])

    return {
        "optimizer": result,
        "init": init,
        "params": est,
        "fitted_yields": fitted_yields,
        "filtered_factors": factors,
        "measurement_error_summary": meas_summary,
        "beta_residuals": beta_resids,
    }


def fit_macro_dns(filtered_factors: pd.DataFrame, macro_df: pd.DataFrame) -> Dict[str, object]:
    macro_df = macro_df.loc[filtered_factors.index].copy()
    state = pd.concat([filtered_factors, macro_df], axis=1)
    state.columns = ["L", "S", "C", "FFR", "CU", "PI"]
    state = state.dropna().copy()

    Phi, mu, Sigma, eps, pred = fit_var1_general(state)
    c = (np.eye(state.shape[1]) - Phi) @ mu

    current = state.iloc[1:].copy()
    predicted = pd.DataFrame(pred, columns=state.columns, index=current.index)
    residuals = current - predicted

    A = Phi[:3, :3]
    B = Phi[:3, 3:]
    c_beta = c[:3]

    beta_resid_df = pd.DataFrame({
        "Month": current.index,
        "L_actual": current["L"].to_numpy(),
        "S_actual": current["S"].to_numpy(),
        "C_actual": current["C"].to_numpy(),
        "L_pred": predicted["L"].to_numpy(),
        "S_pred": predicted["S"].to_numpy(),
        "C_pred": predicted["C"].to_numpy(),
        "L_resid": residuals["L"].to_numpy(),
        "S_resid": residuals["S"].to_numpy(),
        "C_resid": residuals["C"].to_numpy(),
    })

    innovation_targets = beta_resid_df[["Month", "L_resid", "S_resid", "C_resid"]].copy()
    innovation_targets.columns = ["Month", "u_L", "u_S", "u_C"]

    return {
        "state": state,
        "Phi": Phi,
        "mu": mu,
        "c": c,
        "Sigma": Sigma,
        "predicted_state": predicted,
        "state_residuals": residuals,
        "beta_residuals": beta_resid_df,
        "A_beta_lag": A,
        "B_macro_lag": B,
        "c_beta": c_beta,
        "innovation_targets": innovation_targets,
    }


def save_dns_outputs(output_dir: Path, dns: Dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    est = dns["params"]
    result = dns["optimizer"]
    init = dns["init"]
    param_names = dns_param_names(YIELD_COLS)

    pd.Series(result.x).to_csv(output_dir / "dns_fitted_params_vector.csv", index=False)
    pd.Series(init["param0"]).to_csv(output_dir / "dns_initial_params_vector.csv", index=False)
    pd.DataFrame({"parameter": param_names, "value": np.asarray(result.x, dtype=float)}).to_csv(
        output_dir / "dns_fitted_params_labeled.csv",
        index=False,
    )
    pd.DataFrame({"parameter": param_names, "value": np.asarray(init["param0"], dtype=float)}).to_csv(
        output_dir / "dns_initial_params_labeled.csv",
        index=False,
    )
    pd.DataFrame(est["Phi"], index=["L", "S", "C"], columns=["L_lag", "S_lag", "C_lag"]).to_csv(output_dir / "dns_Phi.csv")
    pd.Series(est["mu"], index=["L", "S", "C"]).to_csv(output_dir / "dns_mu.csv")
    pd.Series(est["c"], index=["L", "S", "C"]).to_csv(output_dir / "dns_c.csv")
    pd.DataFrame(est["Q"], index=["L", "S", "C"], columns=["L", "S", "C"]).to_csv(output_dir / "dns_Q_covariance.csv")
    pd.DataFrame(est["H"], index=YIELD_COLS, columns=YIELD_COLS).to_csv(output_dir / "dns_H_covariance.csv")
    dns["filtered_factors"].to_csv(output_dir / "dns_filtered_factors.csv", index_label="Month")
    dns["fitted_yields"].to_csv(output_dir / "dns_fitted_yields.csv", index_label="Month")
    dns["measurement_error_summary"].to_csv(output_dir / "dns_measurement_error_summary_bp.csv", index=False)
    dns["beta_residuals"].to_csv(output_dir / "dns_beta_residuals.csv", index=False)

    with open(output_dir / "dns_optimizer_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"success: {result.success}\n")
        f.write(f"message: {result.message}\n")
        f.write(f"negative_loglik: {result.fun}\n")
        f.write(f"lambda: {est['lambda']}\n")


def save_macro_dns_outputs(output_dir: Path, macro_dns: Dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    cols = ["L", "S", "C", "FFR", "CU", "PI"]
    macro_param_vec, macro_param_names = macro_dns_param_vector_and_names(macro_dns, cols)
    pd.Series(macro_param_vec).to_csv(output_dir / "macro_dns_fitted_params_vector.csv", index=False)
    pd.DataFrame({"parameter": macro_param_names, "value": macro_param_vec}).to_csv(
        output_dir / "macro_dns_fitted_params_labeled.csv",
        index=False,
    )
    pd.DataFrame(macro_dns["Phi"], index=cols, columns=[f"{c}_lag" for c in cols]).to_csv(output_dir / "macro_dns_Phi.csv")
    pd.Series(macro_dns["mu"], index=cols).to_csv(output_dir / "macro_dns_mu.csv")
    pd.Series(macro_dns["c"], index=cols).to_csv(output_dir / "macro_dns_c.csv")
    pd.DataFrame(macro_dns["Sigma"], index=cols, columns=cols).to_csv(output_dir / "macro_dns_state_innovation_covariance.csv")
    pd.DataFrame(macro_dns["A_beta_lag"], index=["L", "S", "C"], columns=["L_lag", "S_lag", "C_lag"]).to_csv(output_dir / "macro_dns_A_beta_lag.csv")
    pd.DataFrame(macro_dns["B_macro_lag"], index=["L", "S", "C"], columns=["FFR_lag", "CU_lag", "PI_lag"]).to_csv(output_dir / "macro_dns_B_macro_lag.csv")
    pd.Series(macro_dns["c_beta"], index=["L", "S", "C"]).to_csv(output_dir / "macro_dns_c_beta.csv")
    macro_dns["state"].to_csv(output_dir / "macro_dns_joint_state.csv", index_label="Month")
    macro_dns["predicted_state"].to_csv(output_dir / "macro_dns_predicted_state.csv", index_label="Month")
    macro_dns["state_residuals"].to_csv(output_dir / "macro_dns_state_residuals.csv", index_label="Month")
    macro_dns["beta_residuals"].to_csv(output_dir / "macro_dns_beta_residuals.csv", index=False)
    macro_dns["innovation_targets"].to_csv(output_dir / "macro_dns_innovation_targets.csv", index=False)


def save_run_metadata(
    output_dir: Path,
    args: argparse.Namespace,
    sample: pd.DataFrame,
    macro_cols: Dict[str, str],
    dns: Dict[str, object],
    macro_dns: Optional[Dict[str, object]],
) -> None:
    metadata = {
        "data_csv": str(Path(args.data_csv).resolve()),
        "macro_csv": str(Path(args.macro_csv).resolve()) if args.macro_csv else None,
        "train_start": args.train_start,
        "train_end": args.train_end,
        "n_obs": int(len(sample)),
        "sample_first_month": str(sample["Month"].min().date()),
        "sample_last_month": str(sample["Month"].max().date()),
        "yield_columns": YIELD_COLS,
        "macro_column_mapping": macro_cols,
        "dns_lambda": float(dns["params"]["lambda"]),
        "dns_negative_loglik": float(dns["optimizer"].fun),
        "macro_dns_fit": macro_dns is not None,
    }
    with open(output_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    sample[["Month", *YIELD_COLS, *([*macro_cols.values()] if macro_cols else [])]].to_csv(
        output_dir / "training_sample_used.csv", index=False
    )


def main() -> None:
    args = parse_args()

    sample, Y, macro_df, macro_cols = prepare_sample(
        data_csv=args.data_csv,
        macro_csv=args.macro_csv,
        train_start=args.train_start,
        train_end=args.train_end,
    )

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("DNS / Macro-DNS benchmark fitting")
    print("=" * 72)
    print(f"Main panel        : {Path(args.data_csv).resolve()}")
    print(f"Macro panel       : {Path(args.macro_csv).resolve() if args.macro_csv else 'None'}")
    print(f"Train window      : {args.train_start} to {args.train_end}")
    print(f"Observations      : {len(sample)}")
    print(f"Output directory  : {outdir.resolve()}")
    print(f"Macro cols found  : {macro_cols if macro_cols else 'None'}")
    print()
    print("Fitting plain DNS...")

    dns = fit_dns(
        Y,
        sample["Month"],
        lambda0=args.lambda0,
        maxiter=args.maxiter,
        maxfun=args.maxfun,
    )
    save_dns_outputs(outdir, dns)

    print("DNS fit complete.")
    print(f"Estimated lambda: {dns['params']['lambda']:.6f}")
    print("Saved DNS outputs, including dns_beta_residuals.csv")
    print()

    macro_dns = None
    if macro_df is None:
        print(
            "Macro-DNS was not fit because FFR / CU / PI were not available in the merged sample.\n"
            "Add those columns to the main panel or pass --macro-csv path/to/macro.csv."
        )
    else:
        print("Fitting Macro-DNS from filtered DNS betas + [FFR, CU, PI]...")
        macro_dns = fit_macro_dns(dns["filtered_factors"], macro_df)
        save_macro_dns_outputs(outdir, macro_dns)
        print("Macro-DNS fit complete.")
        print("Saved Macro-DNS outputs, including macro_dns_beta_residuals.csv and macro_dns_innovation_targets.csv")

    save_run_metadata(outdir, args, sample, macro_cols, dns, macro_dns)
    print("Saved run metadata and training sample snapshot.")


if __name__ == "__main__":
    main()
