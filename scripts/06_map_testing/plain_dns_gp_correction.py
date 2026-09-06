#!/usr/bin/env python3
"""
plain_dns_gp_correction.py

Plain DNS + GP correction on one-step beta residuals using a Matérn-5/2 ARD kernel,
hierarchical amplitude shrinkage, log-normal lengthscale priors, and Half-Normal
innovation-noise priors. Hyperparameters are fit by MAP.

The script is written for the residual file structure:
    Month, L_actual, S_actual, C_actual, L_pred, S_pred, C_pred, L_resid, S_resid, C_resid

Important modeling assumption:
- The GP input for the residual at month t is the lagged DNS factor state from month t-1,
  i.e. [L_actual_{t-1}, S_actual_{t-1}, C_actual_{t-1}].
- This matches the DNS one-step transition structure beta_t = c + A beta_{t-1} + noise.
- If your "actual" betas are not the filtered real-time DNS states, replace them with the
  filtered states before using this script.

Validation / training protocol:
- Expanding-window training.
- Most recent validation block of V months.
- h-month embargo between training and validation.
- Annual re-estimation by default.

If the residual file contains a post-2004 sample, the script will run a recursive
out-of-sample backtest. If not, it will use rolling validation only and then fit a
final model on the full available sample.

Dependencies:
    numpy, pandas, scipy
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.special import expit


# ============================================================
# User settings
# ============================================================

RESIDUAL_CSV = "dns_beta_residuals.csv"
PARAMS_CSV = "dns_fitted_params_labeled.csv"
OUTPUT_DIR = "plain_dns_gp_outputs"

# The first realized target month corresponding to the first 1-month-ahead origin
# in a 2004-01 outer test is 2004-02. If your file does not extend this far, the
# script will fall back to rolling validation only and then fit on the full sample.
TEST_START_TARGET = "2004-02-01"

# Rolling-window design
HORIZON_MONTHS = 1
VALIDATION_MONTHS = 84     # preferred GP design in your framework
EMBARGO_MONTHS = 1         # h-month gap between train and validation
REESTIMATION_STEP = 12     # annual re-estimation
FORECAST_BLOCK = 12        # forecast next 12 target months per refit
MIN_TRAIN_MONTHS = 120     # require at least 10 years of training data

# Candidate prior grids for validation-based selection.
# These are centered around the framework values you specified.
TAU_SCALE_GRID = [0.35, 0.50]
LS_LOG_MEAN_GRID = [math.log(1.00), math.log(1.25)]
NOISE_SCALE_GRID = [0.15, 0.25]
LS_LOG_SD = 0.35

# Optimization
MAXITER = 400
N_RESTARTS = 3
JITTER = 1e-6
RANDOM_SEED = 7300


# ============================================================
# Small utilities
# ============================================================

@dataclass
class Standardizer:
    mean_: np.ndarray
    std_: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean_) / self.std_

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * self.std_ + self.mean_


def fit_standardizer(x: np.ndarray) -> Standardizer:
    mean_ = np.mean(x, axis=0)
    std_ = np.std(x, axis=0, ddof=0)
    std_ = np.where(std_ < 1e-8, 1.0, std_)
    return Standardizer(mean_=mean_, std_=std_)


def halfnormal_logpdf_from_logx(logx: float, scale: float) -> float:
    """
    Log density for x = exp(logx), with x ~ HalfNormal(scale),
    including the Jacobian term + logx.
    """
    x = math.exp(logx)
    return math.log(math.sqrt(2.0 / math.pi)) - math.log(scale) - 0.5 * (x / scale) ** 2 + logx


def normal_logpdf(z: float, mean: float, sd: float) -> float:
    return -0.5 * math.log(2.0 * math.pi) - math.log(sd) - 0.5 * ((z - mean) / sd) ** 2


def matern52_ard(X1: np.ndarray, X2: np.ndarray, lengthscales: np.ndarray) -> np.ndarray:
    """
    Matérn-5/2 kernel without amplitude term.
    X1: (n1, d), X2: (n2, d), lengthscales: (d,)
    """
    # scaled Euclidean distance
    diff = (X1[:, None, :] - X2[None, :, :]) / lengthscales[None, None, :]
    r = np.sqrt(np.sum(diff ** 2, axis=2))
    sqrt5_r = np.sqrt(5.0) * r
    return (1.0 + sqrt5_r + (5.0 / 3.0) * r ** 2) * np.exp(-sqrt5_r)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def parse_params_csv(path: Optional[Path]) -> Dict[str, float]:
    if path is None or not path.exists():
        return {}
    df = pd.read_csv(path)
    if not {"parameter", "value"}.issubset(df.columns):
        return {}
    return dict(zip(df["parameter"], df["value"]))


def make_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# ============================================================
# Data construction
# ============================================================

def load_residual_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = {
        "Month", "L_actual", "S_actual", "C_actual",
        "L_pred", "S_pred", "C_pred",
        "L_resid", "S_resid", "C_resid"
    }
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"Residual CSV is missing columns: {sorted(missing)}")

    df = df.copy()
    df["Month"] = pd.to_datetime(df["Month"])
    df = df.sort_values("Month").reset_index(drop=True)

    # GP input for residual at month t is beta_{t-1}
    df["L_lag"] = df["L_actual"].shift(1)
    df["S_lag"] = df["S_actual"].shift(1)
    df["C_lag"] = df["C_actual"].shift(1)

    df = df.dropna(subset=["L_lag", "S_lag", "C_lag"]).reset_index(drop=True)
    return df


# ============================================================
# Joint MAP GP over the 3 DNS residual series
# ============================================================

class JointPlainDNSGP:
    """
    Three-output residual GP model with:
      - one shared global amplitude scale tau_beta,
      - per-output local amplitudes lambda_j,
      - per-output ARD lengthscales,
      - per-output Gaussian innovation noise.

    alpha_j = tau_beta * lambda_j
    """

    def __init__(
        self,
        tau_scale: float,
        ls_log_mean: float,
        ls_log_sd: float,
        noise_scale: float,
        jitter: float = 1e-6,
        maxiter: int = 400,
        n_restarts: int = 3,
        random_seed: int = 7300,
    ) -> None:
        self.tau_scale = tau_scale
        self.ls_log_mean = ls_log_mean
        self.ls_log_sd = ls_log_sd
        self.noise_scale = noise_scale
        self.jitter = jitter
        self.maxiter = maxiter
        self.n_restarts = n_restarts
        self.random_seed = random_seed

        self.x_scaler_: Optional[Standardizer] = None
        self.y_scaler_: Optional[Standardizer] = None
        self.X_train_std_: Optional[np.ndarray] = None
        self.Y_train_std_: Optional[np.ndarray] = None

        self.opt_result_ = None
        self.fitted_params_ = None

    def _unpack(self, theta: np.ndarray) -> Dict[str, np.ndarray]:
        # theta structure:
        # [log_tau,
        #  log_lambda_L, log_lambda_S, log_lambda_C,
        #  log_ls_L(3), log_ls_S(3), log_ls_C(3),
        #  log_noise_L, log_noise_S, log_noise_C]
        idx = 0
        log_tau = theta[idx]; idx += 1
        log_lambda = theta[idx:idx+3]; idx += 3
        log_ls = theta[idx:idx+9].reshape(3, 3); idx += 9
        log_noise = theta[idx:idx+3]; idx += 3

        tau = np.exp(log_tau)
        local = np.exp(log_lambda)
        alpha = tau * local
        ls = np.exp(log_ls)
        noise = np.exp(log_noise)

        return {
            "log_tau": log_tau,
            "log_lambda": log_lambda,
            "log_ls": log_ls,
            "log_noise": log_noise,
            "tau": tau,
            "local": local,
            "alpha": alpha,
            "ls": ls,
            "noise": noise,
        }

    def _neg_log_posterior(self, theta: np.ndarray, X: np.ndarray, Y: np.ndarray) -> float:
        params = self._unpack(theta)
        n = X.shape[0]

        logpost = 0.0

        # Priors
        logpost += halfnormal_logpdf_from_logx(params["log_tau"], self.tau_scale)

        for j in range(3):
            logpost += halfnormal_logpdf_from_logx(params["log_lambda"][j], 1.0)
            logpost += halfnormal_logpdf_from_logx(params["log_noise"][j], self.noise_scale)
            for d in range(3):
                logpost += normal_logpdf(params["log_ls"][j, d], self.ls_log_mean, self.ls_log_sd)

        # Likelihood
        try:
            for j in range(3):
                Kj = (params["alpha"][j] ** 2) * matern52_ard(X, X, params["ls"][j])
                Kj = Kj + ((params["noise"][j] ** 2) + self.jitter) * np.eye(n)

                cF = cho_factor(Kj, lower=True, check_finite=False)
                alpha_y = cho_solve(cF, Y[:, j], check_finite=False)

                logdet = 2.0 * np.sum(np.log(np.diag(cF[0])))
                ll = -0.5 * Y[:, j].T @ alpha_y - 0.5 * logdet - 0.5 * n * math.log(2.0 * math.pi)
                logpost += ll
        except np.linalg.LinAlgError:
            return 1e12
        except FloatingPointError:
            return 1e12

        return -float(logpost)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "JointPlainDNSGP":
        self.x_scaler_ = fit_standardizer(X)
        self.y_scaler_ = fit_standardizer(Y)

        X_std = self.x_scaler_.transform(X)
        Y_std = self.y_scaler_.transform(Y)

        self.X_train_std_ = X_std
        self.Y_train_std_ = Y_std

        base_theta = np.array(
            [
                math.log(max(self.tau_scale, 1e-4)),             # log_tau
                math.log(0.70), math.log(0.70), math.log(0.70), # local lambdas
                self.ls_log_mean, self.ls_log_mean, self.ls_log_mean,  # L ls
                self.ls_log_mean, self.ls_log_mean, self.ls_log_mean,  # S ls
                self.ls_log_mean, self.ls_log_mean, self.ls_log_mean,  # C ls
                math.log(max(self.noise_scale, 1e-4)),
                math.log(max(self.noise_scale, 1e-4)),
                math.log(max(self.noise_scale, 1e-4)),
            ],
            dtype=float
        )

        rng = np.random.default_rng(self.random_seed)
        best_res = None
        best_val = np.inf

        for r in range(self.n_restarts):
            if r == 0:
                theta0 = base_theta.copy()
            else:
                theta0 = base_theta + rng.normal(0.0, 0.15, size=base_theta.shape)

            res = minimize(
                fun=self._neg_log_posterior,
                x0=theta0,
                args=(X_std, Y_std),
                method="L-BFGS-B",
                options={"maxiter": self.maxiter, "disp": False},
            )

            val = float(res.fun)
            if np.isfinite(val) and val < best_val:
                best_val = val
                best_res = res

        if best_res is None:
            raise RuntimeError("MAP optimization failed for all restarts.")

        self.opt_result_ = best_res
        self.fitted_params_ = self._unpack(best_res.x)
        return self

    def predict_residual_mean(self, X_new: np.ndarray) -> np.ndarray:
        if self.fitted_params_ is None or self.X_train_std_ is None or self.Y_train_std_ is None:
            raise RuntimeError("Model must be fit before prediction.")

        Xn_std = self.x_scaler_.transform(X_new)
        n_train = self.X_train_std_.shape[0]
        n_new = Xn_std.shape[0]

        pred_std = np.zeros((n_new, 3))
        for j in range(3):
            alpha_j = self.fitted_params_["alpha"][j]
            ls_j = self.fitted_params_["ls"][j]
            noise_j = self.fitted_params_["noise"][j]

            K = (alpha_j ** 2) * matern52_ard(self.X_train_std_, self.X_train_std_, ls_j)
            K = K + ((noise_j ** 2) + self.jitter) * np.eye(n_train)
            Ks = (alpha_j ** 2) * matern52_ard(Xn_std, self.X_train_std_, ls_j)

            cF = cho_factor(K, lower=True, check_finite=False)
            pred_std[:, j] = Ks @ cho_solve(cF, self.Y_train_std_[:, j], check_finite=False)

        return self.y_scaler_.inverse_transform(pred_std)

    def get_summary(self) -> Dict[str, float]:
        if self.fitted_params_ is None:
            raise RuntimeError("Model must be fit first.")

        p = self.fitted_params_
        out = {
            "tau_beta": float(p["tau"]),
            "alpha_L": float(p["alpha"][0]),
            "alpha_S": float(p["alpha"][1]),
            "alpha_C": float(p["alpha"][2]),
            "lambda_local_L": float(p["local"][0]),
            "lambda_local_S": float(p["local"][1]),
            "lambda_local_C": float(p["local"][2]),
            "noise_L": float(p["noise"][0]),
            "noise_S": float(p["noise"][1]),
            "noise_C": float(p["noise"][2]),
        }
        for j, fac in enumerate(["L", "S", "C"]):
            for d, feat in enumerate(["L_lag", "S_lag", "C_lag"]):
                out[f"ls_{fac}_{feat}"] = float(p["ls"][j, d])
        return out


# ============================================================
# Split logic
# ============================================================

def split_train_val(
    df_avail: pd.DataFrame,
    val_months: int,
    embargo_months: int,
    min_train_months: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if len(df_avail) < val_months + embargo_months + min_train_months:
        raise ValueError("Not enough observations for requested train/val split.")

    val_start = len(df_avail) - val_months
    train_end = val_start - embargo_months

    train_df = df_avail.iloc[:train_end].copy()
    val_df = df_avail.iloc[val_start:].copy()

    if len(train_df) < min_train_months:
        raise ValueError("Training block shorter than MIN_TRAIN_MONTHS after embargo.")

    return train_df, val_df


def build_config_grid() -> List[Dict[str, float]]:
    grid = []
    for tau_scale in TAU_SCALE_GRID:
        for ls_log_mean in LS_LOG_MEAN_GRID:
            for noise_scale in NOISE_SCALE_GRID:
                grid.append(
                    {
                        "tau_scale": tau_scale,
                        "ls_log_mean": ls_log_mean,
                        "ls_log_sd": LS_LOG_SD,
                        "noise_scale": noise_scale,
                    }
                )
    return grid


# ============================================================
# Validation scoring and backtest loop
# ============================================================

def fit_model_on_frame(df_fit: pd.DataFrame, config: Dict[str, float]) -> JointPlainDNSGP:
    X = df_fit[["L_lag", "S_lag", "C_lag"]].to_numpy(float)
    Y = df_fit[["L_resid", "S_resid", "C_resid"]].to_numpy(float)

    model = JointPlainDNSGP(
        tau_scale=config["tau_scale"],
        ls_log_mean=config["ls_log_mean"],
        ls_log_sd=config["ls_log_sd"],
        noise_scale=config["noise_scale"],
        jitter=JITTER,
        maxiter=MAXITER,
        n_restarts=N_RESTARTS,
        random_seed=RANDOM_SEED,
    )
    model.fit(X, Y)
    return model


def validation_score(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: Dict[str, float],
) -> Tuple[float, Dict[str, float]]:
    model = fit_model_on_frame(train_df, config)

    X_val = val_df[["L_lag", "S_lag", "C_lag"]].to_numpy(float)
    Y_val = val_df[["L_resid", "S_resid", "C_resid"]].to_numpy(float)
    Yhat_val = model.predict_residual_mean(X_val)

    out = {
        "rmse_L": rmse(Y_val[:, 0], Yhat_val[:, 0]),
        "rmse_S": rmse(Y_val[:, 1], Yhat_val[:, 1]),
        "rmse_C": rmse(Y_val[:, 2], Yhat_val[:, 2]),
    }
    out["rmse_pooled"] = float(np.sqrt(np.mean((Y_val - Yhat_val) ** 2)))
    return out["rmse_pooled"], out


def select_config_for_block(
    df_avail: pd.DataFrame,
    grid: List[Dict[str, float]],
    val_months: int,
    embargo_months: int,
    min_train_months: int,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    train_df, val_df = split_train_val(
        df_avail=df_avail,
        val_months=val_months,
        embargo_months=embargo_months,
        min_train_months=min_train_months,
    )

    rows = []
    best_cfg = None
    best_score = np.inf

    for cfg in grid:
        score, metric_dict = validation_score(train_df, val_df, cfg)
        row = {**cfg, **metric_dict}
        rows.append(row)
        if score < best_score:
            best_score = score
            best_cfg = cfg

    result_df = pd.DataFrame(rows).sort_values("rmse_pooled").reset_index(drop=True)
    return best_cfg, result_df


def run_recursive_backtest(df: pd.DataFrame, outdir: Path) -> None:
    grid = build_config_grid()
    test_start = pd.Timestamp(TEST_START_TARGET)
    if test_start not in set(df["Month"].drop_duplicates().sort_values().to_list()):
        raise ValueError("TEST_START_TARGET not found in sample.")
    train_df = df.loc[df["Month"] < test_start].copy()
    test_df = df.loc[df["Month"] >= test_start].copy()
    if len(train_df) < (MIN_TRAIN_MONTHS + VALIDATION_MONTHS + EMBARGO_MONTHS):
        raise ValueError("Not enough pre-test observations for train/validation split.")
    if test_df.empty:
        raise ValueError("No post-test rows available for backtest.")

    # Select hyperparameters using only pre-test data, then fit once on all pre-test rows.
    best_cfg, tuning_df = select_config_for_block(
        df_avail=train_df,
        grid=grid,
        val_months=VALIDATION_MONTHS,
        embargo_months=EMBARGO_MONTHS,
        min_train_months=MIN_TRAIN_MONTHS,
    )
    tuning_df.insert(0, "block_start", test_start)
    model = fit_model_on_frame(train_df, best_cfg)
    fit_summary = model.get_summary()
    fit_summary.update(
        {
            "block_start": test_start,
            "n_fit_obs": len(train_df),
            "selected_tau_scale": best_cfg["tau_scale"],
            "selected_ls_log_mean": best_cfg["ls_log_mean"],
            "selected_noise_scale": best_cfg["noise_scale"],
        }
    )

    X_test = test_df[["L_lag", "S_lag", "C_lag"]].to_numpy(float)
    resid_hat = model.predict_residual_mean(X_test)
    block_predictions = []
    for i, (_, row) in enumerate(test_df.iterrows()):
        pred_L_corr = row["L_pred"] + resid_hat[i, 0]
        pred_S_corr = row["S_pred"] + resid_hat[i, 1]
        pred_C_corr = row["C_pred"] + resid_hat[i, 2]
        block_predictions.append(
            {
                "Month": row["Month"],
                "block_start": test_start,
                "L_actual": row["L_actual"],
                "S_actual": row["S_actual"],
                "C_actual": row["C_actual"],
                "L_pred_dns": row["L_pred"],
                "S_pred_dns": row["S_pred"],
                "C_pred_dns": row["C_pred"],
                "L_resid_actual": row["L_resid"],
                "S_resid_actual": row["S_resid"],
                "C_resid_actual": row["C_resid"],
                "L_resid_gp_hat": resid_hat[i, 0],
                "S_resid_gp_hat": resid_hat[i, 1],
                "C_resid_gp_hat": resid_hat[i, 2],
                "L_pred_dns_gp": pred_L_corr,
                "S_pred_dns_gp": pred_S_corr,
                "C_pred_dns_gp": pred_C_corr,
                "L_err_dns": row["L_pred"] - row["L_actual"],
                "S_err_dns": row["S_pred"] - row["S_actual"],
                "C_err_dns": row["C_pred"] - row["C_actual"],
                "L_err_dns_gp": pred_L_corr - row["L_actual"],
                "S_err_dns_gp": pred_S_corr - row["S_actual"],
                "C_err_dns_gp": pred_C_corr - row["C_actual"],
            }
        )

    pred_df = pd.DataFrame(block_predictions).sort_values("Month").reset_index(drop=True)
    tuning_full = tuning_df.reset_index(drop=True)
    fit_df = pd.DataFrame([fit_summary])

    pred_df.to_csv(outdir / "plain_dns_gp_backtest_predictions.csv", index=False)
    tuning_full.to_csv(outdir / "plain_dns_gp_tuning_results.csv", index=False)
    fit_df.to_csv(outdir / "plain_dns_gp_fit_summaries.csv", index=False)

    if not pred_df.empty:
        summary_rows = []
        for fac in ["L", "S", "C"]:
            dns_rmse = rmse(pred_df[f"{fac}_actual"].to_numpy(), pred_df[f"{fac}_pred_dns"].to_numpy())
            gp_rmse = rmse(pred_df[f"{fac}_actual"].to_numpy(), pred_df[f"{fac}_pred_dns_gp"].to_numpy())
            summary_rows.append(
                {
                    "factor": fac,
                    "dns_rmse": dns_rmse,
                    "dns_gp_rmse": gp_rmse,
                    "rmse_improvement": dns_rmse - gp_rmse,
                }
            )
        pooled_dns = np.sqrt(
            np.mean(
                np.concatenate(
                    [
                        (pred_df["L_pred_dns"] - pred_df["L_actual"]).to_numpy(),
                        (pred_df["S_pred_dns"] - pred_df["S_actual"]).to_numpy(),
                        (pred_df["C_pred_dns"] - pred_df["C_actual"]).to_numpy(),
                    ]
                ) ** 2
            )
        )
        pooled_gp = np.sqrt(
            np.mean(
                np.concatenate(
                    [
                        (pred_df["L_pred_dns_gp"] - pred_df["L_actual"]).to_numpy(),
                        (pred_df["S_pred_dns_gp"] - pred_df["S_actual"]).to_numpy(),
                        (pred_df["C_pred_dns_gp"] - pred_df["C_actual"]).to_numpy(),
                    ]
                ) ** 2
            )
        )
        summary_rows.append(
            {
                "factor": "POOLED",
                "dns_rmse": pooled_dns,
                "dns_gp_rmse": pooled_gp,
                "rmse_improvement": pooled_dns - pooled_gp,
            }
        )
        pd.DataFrame(summary_rows).to_csv(outdir / "plain_dns_gp_backtest_summary.csv", index=False)


def run_rolling_validation_then_full_fit(df: pd.DataFrame, outdir: Path) -> None:
    """
    Use rolling validation windows over the available sample to pick one
    configuration, then refit on the full dataset and export fitted MAP values
    and in-sample GP residual fits.

    This path is used automatically when the file does not contain a post-2004 test era.
    """
    grid = build_config_grid()
    months = sorted(df["Month"].unique())

    earliest_reestimation_idx = MIN_TRAIN_MONTHS + EMBARGO_MONTHS + VALIDATION_MONTHS
    if earliest_reestimation_idx >= len(months):
        raise ValueError("Not enough data for rolling validation with current settings.")

    # Use the last several annual block starts for model selection
    candidate_block_starts = months[earliest_reestimation_idx::REESTIMATION_STEP]
    if len(candidate_block_starts) > 8:
        candidate_block_starts = candidate_block_starts[-8:]

    roll_rows = []
    scores_by_cfg = {}

    for block_start in candidate_block_starts:
        df_avail = df.loc[df["Month"] < block_start].copy()
        if len(df_avail) < (MIN_TRAIN_MONTHS + VALIDATION_MONTHS + EMBARGO_MONTHS):
            continue

        best_cfg, tuning_df = select_config_for_block(
            df_avail=df_avail,
            grid=grid,
            val_months=VALIDATION_MONTHS,
            embargo_months=EMBARGO_MONTHS,
            min_train_months=MIN_TRAIN_MONTHS,
        )
        tuning_df.insert(0, "block_start", block_start)
        roll_rows.append(tuning_df)

        # aggregate scores across block starts by exact config
        for _, row in tuning_df.iterrows():
            key = (row["tau_scale"], row["ls_log_mean"], row["noise_scale"])
            scores_by_cfg.setdefault(key, []).append(row["rmse_pooled"])

    if not scores_by_cfg:
        raise RuntimeError("Rolling validation did not produce any candidate scores.")

    # Choose the config with best average rolling validation RMSE
    best_key = min(scores_by_cfg.keys(), key=lambda k: np.mean(scores_by_cfg[k]))
    final_cfg = {
        "tau_scale": float(best_key[0]),
        "ls_log_mean": float(best_key[1]),
        "ls_log_sd": LS_LOG_SD,
        "noise_scale": float(best_key[2]),
    }

    # Refit on full available sample
    model = fit_model_on_frame(df, final_cfg)
    fit_summary = model.get_summary()
    fit_summary.update(
        {
            "n_fit_obs": len(df),
            "selected_tau_scale": final_cfg["tau_scale"],
            "selected_ls_log_mean": final_cfg["ls_log_mean"],
            "selected_noise_scale": final_cfg["noise_scale"],
        }
    )

    X_full = df[["L_lag", "S_lag", "C_lag"]].to_numpy(float)
    resid_hat = model.predict_residual_mean(X_full)

    full_pred = df.copy()
    full_pred["L_resid_gp_hat"] = resid_hat[:, 0]
    full_pred["S_resid_gp_hat"] = resid_hat[:, 1]
    full_pred["C_resid_gp_hat"] = resid_hat[:, 2]

    full_pred["L_pred_dns_gp"] = full_pred["L_pred"] + full_pred["L_resid_gp_hat"]
    full_pred["S_pred_dns_gp"] = full_pred["S_pred"] + full_pred["S_resid_gp_hat"]
    full_pred["C_pred_dns_gp"] = full_pred["C_pred"] + full_pred["C_resid_gp_hat"]

    full_pred["L_err_dns"] = full_pred["L_pred"] - full_pred["L_actual"]
    full_pred["S_err_dns"] = full_pred["S_pred"] - full_pred["S_actual"]
    full_pred["C_err_dns"] = full_pred["C_pred"] - full_pred["C_actual"]

    full_pred["L_err_dns_gp"] = full_pred["L_pred_dns_gp"] - full_pred["L_actual"]
    full_pred["S_err_dns_gp"] = full_pred["S_pred_dns_gp"] - full_pred["S_actual"]
    full_pred["C_err_dns_gp"] = full_pred["C_pred_dns_gp"] - full_pred["C_actual"]

    summary_rows = []
    for fac in ["L", "S", "C"]:
        dns_rmse = rmse(full_pred[f"{fac}_actual"].to_numpy(), full_pred[f"{fac}_pred"].to_numpy())
        gp_rmse = rmse(full_pred[f"{fac}_actual"].to_numpy(), full_pred[f"{fac}_pred_dns_gp"].to_numpy())
        summary_rows.append(
            {
                "factor": fac,
                "dns_rmse": dns_rmse,
                "dns_gp_rmse": gp_rmse,
                "rmse_improvement": dns_rmse - gp_rmse,
            }
        )
    pooled_dns = np.sqrt(
        np.mean(
            np.concatenate(
                [
                    (full_pred["L_pred"] - full_pred["L_actual"]).to_numpy(),
                    (full_pred["S_pred"] - full_pred["S_actual"]).to_numpy(),
                    (full_pred["C_pred"] - full_pred["C_actual"]).to_numpy(),
                ]
            ) ** 2
        )
    )
    pooled_gp = np.sqrt(
        np.mean(
            np.concatenate(
                [
                    (full_pred["L_pred_dns_gp"] - full_pred["L_actual"]).to_numpy(),
                    (full_pred["S_pred_dns_gp"] - full_pred["S_actual"]).to_numpy(),
                    (full_pred["C_pred_dns_gp"] - full_pred["C_actual"]).to_numpy(),
                ]
            ) ** 2
        )
    )
    summary_rows.append(
        {
            "factor": "POOLED",
            "dns_rmse": pooled_dns,
            "dns_gp_rmse": pooled_gp,
            "rmse_improvement": pooled_dns - pooled_gp,
        }
    )

    tuning_all = pd.concat(roll_rows, ignore_index=True) if roll_rows else pd.DataFrame()
    tuning_all.to_csv(outdir / "plain_dns_gp_rolling_validation_results.csv", index=False)
    pd.DataFrame([fit_summary]).to_csv(outdir / "plain_dns_gp_final_fit_summary.csv", index=False)
    full_pred.to_csv(outdir / "plain_dns_gp_full_sample_fit.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(outdir / "plain_dns_gp_full_sample_summary.csv", index=False)

    with open(outdir / "plain_dns_gp_selected_config.json", "w", encoding="utf-8") as f:
        json.dump(final_cfg, f, indent=2)


# ============================================================
# Main
# ============================================================

def main() -> None:
    np.random.seed(RANDOM_SEED)

    script_dir = Path(__file__).resolve().parent
    resid_path = Path(RESIDUAL_CSV)
    params_path = Path(PARAMS_CSV)
    outdir = Path(OUTPUT_DIR)

    if not resid_path.is_absolute():
        candidate = script_dir / resid_path
        if candidate.exists():
            resid_path = candidate
    if not params_path.is_absolute():
        candidate = script_dir / params_path
        if candidate.exists():
            params_path = candidate
    if not outdir.is_absolute():
        outdir = script_dir / outdir
    make_output_dir(outdir)

    if not resid_path.exists():
        raise FileNotFoundError(f"Residual CSV not found: {resid_path}")

    df = load_residual_frame(resid_path)
    params = parse_params_csv(params_path)

    # Save metadata and DNS fitted params for traceability
    metadata = {
        "residual_csv": str(resid_path.resolve()),
        "params_csv": str(params_path.resolve()) if params_path.exists() else None,
        "n_rows_model_frame": int(len(df)),
        "month_min": str(df["Month"].min().date()),
        "month_max": str(df["Month"].max().date()),
        "test_start_target": TEST_START_TARGET,
        "validation_months": VALIDATION_MONTHS,
        "embargo_months": EMBARGO_MONTHS,
        "reestimation_step": REESTIMATION_STEP,
        "forecast_block": FORECAST_BLOCK,
        "min_train_months": MIN_TRAIN_MONTHS,
        "tau_scale_grid": TAU_SCALE_GRID,
        "ls_log_mean_grid": LS_LOG_MEAN_GRID,
        "noise_scale_grid": NOISE_SCALE_GRID,
        "ls_log_sd": LS_LOG_SD,
    }
    with open(outdir / "plain_dns_gp_run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    if params:
        pd.DataFrame({"parameter": list(params.keys()), "value": list(params.values())}).to_csv(
            outdir / "dns_params_snapshot.csv", index=False
        )

    # Decide whether we have an outer test era
    test_start_ts = pd.Timestamp(TEST_START_TARGET)
    if df["Month"].max() >= test_start_ts:
        print(f"[INFO] Running recursive out-of-sample backtest from target month {TEST_START_TARGET}.")
        run_recursive_backtest(df, outdir)
    else:
        print(
            "[INFO] Residual file ends before TEST_START_TARGET. "
            "Running rolling validation only, then fitting a final MAP GP on the full available sample."
        )
        run_rolling_validation_then_full_fit(df, outdir)

    print(f"[DONE] Outputs written to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
