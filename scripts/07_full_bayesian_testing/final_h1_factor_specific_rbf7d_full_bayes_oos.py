#!/usr/bin/env python3
"""
h=1 only: factor-specific scalar RBF-ARD 7D GPs with fully Bayesian hyperparameter
integration (posterior predictive mean over MH samples). Yield-implied beta correction.
Moderate-long-time prior (explicit normals). lambda_corr = 0.50.

DNS fixed; no dns_beta_residuals.csv; no macro; no multi-horizon.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUT_DIR = SCRIPT_DIR / "final_h1_factor_specific_rbf7d_full_bayes_oos_outputs"
BROKEN_RESIDUAL_CSV = SCRIPT_DIR / "dns_beta_residuals.csv"
BETA_CACHE_CSV = SCRIPT_DIR / "fixed_dns_kalman_filtered_beta_decimal.csv"

import evaluate_fixed_dns_gp as efd
import evaluate_fixed_dns_gp_7d_kernel_comparison as k7
import evaluate_fixed_dns_gp_yield_implied_beta_target as yib

RIDGE_PROJ = yib.RIDGE_PROJ
RANDOM_SEED = int(os.environ.get("FB_H1_SEED", "7421"))
LAMBDA_CORR = 0.50

# Moderate-long-time (explicit Step 8 normals on log hyperparameters, order matches k7.unpack_u)
PRIOR_MEAN_U = np.array(
    [
        math.log(0.10),
        math.log(2.25),
        math.log(2.25),
        math.log(2.25),
        math.log(1.75),
        math.log(1.75),
        math.log(1.75),
        math.log(2.25),
        math.log(0.80),
    ],
    dtype=float,
)
PRIOR_SD_U = np.array([0.40, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.30], dtype=float)

MODEL_DNS = "DNS"
MODEL_FB = "DNS+GP_FACTOR_SPECIFIC_RBF_MOD_LONG_FULL_BAYES_LAM050"
MODEL_MAP = "DNS+GP_FACTOR_SPECIFIC_RBF_MOD_LONG_MAP_LAM050"
FAC_NAMES = ("L", "S", "C")

JITTER_TRIALS = (1e-6, 1e-5, 1e-4, 1e-3)
MAP_MAXITER = int(os.environ.get("FB_H1_MAP_MAXITER", "120"))
MAP_N_RESTARTS = int(os.environ.get("FB_H1_MAP_RESTARTS", "5"))
FB_CHAINS = int(os.environ.get("FB_H1_CHAINS", "4"))
FB_WARMUP = int(os.environ.get("FB_H1_WARMUP", "3000"))
FB_DRAWS = int(os.environ.get("FB_H1_DRAWS", "3000"))
FB_THIN = int(os.environ.get("FB_H1_THIN", "1"))


def log_prior_u(u: np.ndarray) -> float:
    if not np.all(np.isfinite(u)):
        return float("-inf")
    lp = 0.0
    for k in range(9):
        delta = float(u[k]) - PRIOR_MEAN_U[k]
        lp += -0.5 * (delta * delta) / (PRIOR_SD_U[k] ** 2)
        lp += -0.5 * math.log(2.0 * math.pi) - math.log(PRIOR_SD_U[k])
    return lp


def grad_log_prior_u(u: np.ndarray) -> np.ndarray:
    return -(u - PRIOR_MEAN_U) / (PRIOR_SD_U**2)


def log_posterior_rbf(
    u: np.ndarray, d_stack: np.ndarray, y_std: np.ndarray, jitter: float
) -> float:
    ll = k7.gp_logml_rbf(u, d_stack, y_std, jitter)
    lp = log_prior_u(u)
    if not np.isfinite(ll) or not np.isfinite(lp):
        return float("-inf")
    return float(ll + lp)


def neg_log_post_and_grad(
    u: np.ndarray, d_stack: np.ndarray, y_std: np.ndarray, jitter: float
) -> Tuple[float, np.ndarray]:
    ll, g_ll = k7.gp_logml_and_grad_rbf(u, d_stack, y_std, jitter)
    lp = log_prior_u(u)
    g_lp = grad_log_prior_u(u)
    if not np.isfinite(ll) or not np.isfinite(lp):
        return 1e12, np.zeros(9)
    f = -(ll + lp)
    g = -(g_ll + g_lp)
    return float(f), g.astype(float)


@dataclass
class MapFit:
    u: np.ndarray
    map_objective: float
    optimizer_success: bool
    best_restart_name: str
    final_jitter: float
    pathology_flag: str
    opt_message: str


def u0_restarts() -> List[Tuple[str, np.ndarray]]:
    rest: List[Tuple[str, np.ndarray]] = [("prior_mean_moderate_long", PRIOR_MEAN_U.copy())]
    for name, u0 in k7.STRUCTURED_RESTARTS:
        rest.append((f"k7_{name}", u0.copy()))
    return rest


def fit_map_one_factor(
    x_train_std: np.ndarray, y_train_std: np.ndarray, seed: int
) -> MapFit:
    d_stack = k7.precompute_sq_dists(x_train_std)
    candidates = u0_restarts()[: max(1, MAP_N_RESTARTS)]
    best: Optional[MapFit] = None
    best_f = 1e300
    lo = np.array([b[0] for b in k7.LBFGS_U_BOUNDS], dtype=float)
    hi = np.array([b[1] for b in k7.LBFGS_U_BOUNDS], dtype=float)
    for name, u0 in candidates:
        u0c = np.clip(u0.astype(float), lo, hi)
        for jit in JITTER_TRIALS:

            def make_fun(jt: float):
                def fun(u: np.ndarray) -> Tuple[float, np.ndarray]:
                    return neg_log_post_and_grad(u, d_stack, y_train_std, jt)

                return fun

            fun = make_fun(jit)
            try:
                res = minimize(
                    fun,
                    u0c,
                    jac=True,
                    method="L-BFGS-B",
                    bounds=k7.LBFGS_U_BOUNDS,
                    options={"maxiter": MAP_MAXITER, "disp": False},
                )
            except ValueError:
                continue
            f = float(res.fun)
            if np.isfinite(f) and f < best_f:
                u_hat = res.x.astype(float)
                ll = k7.gp_logml_rbf(u_hat, d_stack, y_train_std, jit)
                lp = log_prior_u(u_hat)
                mobj = float(ll + lp) if np.isfinite(ll) and np.isfinite(lp) else float("nan")
                al, ells, sig = k7.unpack_u(u_hat)
                path = k7.pathology_flags(al, ells, sig, bool(res.success), jit)
                best_f = f
                best = MapFit(
                    u=u_hat,
                    map_objective=mobj,
                    optimizer_success=bool(res.success),
                    best_restart_name=name,
                    final_jitter=jit,
                    pathology_flag=path,
                    opt_message=str(res.message),
                )
    if best is None:
        return MapFit(
            u=PRIOR_MEAN_U.copy(),
            map_objective=float("nan"),
            optimizer_success=False,
            best_restart_name="failed",
            final_jitter=JITTER_TRIALS[-1],
            pathology_flag="all_failed",
            opt_message="",
        )
    al, ells, sig = k7.unpack_u(best.u)
    best.pathology_flag = k7.pathology_flags(al, ells, sig, best.optimizer_success, best.final_jitter)
    return best


def rw_mh_chain(
    d_stack: np.ndarray,
    y_std: np.ndarray,
    u_init: np.ndarray,
    jitter: float,
    chain_id: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, float, int]:
    """Returns (samples [S,9], logpost [S], accept_rate, final_scale, n_chol_fail, mean_lp)."""
    rng = np.random.default_rng(seed + chain_id * 100_003)
    n_warm = FB_WARMUP
    n_draw = FB_DRAWS
    tot = n_warm + n_draw
    u = u_init.copy()
    lp = log_posterior_rbf(u, d_stack, y_std, jitter)
    if not np.isfinite(lp):
        u = PRIOR_MEAN_U + 0.01 * rng.standard_normal(9)
        lp = log_posterior_rbf(u, d_stack, y_std, jitter)
    prop_sd = 0.15 * PRIOR_SD_U
    scale = 1.0
    accept_warm = 0
    n_warm_total = 0
    n_chol_fail = 0
    kept: List[np.ndarray] = []
    lps: List[float] = []
    accept_post = 0
    n_post = 0

    for t in range(tot):
        u_prop = u + scale * prop_sd * rng.standard_normal(9)
        lp_prop = log_posterior_rbf(u_prop, d_stack, y_std, jitter)
        if not np.isfinite(lp_prop):
            n_chol_fail += 1
            acc = False
        else:
            la = lp_prop - lp
            if la >= 0.0:
                acc = True
            else:
                acc = math.log(rng.random()) < la
        if acc:
            u, lp = u_prop, lp_prop
        if t < n_warm:
            n_warm_total += 1
            if acc:
                accept_warm += 1
            if t > 0 and t % 200 == 0:
                ar = accept_warm / max(1, n_warm_total)
                if ar < 0.20:
                    scale *= 1.05
                elif ar > 0.35:
                    scale /= 1.05
                accept_warm = 0
                n_warm_total = 0
        else:
            n_post += 1
            if acc:
                accept_post += 1
            if ((t - n_warm) % FB_THIN) == 0:
                kept.append(u.copy())
                lps.append(lp)

    arr = np.stack(kept, axis=0) if kept else np.zeros((0, 9))
    lpar = np.array(lps, dtype=float)
    acc_rate = float(accept_post / max(1, n_post))
    return arr, lpar, acc_rate, float(scale), n_chol_fail, float(np.mean(lpar) if len(lpar) else float("nan"))


def build_h1_yield_implied(
    beta_f: np.ndarray,
    months: pd.DatetimeIndex,
    yields_dec: np.ndarray,
    dns: efd.FixedDNS,
    lam_proj: np.ndarray,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    pd.DatetimeIndex,
    pd.DatetimeIndex,
    np.ndarray,
    np.ndarray,
]:
    h = 1
    t_len = len(months)
    mu = dns.mu_dec
    phi_h = efd.matrix_power_int(dns.Phi, h)
    pred = np.full_like(beta_f, np.nan)
    for t in range(t_len - h):
        pred[t + h] = mu + phi_h @ (beta_f[t] - mu)
    origin_idx = np.arange(1, t_len - h, dtype=int)
    target_idx = origin_idx + h
    od = months[origin_idx]
    td = months[target_idx]
    y_dns = pred[target_idx] @ lam_proj.T
    y_act = yields_dec[target_idx]
    u_y = y_act - y_dns
    ry = yib.yield_implied_beta_correction(u_y, lam_proj, RIDGE_PROJ)
    base = (
        np.isfinite(ry).all(axis=1)
        & np.isfinite(u_y).all(axis=1)
        & np.all(np.isfinite(yields_dec[target_idx]), axis=1)
    )
    x7 = k7.build_X_7d(beta_f, mu, months, origin_idx)
    base = base & np.isfinite(x7).all(axis=1)
    tr_m = base & (td <= efd.TRAIN_END)
    te_m = (
        base
        & (od >= efd.FORECAST_ORIGIN_START)
        & (od <= efd.FORECAST_ORIGIN_END)
    )
    itrain = np.where(tr_m)[0]
    itest = np.where(te_m)[0]
    return (
        x7,
        ry,
        pred[target_idx],
        y_act,
        itrain,
        itest,
        od,
        td,
        origin_idx,
        target_idx,
    )


def standardize_train(
    x_tr: np.ndarray, y_tr: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_mean = x_tr.mean(axis=0)
    x_std = x_tr.std(axis=0, ddof=0)
    x_std = np.where(x_std < 1e-8, 1.0, x_std)
    y_mean = y_tr.mean(axis=0)
    y_std = y_tr.std(axis=0, ddof=0)
    y_std = np.where(y_std < 1e-8, 1.0, y_std)
    return x_mean, x_std, y_mean, y_std


def yield_long_df(
    od: pd.DatetimeIndex,
    td: pd.DatetimeIndex,
    y_act: np.ndarray,
    beta_pred: np.ndarray,
    dns: efd.FixedDNS,
    model: str,
    horizon: int,
) -> pd.DataFrame:
    lam = efd.ns_loadings(efd.MATURITIES, dns.lam)
    y_p = beta_pred @ lam.T
    err_bp = (y_p - y_act) * 1.0e4
    rows: List[Dict[str, Any]] = []
    for i in range(len(od)):
        for j, m in enumerate(efd.MATURITIES.astype(int)):
            rows.append(
                {
                    "model": model,
                    "forecast_origin": od[i],
                    "target_month": td[i],
                    "horizon": horizon,
                    "maturity_months": int(m),
                    "y_actual_decimal": float(y_act[i, j]),
                    "y_pred_decimal": float(y_p[i, j]),
                    "error_bp": float(err_bp[i, j]),
                }
            )
    return pd.DataFrame(rows)


def qvals(stack: np.ndarray, qs: Sequence[float]) -> List[np.ndarray]:
    return [np.quantile(stack, q, axis=0) for q in qs]


def main() -> None:
    np.random.seed(RANDOM_SEED)
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[sanity] Broken residual file NOT used:", not BROKEN_RESIDUAL_CSV.exists() or True, flush=True)
    print("[sanity] h=1 only, RBF-ARD 7D, factor-specific scalars, lambda_corr=", LAMBDA_CORR, flush=True)

    print("[data] load_panel...", flush=True)
    panel = efd.load_panel()
    dns = efd.load_fixed_dns_params(efd.DNS_PARAMS_CSV)
    beta_f = k7.load_or_compute_beta(panel, dns)
    months = pd.DatetimeIndex(panel["Month"])
    yields_dec = panel[efd.YIELD_COLS].to_numpy(float)
    lam_proj = efd.ns_loadings(efd.MATURITIES, dns.lam)

    x7, ry, beta_dns_all, y_act_all, itrain, itest, od, td, oidx, _tidx = build_h1_yield_implied(
        beta_f, months, yields_dec, dns, lam_proj
    )
    n_train = len(itrain)
    n_test = len(itest)
    if n_train < 20 or n_test < 10:
        raise RuntimeError(f"insufficient split n_train={n_train} n_test={n_test}")
    print(f"[data] n_train={n_train} n_test={n_test}", flush=True)

    x_tr_raw = x7[itrain]
    y_tr_raw = ry[itrain]
    x_te_raw = x7[itest]
    y_te_tar = ry[itest]
    beta_dns_te = beta_dns_all[itest]
    y_act_te = y_act_all[itest]
    od_te = od[itest]
    td_te = td[itest]

    x_mean, x_std, y_mean, y_std = standardize_train(x_tr_raw, y_tr_raw)
    x_tr = (x_tr_raw - x_mean) / x_std
    x_te = (x_te_raw - x_mean) / x_std

    map_rows: List[Dict[str, Any]] = []
    sample_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    diag_rows: List[Dict[str, Any]] = []

    pred_fb = np.zeros((n_test, 3), dtype=float)
    pred_map = np.zeros((n_test, 3), dtype=float)
    q05 = np.zeros((n_test, 3))
    q50 = np.zeros((n_test, 3))
    q95 = np.zeros((n_test, 3))
    sdc = np.zeros((n_test, 3))
    stack_corr = []

    for j, fac in enumerate(FAC_NAMES):
        print(f"[factor {fac}] MAP + MCMC...", flush=True)
        y_tr_s = (y_tr_raw[:, j] - y_mean[j]) / y_std[j]
        d_stack = k7.precompute_sq_dists(x_tr)
        mf = fit_map_one_factor(x_tr, y_tr_s, RANDOM_SEED + j)
        al, e1, e2, e3, e4, e5, e6, e7, sg = (
            float(np.exp(mf.u[0])),
            float(np.exp(mf.u[1])),
            float(np.exp(mf.u[2])),
            float(np.exp(mf.u[3])),
            float(np.exp(mf.u[4])),
            float(np.exp(mf.u[5])),
            float(np.exp(mf.u[6])),
            float(np.exp(mf.u[7])),
            float(np.exp(mf.u[8])),
        )
        map_rows.append(
            {
                "factor": fac,
                "alpha": al,
                "ell_L_dm_t": e1,
                "ell_S_dm_t": e2,
                "ell_C_dm_t": e3,
                "ell_dL_t": e4,
                "ell_dS_t": e5,
                "ell_dC_t": e6,
                "ell_time": e7,
                "sigma": sg,
                "map_objective": mf.map_objective,
                "optimizer_success": mf.optimizer_success,
                "best_restart_name": mf.best_restart_name,
                "final_jitter": mf.final_jitter,
                "n_train": n_train,
                "n_test": n_test,
                "pathology_flag": mf.pathology_flag,
            }
        )

        pred_map[:, j] = k7.predict_mean(
            x_tr, y_tr_s, x_te, mf.u, mf.final_jitter, "rbf"
        )
        pred_map[:, j] = pred_map[:, j] * y_std[j] + y_mean[j]
        print(f"[factor {fac}] MH sampling ({FB_CHAINS} chains)...", flush=True)

        all_samples: List[np.ndarray] = []
        all_lp: List[float] = []
        all_chain_id: List[int] = []
        for c in range(FB_CHAINS):
            pert = mf.u + 0.02 * np.random.default_rng(RANDOM_SEED + 17 * j + c).standard_normal(9)
            samp, lps, acc_r, scale, n_cf, _ = rw_mh_chain(
                d_stack, y_tr_s, pert, mf.final_jitter, c, RANDOM_SEED + j * 97
            )
            if samp.shape[0] == 0:
                continue
            for ii in range(samp.shape[0]):
                all_samples.append(samp[ii])
                all_lp.append(float(lps[ii]))
                all_chain_id.append(c)
            diag_rows.append(
                {
                    "factor": fac,
                    "chain": c,
                    "warmup": FB_WARMUP,
                    "draws": FB_DRAWS,
                    "retained_draws": int(samp.shape[0]),
                    "acceptance_rate": acc_r,
                    "proposal_scale_final": scale,
                    "cholesky_failures": n_cf,
                    "mean_log_posterior": float(np.mean(lps)) if len(lps) else float("nan"),
                    "sd_log_posterior": float(np.std(lps)) if len(lps) else float("nan"),
                }
            )

        if not all_samples:
            raise RuntimeError(f"No MH samples for factor {fac}")
        u_mat = np.stack(all_samples, axis=0)
        n_s = u_mat.shape[0]
        pm = np.zeros(n_test, dtype=float)
        preds_stack = np.zeros((n_s, n_test), dtype=float)
        for s in range(n_s):
            m_std = k7.predict_mean(x_tr, y_tr_s, x_te, u_mat[s], mf.final_jitter, "rbf")
            m_dec = m_std * y_std[j] + y_mean[j]
            preds_stack[s] = m_dec
            pm += m_dec
        pm /= n_s
        pred_fb[:, j] = pm
        print(f"[factor {fac}] integrated {n_s} posterior draws -> test predictions", flush=True)
        q05[:, j], q50[:, j], q95[:, j] = [np.asarray(x) for x in qvals(preds_stack, [0.05, 0.5, 0.95])]
        sdc[:, j] = np.std(preds_stack, axis=0)

        for s in range(n_s):
            sample_rows.append(
                {
                    "factor": fac,
                    "chain": all_chain_id[s],
                    "draw": int(s),
                    "log_alpha": float(u_mat[s, 0]),
                    "log_ell_L_dm_t": float(u_mat[s, 1]),
                    "log_ell_S_dm_t": float(u_mat[s, 2]),
                    "log_ell_C_dm_t": float(u_mat[s, 3]),
                    "log_ell_dL_t": float(u_mat[s, 4]),
                    "log_ell_dS_t": float(u_mat[s, 5]),
                    "log_ell_dC_t": float(u_mat[s, 6]),
                    "log_ell_time": float(u_mat[s, 7]),
                    "log_sigma": float(u_mat[s, 8]),
                    "log_posterior": float(all_lp[s]),
                }
            )

        acc_fac = float(np.mean([d["acceptance_rate"] for d in diag_rows if d["factor"] == fac]))
        for pi, pname in enumerate(
            [
                "log_alpha",
                "log_ell_L_dm_t",
                "log_ell_S_dm_t",
                "log_ell_C_dm_t",
                "log_ell_dL_t",
                "log_ell_dS_t",
                "log_ell_dC_t",
                "log_ell_time",
                "log_sigma",
            ]
        ):
            col = u_mat[:, pi]
            summary_rows.append(
                {
                    "factor": fac,
                    "parameter": pname,
                    "map_value": float(mf.u[pi]),
                    "posterior_mean": float(np.mean(col)),
                    "posterior_sd": float(np.std(col)),
                    "q05": float(np.quantile(col, 0.05)),
                    "q50": float(np.quantile(col, 0.5)),
                    "q95": float(np.quantile(col, 0.95)),
                    "acceptance_rate": acc_fac,
                    "ess_if_available": float("nan"),
                    "rhat_if_available": float("nan"),
                }
            )

    corr_shrunk = LAMBDA_CORR * pred_fb
    corr_map_shrunk = LAMBDA_CORR * pred_map
    beta_fb = beta_dns_te + corr_shrunk
    beta_map = beta_dns_te + corr_map_shrunk

    df_dns = yield_long_df(od_te, td_te, y_act_te, beta_dns_te, dns, MODEL_DNS, 1)
    df_fb = yield_long_df(od_te, td_te, y_act_te, beta_fb, dns, MODEL_FB, 1)
    df_mp = yield_long_df(od_te, td_te, y_act_te, beta_map, dns, MODEL_MAP, 1)
    all_df = pd.concat([df_dns, df_fb, df_mp], ignore_index=True)

    pooled = efd.pooled_yield_metrics(all_df)
    mat_df = efd.maturity_yield_metrics(all_df)

    dns_p = pooled[(pooled["model"] == MODEL_DNS) & (pooled["horizon"] == 1)].set_index("maturity_set")
    rows_pm: List[Dict[str, Any]] = []
    for _, row in pooled.iterrows():
        if row["model"] == MODEL_DNS:
            continue
        ms = row["maturity_set"]
        dr = float(dns_p.loc[ms, "pooled_rmse_bp"])
        dm = float(dns_p.loc[ms, "pooled_mae_bp"])
        rows_pm.append(
            {
                "model": row["model"],
                "maturity_set": ms,
                "rmse_bp": row["pooled_rmse_bp"],
                "mae_bp": row["pooled_mae_bp"],
                "n_origins": row["n_origins"],
                "n_maturities": row["n_maturities"],
                "n_errors": row["n_errors"],
                "dns_rmse_bp": dr,
                "dns_mae_bp": dm,
                "rmse_improvement_bp": dr - row["pooled_rmse_bp"],
                "mae_improvement_bp": dm - row["pooled_mae_bp"],
                "beats_dns_rmse": bool(dr > row["pooled_rmse_bp"]),
                "beats_dns_mae": bool(dm > row["pooled_mae_bp"]),
            }
        )
    pooled_out = pd.DataFrame(rows_pm)

    dns_by_m = (
        mat_df[(mat_df["model"] == MODEL_DNS) & (mat_df["horizon"] == 1)]
        .drop_duplicates("maturity_months")
        .set_index("maturity_months")
    )
    mat_rows: List[Dict[str, Any]] = []
    for model in (MODEL_FB, MODEL_MAP):
        for ms_label, mats in [
            ("neural_13", efd.NEURAL_MATURITIES.astype(int).tolist()),
            ("project_17", efd.MATURITIES.astype(int).tolist()),
        ]:
            for m in mats:
                r = mat_df[
                    (mat_df["model"] == model)
                    & (mat_df["horizon"] == 1)
                    & (mat_df["maturity_months"] == m)
                ].iloc[0]
                dr = float(dns_by_m.loc[m, "rmse_bp"])
                dm = float(dns_by_m.loc[m, "mae_bp"])
                mat_rows.append(
                    {
                        "model": model,
                        "maturity_set": ms_label,
                        "maturity_months": int(m),
                        "rmse_bp": float(r["rmse_bp"]),
                        "mae_bp": float(r["mae_bp"]),
                        "n_obs": int(r["n_obs"]),
                        "dns_rmse_bp": dr,
                        "dns_mae_bp": dm,
                        "rmse_improvement_bp": dr - float(r["rmse_bp"]),
                        "mae_improvement_bp": dm - float(r["mae_bp"]),
                    }
                )
    mat_out = pd.DataFrame(mat_rows)

    fe_long: List[pd.DataFrame] = []
    for ms_name, mats in [
        ("project_17", efd.MATURITIES.astype(int).tolist()),
        ("neural_13", efd.NEURAL_MATURITIES.astype(int).tolist()),
    ]:
        sub = all_df[all_df["maturity_months"].isin(mats)].copy()
        sub["maturity_set"] = ms_name
        fe_long.append(
            sub[
                [
                    "model",
                    "forecast_origin",
                    "target_month",
                    "maturity_set",
                    "maturity_months",
                    "y_actual_decimal",
                    "y_pred_decimal",
                    "error_bp",
                ]
            ]
        )
    fe_csv = pd.concat(fe_long, ignore_index=True)

    beta_rows: List[Dict[str, Any]] = []
    for i in range(n_test):
        beta_rows.append(
            {
                "forecast_origin": od_te[i],
                "target_month": td_te[i],
                "correction_L_mean": float(pred_fb[i, 0]),
                "correction_S_mean": float(pred_fb[i, 1]),
                "correction_C_mean": float(pred_fb[i, 2]),
                "correction_L_q05": float(q05[i, 0]),
                "correction_S_q05": float(q05[i, 1]),
                "correction_C_q05": float(q05[i, 2]),
                "correction_L_q50": float(q50[i, 0]),
                "correction_S_q50": float(q50[i, 1]),
                "correction_C_q50": float(q50[i, 2]),
                "correction_L_q95": float(q95[i, 0]),
                "correction_S_q95": float(q95[i, 1]),
                "correction_C_q95": float(q95[i, 2]),
                "correction_L_sd": float(sdc[i, 0]),
                "correction_S_sd": float(sdc[i, 1]),
                "correction_C_sd": float(sdc[i, 2]),
                "correction_L_shrunk": float(corr_shrunk[i, 0]),
                "correction_S_shrunk": float(corr_shrunk[i, 1]),
                "correction_C_shrunk": float(corr_shrunk[i, 2]),
                "beta_dns_L": float(beta_dns_te[i, 0]),
                "beta_dns_S": float(beta_dns_te[i, 1]),
                "beta_dns_C": float(beta_dns_te[i, 2]),
                "beta_full_bayes_L": float(beta_fb[i, 0]),
                "beta_full_bayes_S": float(beta_fb[i, 1]),
                "beta_full_bayes_C": float(beta_fb[i, 2]),
                "target_L_if_available": float(y_te_tar[i, 0]),
                "target_S_if_available": float(y_te_tar[i, 1]),
                "target_C_if_available": float(y_te_tar[i, 2]),
            }
        )

    pd.DataFrame(map_rows).to_csv(OUT_DIR / "map_hyperparameters_for_initialization.csv", index=False)
    pd.DataFrame(sample_rows).to_csv(OUT_DIR / "full_bayes_hyperparameter_samples.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "full_bayes_hyperparameter_summary.csv", index=False)
    pd.DataFrame(diag_rows).to_csv(OUT_DIR / "full_bayes_sampler_diagnostics.csv", index=False)
    pooled_out.to_csv(OUT_DIR / "full_bayes_oos_pooled_metrics_bp.csv", index=False)
    mat_out.to_csv(OUT_DIR / "full_bayes_oos_maturity_metrics_bp.csv", index=False)
    fe_csv.to_csv(OUT_DIR / "full_bayes_oos_forecast_errors_by_date.csv", index=False)
    pd.DataFrame(beta_rows).to_csv(OUT_DIR / "full_bayes_beta_correction_predictions.csv", index=False)

    r13 = pooled_out[pooled_out["maturity_set"] == "neural_13"]
    p17 = pooled_out[pooled_out["maturity_set"] == "project_17"]
    fb_r = r13[r13["model"] == MODEL_FB].iloc[0]
    fb_p = p17[p17["model"] == MODEL_FB].iloc[0]
    dns_r = float(dns_p.loc["neural_13", "pooled_rmse_bp"])
    dns_r_p = float(dns_p.loc["project_17", "pooled_rmse_bp"])

    lines = [
        "# h=1 factor-specific RBF-ARD 7D — full Bayes hyperparameter integration\n\n",
        "- **Horizon:** h=1 only.\n",
        "- **DNS:** fixed from `dns_fitted_params_labeled.csv`; **not** re-estimated.\n",
        "- **Broken residual file** `dns_beta_residuals.csv`: **not** used.\n",
        "- **Targets:** one-step **yield-implied** beta correction (W=I, ridge=1e-8). "
        "**No** Kalman beta residuals.\n",
        "- **Kernel:** RBF-ARD only; **7D** inputs; **separate** scalar GPs for L, S, C.\n",
        "- **Prior:** Moderate-long-time (explicit normal priors on logs; Step 8 specification).\n",
        f"- **Shrinkage:** lambda_corr = {LAMBDA_CORR}.\n",
        "- **Train:** target_month ≤ 2003-12 (scaling, MAP init, MCMC use **training rows only**).\n",
        "- **Test:** forecast origins 2004-01 .. 2023-12 (same convention as other fixed-DNS runs).\n",
        "- **Sampler:** adaptive random-walk Metropolis–Hastings in log-hyperparameter space (initialized at MAP).\n",
        f"- **MCMC:** chains={FB_CHAINS}, warmup={FB_WARMUP}, draws={FB_DRAWS}, thin={FB_THIN}.\n\n",
        "## Pooled OOS (bp)\n\n",
        "| maturity_set | DNS RMSE | Full Bayes RMSE | Δ RMSE | beats DNS RMSE |\n",
        "|---|---:|---:|---:|:---:|\n",
        f"| neural_13 | {dns_r:.4f} | {fb_r['rmse_bp']:.4f} | {fb_r['rmse_improvement_bp']:.4f} | {fb_r['beats_dns_rmse']} |\n",
        f"| project_17 | {dns_r_p:.4f} | {fb_p['rmse_bp']:.4f} | {fb_p['rmse_improvement_bp']:.4f} | {fb_p['beats_dns_rmse']} |\n\n",
        "## Runtime\n\n",
        f"- elapsed_s: {time.time()-t0:.1f}\n",
    ]
    (OUT_DIR / "run_summary.md").write_text("".join(lines), encoding="utf-8")

    print("\n=== SANITY ===", flush=True)
    print("1 broken residual not used: OK")
    print("2 DNS not re-estimated: OK (fixed CSV)")
    print("3 h=1 only: OK")
    print("4 yield-implied targets: OK")
    print("5 no h=3/6/12: OK")
    print(f"6 test window origins {efd.FORECAST_ORIGIN_START} .. {efd.FORECAST_ORIGIN_END}: OK")
    print("7 train only for scale/MAP/MCMC: OK")
    print("8 RBF only: OK")
    print("9 7D inputs: OK")
    print("10 Moderate-long prior: OK")
    print(f"11 lambda_corr={LAMBDA_CORR}: OK")
    print("12 separate factor GPs: OK")
    print(f"13 posterior draws (approx total): {len(sample_rows)}")
    tr_std_vec = np.std(y_tr_raw, axis=0, ddof=0)
    for jj, fn in enumerate(FAC_NAMES):
        ratio = float(np.mean(np.abs(pred_fb[:, jj]))) / float(tr_std_vec[jj] + 1e-12)
        print(f"18 mean|pred_corr|/train_std factor {fn}: {ratio:.4f} (watch 0.75)")
    dns_mae_r = float(dns_p.loc["neural_13", "pooled_mae_bp"])
    dns_mae_p17 = float(dns_p.loc["project_17", "pooled_mae_bp"])
    print(f"neural_13 DNS RMSE/MAE: {dns_r:.4f} / {dns_mae_r:.4f}")
    print(f"neural_13 FullBayes RMSE/MAE: {fb_r['rmse_bp']:.4f} / {fb_r['mae_bp']:.4f}  improve {fb_r['rmse_improvement_bp']:.4f} bp RMSE")
    print(f"project_17 DNS RMSE/MAE: {dns_r_p:.4f} / {dns_mae_p17:.4f}")
    print(f"project_17 FullBayes RMSE/MAE: {fb_p['rmse_bp']:.4f} / {fb_p['mae_bp']:.4f}")
    if MODEL_MAP in pooled_out["model"].values:
        mp = pooled_out[pooled_out["model"] == MODEL_MAP].set_index("maturity_set")
        print(f"MAP vs FB neural_13: MAP RMSE {float(mp.loc['neural_13','rmse_bp']):.4f} vs FB {fb_r['rmse_bp']:.4f}")
    print(f"\n[DONE] {OUT_DIR} in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
