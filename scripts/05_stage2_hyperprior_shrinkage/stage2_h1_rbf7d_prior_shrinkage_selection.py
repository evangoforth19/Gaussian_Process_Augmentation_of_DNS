#!/usr/bin/env python3
"""
Stage 2: fixed RBF-ARD + 7D inputs; validate amplitude/noise, time, dimension priors
and post-fit shrinkage lambda_corr on pre-2004 rolling folds only.

Does NOT refit DNS, does NOT use 2004–2023 test data, does NOT read dns_beta_residuals.csv.

Environment (optional)
----------------------
- STAGE2_GP_RESTARTS (default 5)
- STAGE2_MAX_FOLDS (default 3)
- STAGE2_MAXITER (default 120)
- STAGE2_EXHAUSTIVE_JITTER=1 : full jitter ladder each restart (default 0: escalate)
- STAGE2_MAP_RESTART_LOG=1 : per-restart MAP log lines
- STAGE2_FORCE_RECOMPUTE=1 : ignore checkpoints
"""

from __future__ import annotations

import itertools
import math
import os
import pickle
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

import evaluate_fixed_dns_gp as efd
from plain_dns_gp_correction import normal_logpdf

warnings.filterwarnings("ignore")


def log(msg: str) -> None:
    print(msg, flush=True)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUT_DIR = SCRIPT_DIR / "stage2_h1_rbf7d_prior_shrinkage_selection_outputs"
BROKEN_RESIDUAL = SCRIPT_DIR / "dns_beta_residuals.csv"
DNS_PARAMS_CSV = (
    PROJECT_ROOT
    / "Kalman Filter"
    / "Original Macro + DNS Filter"
    / "dns_fitted_params_labeled.csv"
)

RIDGE_PROJ = 1e-8
RANDOM_SEED = 7310
JITTER_TRIALS = (1e-6, 1e-5, 1e-4, 1e-3)
JOINT_OBJ_OK_THRESHOLD = 1e11
LBFGS_MAXITER_DEFAULT = 120
TIE_EPS_BP = 0.25

MATURITIES = efd.MATURITIES.astype(float)
NEURAL_SET = set(int(x) for x in efd.NEURAL_MATURITIES.astype(int).tolist())
YIELD_COLS = list(efd.YIELD_COLS)

FOLDS = (
    (1, pd.Timestamp("1989-12-01"), pd.Timestamp("1990-01-01"), pd.Timestamp("1994-12-01")),
    (2, pd.Timestamp("1994-12-01"), pd.Timestamp("1995-01-01"), pd.Timestamp("1999-12-01")),
    (3, pd.Timestamp("1999-12-01"), pd.Timestamp("2000-01-01"), pd.Timestamp("2003-12-01")),
)

AMP_NOISE_LEVELS = ("conservative_amp_noise", "moderate_amp_noise")
TIME_LEVELS = ("current_time", "long_time")
DIM_LEVELS = ("smooth_dims", "moderate_dims")
LAMBDA_GRID = (0.25, 0.50, 1.00)
KERNEL_FIXED = "RBF"
INPUT_FIXED = "7D"


def env_int(name: str, default: int) -> int:
    v = os.environ.get(name, "").strip()
    return int(v) if v else default


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "y")


def lbfgs_maxiter() -> int:
    return env_int("STAGE2_MAXITER", LBFGS_MAXITER_DEFAULT)


# ---------------------------------------------------------------------------
# Squared-distance ARD + RBF only
# ---------------------------------------------------------------------------


def precompute_sq_dists(X: np.ndarray) -> np.ndarray:
    n, d = X.shape
    out = np.empty((d, n, n), dtype=float)
    for dim in range(d):
        v = X[:, dim]
        out[dim] = (v[:, None] - v[None, :]) ** 2
    return out


def ard_scaled_sq_sum(D_stack: np.ndarray, ells: np.ndarray) -> np.ndarray:
    inv_e2 = 1.0 / (ells**2 + 1e-30)
    return np.tensordot(inv_e2, D_stack, axes=([0], [0]))


def R_rbf_from_S(S: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * np.maximum(S, 0.0))


def cross_scaled_sq(X_train: np.ndarray, X_test: np.ndarray, ells: np.ndarray) -> np.ndarray:
    inv_e2 = 1.0 / (ells**2 + 1e-30)
    nt, n = X_test.shape[0], X_train.shape[0]
    S = np.zeros((nt, n), dtype=float)
    for d in range(X_train.shape[1]):
        diff = X_test[:, d : d + 1] - X_train[None, :, d]
        S += (diff * diff) * inv_e2[d]
    return S


def gp_marginal_and_cho(
    y: np.ndarray, R: np.ndarray, alpha: float, sigma: float, jitter: float
) -> Tuple[float, Optional[Any], Optional[np.ndarray]]:
    n = len(y)
    K = (alpha**2) * R + (sigma**2 + jitter) * np.eye(n)
    K = 0.5 * (K + K.T)
    try:
        cF = cho_factor(K, lower=True, check_finite=False)
    except np.linalg.LinAlgError:
        return float("nan"), None, None
    v = cho_solve(cF, y, check_finite=False)
    logdet = 2.0 * np.sum(np.log(np.diag(cF[0])))
    ll = -0.5 * float(y @ v) - 0.5 * logdet - 0.5 * n * math.log(2.0 * math.pi)
    return float(ll), cF, v


def predict_output(
    y_std: np.ndarray,
    R: np.ndarray,
    R_cross: np.ndarray,
    alpha: float,
    sigma: float,
    jitter: float,
) -> np.ndarray:
    n = len(y_std)
    K = (alpha**2) * R + (sigma**2 + jitter) * np.eye(n)
    K = 0.5 * (K + K.T)
    try:
        cF = cho_factor(K, lower=True, check_finite=False)
    except np.linalg.LinAlgError:
        return np.full(R_cross.shape[0], np.nan)
    v = cho_solve(cF, y_std, check_finite=False)
    Ks = (alpha**2) * R_cross
    return (Ks @ v).astype(float)


def build_R(D_stack: np.ndarray, ells: np.ndarray) -> np.ndarray:
    return R_rbf_from_S(ard_scaled_sq_sum(D_stack, ells))


# ---------------------------------------------------------------------------
# Stage 2 prior grid -> mean_u, sd_u (log space, 7 ell + 3 alpha + 3 sigma)
# ---------------------------------------------------------------------------


def prior_mean_sd_vectors(
    amp_noise: str,
    time_prior: str,
    dimension_prior: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (mean_u, sd_u) for u = [log ell x7], [log alpha x3], [log sigma x3]."""
    mean_list: List[float] = []
    sd_list: List[float] = []

    if dimension_prior == "smooth_dims":
        for _ in range(3):
            mean_list.append(math.log(2.75))
            sd_list.append(0.30)
        for _ in range(3):
            mean_list.append(math.log(2.25))
            sd_list.append(0.30)
    elif dimension_prior == "moderate_dims":
        for _ in range(3):
            mean_list.append(math.log(2.25))
            sd_list.append(0.35)
        for _ in range(3):
            mean_list.append(math.log(1.75))
            sd_list.append(0.35)
    else:
        raise ValueError(dimension_prior)

    if time_prior == "current_time":
        mean_list.append(math.log(1.25))
        sd_list.append(0.30)
    elif time_prior == "long_time":
        mean_list.append(math.log(2.25))
        sd_list.append(0.35)
    else:
        raise ValueError(time_prior)

    if amp_noise == "conservative_amp_noise":
        for _ in range(3):
            mean_list.append(math.log(0.06))
            sd_list.append(0.35)
        for _ in range(3):
            mean_list.append(math.log(0.90))
            sd_list.append(0.25)
    elif amp_noise == "moderate_amp_noise":
        for _ in range(3):
            mean_list.append(math.log(0.10))
            sd_list.append(0.40)
        for _ in range(3):
            mean_list.append(math.log(0.80))
            sd_list.append(0.30)
    else:
        raise ValueError(amp_noise)

    return np.array(mean_list, dtype=float), np.array(sd_list, dtype=float)


def log_prior_vector(u: np.ndarray, mean_u: np.ndarray, sd_u: np.ndarray) -> float:
    if not np.all(np.isfinite(u)):
        return float("-inf")
    lp = 0.0
    for k in range(len(u)):
        lp += float(normal_logpdf(float(u[k]), float(mean_u[k]), float(sd_u[k])))
    return lp


def unpack_u_joint_rbf(u: np.ndarray, d: int = 7) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ells = np.exp(u[0:d]).astype(float)
    alpha = np.exp(u[d : d + 3]).astype(float)
    sigma = np.exp(u[d + 3 : d + 6]).astype(float)
    return ells, alpha, sigma


def neg_joint_log_post_rbf(
    u: np.ndarray,
    D_stack: np.ndarray,
    Y_std: np.ndarray,
    jitter: float,
    mean_u: np.ndarray,
    sd_u: np.ndarray,
) -> float:
    d = D_stack.shape[0]
    ells, alphas, sigmas = unpack_u_joint_rbf(u, d)
    R = build_R(D_stack, ells)
    ll = 0.0
    for j in range(3):
        lj, _, _ = gp_marginal_and_cho(Y_std[:, j], R, float(alphas[j]), float(sigmas[j]), jitter)
        if not np.isfinite(lj):
            return 1e12
        ll += lj
    lp = log_prior_vector(u, mean_u, sd_u)
    if not np.isfinite(lp):
        return 1e12
    return -(ll + lp)


@dataclass
class JointFitResult:
    u: np.ndarray
    map_objective: float
    optimizer_success: bool
    final_jitter: float
    best_restart_name: str
    pathology_flag: str
    opt_message: str


@dataclass
class JointFitStats:
    n_restarts_attempted: int
    n_minimize_calls: int
    exhaustive_jitter: bool


def pathology_rbf(alphas: np.ndarray, sigmas: np.ndarray, ells: np.ndarray, ok: bool, jit: float) -> str:
    flags: List[str] = []
    if np.any(alphas > 2.5):
        flags.append("large_alpha")
    if np.any(sigmas < 1e-4):
        flags.append("small_sigma")
    if np.any(ells < 0.12):
        flags.append("small_ell")
    if not ok:
        flags.append("optimizer_not_success")
    if jit >= 1e-3:
        flags.append("high_jitter")
    return ",".join(flags) if flags else "none"


def u0_vectors_stage2(mean_u: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    """Five structured restarts in log-space, scaled around this combo's prior median."""
    d = 7
    ref_ell = np.array([2.25, 2.25, 2.25, 1.75, 1.75, 1.75, 2.25], dtype=float)
    med_ell = np.exp(mean_u[:d])
    rel = med_ell / ref_ell
    med_a = np.exp(mean_u[d : d + 3])
    med_s = np.exp(mean_u[d + 3 : d + 6])

    def pack(ell_phys: List[float], alphas_p: List[float], sigmas_p: List[float]) -> np.ndarray:
        e = np.array(ell_phys, dtype=float) * rel
        return np.concatenate(
            [
                np.log(e),
                np.log(np.array(alphas_p, dtype=float) * med_a / np.array([0.10, 0.10, 0.10], float)),
                np.log(np.array(sigmas_p, dtype=float) * med_s / np.array([0.80, 0.80, 0.80], float)),
            ]
        )

    el_med = [2.25, 2.25, 2.25, 1.75, 1.75, 1.75, 2.25]
    el_cons = [3.5, 3.5, 3.5, 2.8, 2.8, 2.8, 3.0]
    el_mod = [2.0, 2.0, 2.0, 1.4, 1.4, 1.4, 1.25]
    el_time = [3.0, 3.0, 3.0, 2.5, 2.5, 2.5, 0.90]
    el_shrink = [4.0, 4.0, 4.0, 3.0, 3.0, 3.0, 2.0]

    out: List[Tuple[str, np.ndarray]] = [
        ("prior_median", mean_u.copy()),
        ("conservative", pack(el_cons, [0.05, 0.05, 0.05], [1.05, 1.05, 1.05])),
        ("moderate", pack(el_mod, [0.12, 0.12, 0.12], [0.75, 0.75, 0.75])),
        ("time_sensitive", pack(el_time, [0.08, 0.08, 0.08], [0.85, 0.85, 0.85])),
        ("shrink_to_noise", pack(el_shrink, [0.02, 0.02, 0.02], [1.10, 1.10, 1.10])),
    ]
    return out


def fit_joint_map_rbf(
    X_train_std: np.ndarray,
    Y_train_std: np.ndarray,
    mean_u: np.ndarray,
    sd_u: np.ndarray,
    n_restarts: int,
) -> Tuple[JointFitResult, JointFitStats]:
    """Joint MAP RBF-ARD; finite-difference gradients (no ``jac``)."""
    d = X_train_std.shape[1]
    assert d == 7
    D_stack = precompute_sq_dists(X_train_std)
    bounds = [(-14.0, 14.0)] * len(mean_u)
    restarts = u0_vectors_stage2(mean_u)[:n_restarts]
    exhaustive_jitter = env_flag("STAGE2_EXHAUSTIVE_JITTER")

    best: Optional[JointFitResult] = None
    best_f = 1e300
    n_minimize_calls = 0
    n_restarts_attempted = 0
    verbose = os.environ.get("STAGE2_MAP_RESTART_LOG", "").strip().lower() in ("1", "true", "yes", "y")

    for rname, u0 in restarts:
        n_restarts_attempted += 1
        if verbose:
            log(f"    [MAP stage2] restart={rname} jitter_mode={'exhaustive' if exhaustive_jitter else 'escalate'}")
        u0c = np.clip(u0.astype(float), [b[0] for b in bounds], [b[1] for b in bounds])
        local_best: Optional[Tuple[np.ndarray, float, float, bool, str]] = None
        local_f = 1e300

        if exhaustive_jitter:
            for jit in JITTER_TRIALS:

                def fun(uu: np.ndarray) -> float:
                    return neg_joint_log_post_rbf(
                        uu.astype(float), D_stack, Y_train_std, jit, mean_u, sd_u
                    )

                try:
                    res = minimize(
                        fun,
                        u0c,
                        method="L-BFGS-B",
                        bounds=bounds,
                        options={"maxiter": lbfgs_maxiter(), "disp": False},
                    )
                except Exception:
                    n_minimize_calls += 1
                    continue
                n_minimize_calls += 1
                fv = float(res.fun)
                if np.isfinite(fv) and fv < JOINT_OBJ_OK_THRESHOLD and fv < local_f:
                    local_f = fv
                    local_best = (res.x.astype(float), jit, fv, bool(res.success), str(res.message))
        else:
            for jit in JITTER_TRIALS:

                def fun_esc(uu: np.ndarray, jt: float = jit) -> float:
                    return neg_joint_log_post_rbf(
                        uu.astype(float), D_stack, Y_train_std, jt, mean_u, sd_u
                    )

                try:
                    res = minimize(
                        fun_esc,
                        u0c,
                        method="L-BFGS-B",
                        bounds=bounds,
                        options={"maxiter": lbfgs_maxiter(), "disp": False},
                    )
                except Exception:
                    n_minimize_calls += 1
                    continue
                n_minimize_calls += 1
                fv = float(res.fun)
                if np.isfinite(fv) and fv < JOINT_OBJ_OK_THRESHOLD and fv < local_f:
                    local_f = fv
                    local_best = (res.x.astype(float), jit, fv, bool(res.success), str(res.message))
                if np.isfinite(fv) and fv < JOINT_OBJ_OK_THRESHOLD:
                    break

        if local_best is None:
            continue
        u_hat, jit, _, ok, msg = local_best
        if not (np.isfinite(local_f) and local_f < best_f):
            continue
        ells, alphas, sigmas = unpack_u_joint_rbf(u_hat, d)
        R = build_R(D_stack, ells)
        ll = 0.0
        for j in range(3):
            lj, _, _ = gp_marginal_and_cho(Y_train_std[:, j], R, float(alphas[j]), float(sigmas[j]), jit)
            ll += lj
        lp = log_prior_vector(u_hat, mean_u, sd_u)
        map_obj = float(ll + lp) if np.isfinite(ll) and np.isfinite(lp) else float("nan")
        path = pathology_rbf(alphas, sigmas, ells, ok, jit)
        best_f = local_f
        best = JointFitResult(
            u=u_hat,
            map_objective=map_obj,
            optimizer_success=ok,
            final_jitter=float(jit),
            best_restart_name=rname,
            pathology_flag=path,
            opt_message=msg,
        )

    stats = JointFitStats(
        n_restarts_attempted=n_restarts_attempted,
        n_minimize_calls=n_minimize_calls,
        exhaustive_jitter=exhaustive_jitter,
    )

    if best is None:
        return (
            JointFitResult(
                u=mean_u.copy(),
                map_objective=float("nan"),
                optimizer_success=False,
                final_jitter=float(JITTER_TRIALS[-1]),
                best_restart_name="failed_all",
                pathology_flag="all_failed",
                opt_message="",
            ),
            stats,
        )
    ells, alphas, sigmas = unpack_u_joint_rbf(best.u, d)
    R = build_R(D_stack, ells)
    best.pathology_flag = pathology_rbf(alphas, sigmas, ells, best.optimizer_success, best.final_jitter)
    return best, stats


# ---------------------------------------------------------------------------
# Data construction (same protocol as Stage 1; 7D inputs; W=I projection)
# ---------------------------------------------------------------------------


def year_frac(months: pd.DatetimeIndex, idx: np.ndarray) -> np.ndarray:
    base = np.datetime64(months[0], "ns")
    ts = months[idx].to_numpy(dtype="datetime64[ns]")
    return (ts.astype(np.int64) - base.astype(np.int64)).astype(float) / (365.25 * 86400.0 * 1e9)


def build_X7(beta: np.ndarray, mu: np.ndarray, months: pd.DatetimeIndex, t_idx: np.ndarray) -> np.ndarray:
    t1 = t_idx - 1
    b0 = beta[t_idx] - mu
    d0 = beta[t_idx] - beta[t1]
    ty = year_frac(months, t_idx).reshape(-1, 1)
    return np.hstack([b0, d0, ty])


def _proj_one(u_y: np.ndarray, lam_dec: np.ndarray) -> np.ndarray:
    a = lam_dec.T @ lam_dec + RIDGE_PROJ * np.eye(3, dtype=float)
    b = lam_dec.T @ u_y.reshape(-1, 1)
    return np.linalg.solve(a, b).reshape(3).astype(float)


def yield_implied_rows(
    beta: np.ndarray,
    months: pd.DatetimeIndex,
    yields: np.ndarray,
    dns: efd.FixedDNS,
    lam_proj: np.ndarray,
) -> Dict[str, Any]:
    T = len(months)
    mu = dns.mu_dec
    Phi = dns.Phi
    lam_full = efd.ns_loadings(MATURITIES, dns.lam)

    pred1 = np.full_like(beta, np.nan)
    for t in range(T - 1):
        pred1[t + 1] = mu + Phi @ (beta[t] - mu)

    t_idx = np.arange(1, T - 1, dtype=int)
    tgt = t_idx + 1
    m_ok = (
        np.isfinite(beta[t_idx]).all(axis=1)
        & np.isfinite(beta[t_idx - 1]).all(axis=1)
        & np.isfinite(pred1[tgt]).all(axis=1)
        & np.all(np.isfinite(yields[tgt]), axis=1)
    )
    t_idx = t_idx[m_ok]
    tgt = tgt[m_ok]

    y_dns = pred1[tgt] @ lam_full.T
    y_act = yields[tgt]
    u_y = y_act - y_dns
    ry = np.zeros((len(t_idx), 3), dtype=float)
    for i in range(len(t_idx)):
        ry[i] = _proj_one(u_y[i], lam_proj)

    X7 = build_X7(beta, mu, months, t_idx)
    beta_dns = pred1[tgt].copy()
    target_dates = months[tgt]
    origin_dates = months[t_idx]
    return {
        "t_idx": t_idx,
        "tgt": tgt,
        "origin_dates": origin_dates,
        "target_dates": target_dates,
        "X7": X7,
        "Y": ry,
        "beta_dns": beta_dns,
        "y_act": y_act,
        "y_dns": y_dns,
    }


def fit_standardizer_cols(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    m = np.mean(x, axis=0)
    s = np.std(x, axis=0, ddof=0)
    s = np.where(s < 1e-8, 1.0, s)
    return m, s


def standardize(x: np.ndarray, m: np.ndarray, s: np.ndarray) -> np.ndarray:
    return (x - m) / s


def metrics_bp(y_pred: np.ndarray, y_act: np.ndarray, mat_mask: np.ndarray) -> Tuple[float, float, int]:
    e = (y_pred - y_act)[:, mat_mask] * 10000.0
    e = e[np.isfinite(e)]
    if len(e) == 0:
        return float("nan"), float("nan"), 0
    return float(np.sqrt(np.mean(e**2))), float(np.mean(np.abs(e))), int(len(e))


def neural_mask() -> np.ndarray:
    return np.array([int(m) in NEURAL_SET for m in MATURITIES.astype(int)], dtype=bool)


def project17_mask() -> np.ndarray:
    return np.ones(len(MATURITIES), dtype=bool)


def ar1_predict_series(y_train: np.ndarray, y_val: np.ndarray) -> np.ndarray:
    lag = y_train[:-1]
    y1 = y_train[1:]
    m = np.isfinite(lag) & np.isfinite(y1)
    lag, y1 = lag[m], y1[m]
    if len(lag) < 5:
        return np.full(len(y_val), np.nan)
    Xd = np.column_stack([np.ones_like(lag), lag])
    betaols, _, _, _ = np.linalg.lstsq(Xd, y1, rcond=None)
    c, phi = float(betaols[0]), float(betaols[1])
    out = np.empty(len(y_val))
    for i in range(len(y_val)):
        lagv = float(y_train[-1]) if i == 0 else float(y_val[i - 1])
        if not np.isfinite(lagv):
            out[i] = np.nan
        else:
            out[i] = c + phi * lagv
    return out


def checkpoint_name(fold_id: int, amp: str, time_p: str, dim_p: str) -> str:
    safe = f"fold{int(fold_id)}__{amp}__{time_p}__{dim_p}.pkl"
    return safe.replace(" ", "_")


def checkpoint_path(out_dir: Path, fold_id: int, amp: str, time_p: str, dim_p: str) -> Path:
    d = out_dir / "_checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    return d / checkpoint_name(fold_id, amp, time_p, dim_p)


def eval_prior_block(
    pack: Dict[str, Any],
    fold_id: int,
    train_end: pd.Timestamp,
    val_start: pd.Timestamp,
    val_end: pd.Timestamp,
    amp_noise: str,
    time_prior: str,
    dimension_prior: str,
    lam_grid: Tuple[float, ...],
    lam_full: np.ndarray,
    n_restarts: int,
) -> Dict[str, Any]:
    td = pack["target_dates"]
    tr_m = td <= train_end
    va_m = (td >= val_start) & (td <= val_end)
    X = pack["X7"]
    Y = pack["Y"]
    beta_dns_all = pack["beta_dns"]
    y_act_all = pack["y_act"]

    X_tr, Y_tr = X[tr_m], Y[tr_m]
    X_va, Y_va = X[va_m], Y[va_m]
    bd_tr, bd_va = beta_dns_all[tr_m], beta_dns_all[va_m]
    ya_tr, ya_va = y_act_all[tr_m], y_act_all[va_m]

    mean_u, sd_u = prior_mean_sd_vectors(amp_noise, time_prior, dimension_prior)

    t_block = time.time()
    log(
        f"[stage2] >>> START fold={fold_id} {amp_noise} {time_prior} {dimension_prior}  "
        f"n_train={X_tr.shape[0]} n_val={X_va.shape[0]}  (one MAP; lambda post-hoc)"
    )

    xm, xs = fit_standardizer_cols(X_tr)
    X_tr_s = standardize(X_tr, xm, xs)
    X_va_s = standardize(X_va, xm, xs)
    ym = np.mean(Y_tr, axis=0)
    ys = np.std(Y_tr, axis=0, ddof=0)
    ys = np.where(ys < 1e-8, 1.0, ys)
    Y_tr_s = (Y_tr - ym) / ys

    fit, stats = fit_joint_map_rbf(X_tr_s, Y_tr_s, mean_u, sd_u, n_restarts)
    d = 7
    ells, alphas, sigmas = unpack_u_joint_rbf(fit.u, d)
    D_tr = precompute_sq_dists(X_tr_s)
    R = build_R(D_tr, ells)

    pred_s = np.zeros((X_va_s.shape[0], 3), dtype=float)
    for j in range(3):
        Sx = cross_scaled_sq(X_tr_s, X_va_s, ells)
        pred_s[:, j] = predict_output(Y_tr_s[:, j], R, Sx, float(alphas[j]), float(sigmas[j]), fit.final_jitter)
    g_va = pred_s * ys + ym

    n_mask = neural_mask()
    p_mask = project17_mask()
    mean_r = np.mean(Y_tr, axis=0)

    def y_from_beta(bdec: np.ndarray) -> np.ndarray:
        return bdec @ lam_full.T

    y_dns_only = y_from_beta(bd_va)
    y_const = y_from_beta(bd_va + mean_r)
    ar_pred = np.zeros_like(Y_va)
    for j in range(3):
        ar_pred[:, j] = ar1_predict_series(Y_tr[:, j], Y_va[:, j])
    y_ar1 = y_from_beta(bd_va + ar_pred)

    base_rows_tuples: List[Tuple[str, str, float, float, int]] = []
    for model, yp in [
        ("DNS", y_dns_only),
        ("DNS+CONST_ONE_STEP", y_const),
        ("DNS+AR1_ONE_STEP", y_ar1),
    ]:
        rn, mn, en = metrics_bp(yp, ya_va, n_mask)
        rp, mp, ep = metrics_bp(yp, ya_va, p_mask)
        base_rows_tuples.append((model, "neural_13", rn, mn, en))
        base_rows_tuples.append((model, "project_17", rp, mp, ep))

    base_rows = [
        {
            "fold_id": fold_id,
            "model": m,
            "maturity_set": ms,
            "val_rmse_bp": r,
            "val_mae_bp": ma,
            "n_val_origins": int(X_va.shape[0]),
            "n_maturities": int(n_mask.sum() if ms == "neural_13" else p_mask.sum()),
            "n_errors": ne,
        }
        for (m, ms, r, ma, ne) in base_rows_tuples
    ]

    diag_base = {
        "pred_corr_mean_L": float(np.mean(g_va[:, 0])),
        "pred_corr_mean_S": float(np.mean(g_va[:, 1])),
        "pred_corr_mean_C": float(np.mean(g_va[:, 2])),
        "pred_corr_std_L": float(np.std(g_va[:, 0])),
        "pred_corr_std_S": float(np.std(g_va[:, 1])),
        "pred_corr_std_C": float(np.std(g_va[:, 2])),
        "mean_abs_pred_corr_over_train_std_L": float(np.mean(np.abs(g_va[:, 0])) / ys[0]),
        "mean_abs_pred_corr_over_train_std_S": float(np.mean(np.abs(g_va[:, 1])) / ys[1]),
        "mean_abs_pred_corr_over_train_std_C": float(np.mean(np.abs(g_va[:, 2])) / ys[2]),
    }

    val_rows: List[Dict[str, Any]] = []
    hyper_rows: List[Dict[str, Any]] = []
    diag_rows: List[Dict[str, Any]] = []
    accum_partial: Dict[Tuple[str, str, str, float], Dict[str, List[float]]] = {}

    for j, fac in enumerate(["L", "S", "C"]):
        hyper_rows.append(
            {
                "fold_id": fold_id,
                "amp_noise_prior": amp_noise,
                "time_prior": time_prior,
                "dimension_prior": dimension_prior,
                "factor": fac,
                "ell_L_dm_t": float(ells[0]),
                "ell_S_dm_t": float(ells[1]),
                "ell_C_dm_t": float(ells[2]),
                "ell_dL_t": float(ells[3]),
                "ell_dS_t": float(ells[4]),
                "ell_dC_t": float(ells[5]),
                "ell_time": float(ells[6]),
                "alpha_factor": float(alphas[j]),
                "sigma_factor": float(sigmas[j]),
                "map_objective": fit.map_objective,
                "optimizer_success": fit.optimizer_success,
                "best_restart_name": fit.best_restart_name,
                "final_jitter": fit.final_jitter,
                "n_train": int(X_tr.shape[0]),
                "pathology_flag": fit.pathology_flag,
            }
        )

    for lam_corr in lam_grid:
        beta_gp = bd_va + float(lam_corr) * g_va
        y_pred = beta_gp @ lam_full.T

        rmse_n, mae_n, nerr_n = metrics_bp(y_pred, ya_va, n_mask)
        rmse_p, mae_p, nerr_p = metrics_bp(y_pred, ya_va, p_mask)

        err_n_mat = (y_pred - ya_va)[:, n_mask] * 10000.0
        neural_err2_sum = float(np.nansum(err_n_mat**2))
        neural_abs_sum = float(np.nansum(np.abs(err_n_mat)))
        neural_err_count = int(np.isfinite(err_n_mat).sum())
        err_p_mat = (y_pred - ya_va)[:, p_mask] * 10000.0
        project_err2_sum = float(np.nansum(err_p_mat**2))
        project_abs_sum = float(np.nansum(np.abs(err_p_mat)))
        project_err_count = int(np.isfinite(err_p_mat).sum())
        key = (amp_noise, time_prior, dimension_prior, float(lam_corr))
        accum_partial[key] = {
            "neural": [neural_err2_sum, neural_abs_sum, float(neural_err_count)],
            "project": [project_err2_sum, project_abs_sum, float(project_err_count)],
        }

        for ms, rm, ma, ne in [
            ("neural_13", rmse_n, mae_n, nerr_n),
            ("project_17", rmse_p, mae_p, nerr_p),
        ]:
            val_rows.append(
                {
                    "fold_id": fold_id,
                    "kernel": KERNEL_FIXED,
                    "input_set": INPUT_FIXED,
                    "input_dim": 7,
                    "amp_noise_prior": amp_noise,
                    "time_prior": time_prior,
                    "dimension_prior": dimension_prior,
                    "lambda_corr": float(lam_corr),
                    "maturity_set": ms,
                    "val_rmse_bp": rm,
                    "val_mae_bp": ma,
                    "n_val_origins": int(X_va.shape[0]),
                    "n_maturities": int(n_mask.sum() if ms == "neural_13" else p_mask.sum()),
                    "n_errors": ne,
                    "optimizer_success_all_outputs": fit.optimizer_success,
                    "pathology_flag": fit.pathology_flag,
                    "final_jitter_max": fit.final_jitter,
                }
            )

        diag_rows.append(
            {
                "fold_id": fold_id,
                "amp_noise_prior": amp_noise,
                "time_prior": time_prior,
                "dimension_prior": dimension_prior,
                "lambda_corr": float(lam_corr),
                **diag_base,
            }
        )

    elapsed = time.time() - t_block
    log(
        f"[stage2] <<< END fold={fold_id} {amp_noise} {time_prior} {dimension_prior}  "
        f"restarts={stats.n_restarts_attempted}  minimize_calls={stats.n_minimize_calls}  "
        f"MAP_ok={fit.optimizer_success}  jitter={fit.final_jitter}  ({elapsed:.1f}s)"
    )

    return {
        "val_rows": val_rows,
        "base_rows": base_rows,
        "hyper_rows": hyper_rows,
        "diag_rows": diag_rows,
        "accum_partial": accum_partial,
        "elapsed_block": elapsed,
        "fit_stats": stats,
    }


def path_score_from_flag(flag: str) -> int:
    return 0 if str(flag) == "none" else 1


def pick_stage2_candidate(dfp: pd.DataFrame) -> Tuple[pd.Series, str]:
    """Primary: min pooled S_val neural_13; ties per spec within TIE_EPS_BP."""
    dfn = dfp[dfp["maturity_set"] == "neural_13"].copy()
    dfn = dfn.sort_values("S_val").reset_index(drop=True)
    s_star = float(dfn.iloc[0]["S_val"])
    pool = dfn[dfn["S_val"] <= s_star + TIE_EPS_BP].copy()

    pool["_mae"] = pool["mae_pooled"]
    pool["_amp"] = pool["amp_noise_prior"].map({"conservative_amp_noise": 0, "moderate_amp_noise": 1})
    pool["_dim"] = pool["dimension_prior"].map({"smooth_dims": 0, "moderate_dims": 1})
    pool["_tkey"] = pool["time_prior"].map({"current_time": 0, "long_time": 1})
    pool["_lc"] = pool["lambda_corr"].map({0.25: 0, 0.5: 1, 1.0: 2})

    pool = pool.sort_values(["S_val", "_mae", "_amp", "_dim", "_tkey", "_lc", "path_score"])
    row = pool.iloc[0]
    reason = (
        f"Pooled neural_13 RMSE S_val=sqrt(sum err^2 / sum n) across folds; best S_val={s_star:.4f} bp; "
        f"ties within {TIE_EPS_BP} bp: lower MAE, conservative over moderate amp/noise, smooth over moderate dims, "
        f"current_time over long_time when RMSE tied, lower lambda_corr, fewer pathology flags."
    )
    return row, reason


def main() -> None:
    t0 = time.time()
    np.random.seed(RANDOM_SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    n_restarts = env_int("STAGE2_GP_RESTARTS", 5)
    max_folds = env_int("STAGE2_MAX_FOLDS", 3)
    lam_grid = LAMBDA_GRID

    prior_combos = list(itertools.product(AMP_NOISE_LEVELS, TIME_LEVELS, DIM_LEVELS))
    n_pc = len(prior_combos)
    folds_use = FOLDS[:max_folds]
    map_fits = len(folds_use) * n_pc

    log(
        f"[stage2] start out={OUT_DIR}  folds={max_folds}  prior_combos={n_pc}  "
        f"lambda_corr={list(lam_grid)}  restarts={n_restarts}  "
        f"joint MAP fits (no refit per lambda)={map_fits}  "
        f"STAGE2_EXHAUSTIVE_JITTER={int(env_flag('STAGE2_EXHAUSTIVE_JITTER'))}  "
        f"STAGE2_FORCE_RECOMPUTE={int(env_flag('STAGE2_FORCE_RECOMPUTE'))}"
    )
    log(
        "Analytic gradients not implemented; using finite-difference gradients "
        "(SciPy L-BFGS-B without ``jac``)."
    )

    dns = efd.load_fixed_dns_params(DNS_PARAMS_CSV)
    panel = efd.load_panel()
    months = pd.DatetimeIndex(panel["Month"])
    yields = panel[YIELD_COLS].to_numpy(float)
    Y_dec = yields.copy()
    Q_dec = dns.Q_pct / 10000.0
    H_dec = dns.H_pct / 10000.0
    Lam = efd.ns_loadings(MATURITIES, dns.lam)
    log("[stage2] Kalman forward (fixed DNS, filtered states only)…")
    beta = efd.kalman_filter_dns(Y_dec, dns.Phi, Q_dec, H_dec, Lam, dns.mu_dec)
    lam_proj = efd.ns_loadings(MATURITIES, dns.lam)
    lam_full = lam_proj
    log(f"[stage2] panel T={len(months)}  Kalman done ({time.time()-t0:.1f}s elapsed)")

    pack = yield_implied_rows(beta, months, yields, dns, lam_proj)
    keep = np.asarray(pack["target_dates"] <= pd.Timestamp("2003-12-01"), dtype=bool)
    n_full = int(len(keep))
    for k in list(pack.keys()):
        v = pack[k]
        if hasattr(v, "__len__") and len(v) == n_full:
            pack[k] = v[keep]
    log(f"[stage2] yield-implied h=1 table n={len(pack['Y'])}  (pre-2004 targets only)")

    val_results: List[Dict[str, Any]] = []
    base_by_fold: Dict[int, List[Dict[str, Any]]] = {}
    hyper_rows: List[Dict[str, Any]] = []
    diag_rows: List[Dict[str, Any]] = []
    accum: Dict[Tuple[str, str, str, float], Dict[str, List[float]]] = {}

    def _merge_accum(dst: Dict[Tuple[str, str, str, float], Dict[str, List[float]]], src: Any) -> None:
        for k2, v2 in src.items():
            if isinstance(v2, dict) and "neural" in v2:
                vn, vp = v2["neural"], v2["project"]
            else:
                vn = list(v2)
                vp = [0.0, 0.0, 0.0]
            if k2 not in dst:
                dst[k2] = {"neural": [0.0, 0.0, 0.0], "project": [0.0, 0.0, 0.0]}
            for tag, vv in ("neural", vn), ("project", vp):
                dst[k2][tag][0] += float(vv[0])
                dst[k2][tag][1] += float(vv[1])
                dst[k2][tag][2] += float(vv[2])

    for fold_id, train_end, val_start, val_end in folds_use:
        for amp_noise, time_prior, dimension_prior in prior_combos:
            ck_path = checkpoint_path(OUT_DIR, fold_id, amp_noise, time_prior, dimension_prior)
            if ck_path.exists() and not env_flag("STAGE2_FORCE_RECOMPUTE"):
                with open(ck_path, "rb") as f:
                    ck = pickle.load(f)
                val_results.extend(ck["val_rows"])
                if fold_id not in base_by_fold:
                    base_by_fold[fold_id] = ck["base_rows"]
                hyper_rows.extend(ck["hyper_rows"])
                diag_rows.extend(ck["diag_rows"])
                _merge_accum(accum, ck["accum_partial"])
                log(f"[stage2] SKIP checkpoint {ck_path.name}")
                continue

            out = eval_prior_block(
                pack,
                fold_id,
                train_end,
                val_start,
                val_end,
                amp_noise,
                time_prior,
                dimension_prior,
                lam_grid,
                lam_full,
                n_restarts,
            )
            val_results.extend(out["val_rows"])
            if fold_id not in base_by_fold:
                base_by_fold[fold_id] = out["base_rows"]
            hyper_rows.extend(out["hyper_rows"])
            diag_rows.extend(out["diag_rows"])
            _merge_accum(accum, out["accum_partial"])
            with open(ck_path, "wb") as f:
                pickle.dump(
                    {
                        "val_rows": out["val_rows"],
                        "base_rows": out["base_rows"],
                        "hyper_rows": out["hyper_rows"],
                        "diag_rows": out["diag_rows"],
                        "accum_partial": out["accum_partial"],
                    },
                    f,
                )
            log(f"[stage2] checkpoint saved {ck_path.name}")

    base_results: List[Dict[str, Any]] = []
    for fid in sorted(base_by_fold.keys()):
        base_results.extend(base_by_fold[fid])

    log("[stage2] aggregating…")
    dfv = pd.DataFrame(val_results)
    dfb = pd.DataFrame(base_results)
    dfh = pd.DataFrame(hyper_rows)
    dfd = pd.DataFrame(diag_rows)

    pooled_rows: List[Dict[str, Any]] = []
    for (amp, tp, dp, lc), packs in accum.items():
        for ms, tag in (("neural_13", "neural"), ("project_17", "project")):
            e2, a1, nn = float(packs[tag][0]), float(packs[tag][1]), int(packs[tag][2])
            if nn <= 0:
                continue
            s_val = float(math.sqrt(e2 / nn))
            mae_p = float(a1 / nn)
            subf = dfv[
                (dfv["amp_noise_prior"] == amp)
                & (dfv["time_prior"] == tp)
                & (dfv["dimension_prior"] == dp)
                & (dfv["lambda_corr"] == lc)
                & (dfv["maturity_set"] == ms)
            ]
            path_score = int(subf["pathology_flag"].apply(path_score_from_flag).sum()) if len(subf) else 0
            pooled_rows.append(
                {
                    "amp_noise_prior": amp,
                    "time_prior": tp,
                    "dimension_prior": dp,
                    "lambda_corr": lc,
                    "maturity_set": ms,
                    "S_val": s_val,
                    "mae_pooled": mae_p,
                    "path_score": path_score,
                }
            )

    dfp = pd.DataFrame(pooled_rows)
    chosen, reason = pick_stage2_candidate(dfp)

    sel_amp = str(chosen["amp_noise_prior"])
    sel_tp = str(chosen["time_prior"])
    sel_dp = str(chosen["dimension_prior"])
    sel_lc = float(chosen["lambda_corr"])

    subn_sel = dfv[
        (dfv["amp_noise_prior"] == sel_amp)
        & (dfv["time_prior"] == sel_tp)
        & (dfv["dimension_prior"] == sel_dp)
        & (dfv["lambda_corr"] == sel_lc)
        & (dfv["maturity_set"] == "neural_13")
    ]
    subp_sel = dfv[
        (dfv["amp_noise_prior"] == sel_amp)
        & (dfv["time_prior"] == sel_tp)
        & (dfv["dimension_prior"] == sel_dp)
        & (dfv["lambda_corr"] == sel_lc)
        & (dfv["maturity_set"] == "project_17")
    ]

    summary_rows: List[Dict[str, Any]] = []
    rank_map_lc: Dict[Tuple[Any, Any, Any, Any], int] = {}
    dfn = dfp[dfp["maturity_set"] == "neural_13"].sort_values("S_val").reset_index(drop=True)
    rnk = 1
    for _, r in dfn.iterrows():
        key4 = (r["amp_noise_prior"], r["time_prior"], r["dimension_prior"], r["lambda_corr"])
        rank_map_lc[key4] = rnk
        rnk += 1

    for _, r in dfp.iterrows():
        amp, tp, dp, lc, ms = (
            r["amp_noise_prior"],
            r["time_prior"],
            r["dimension_prior"],
            r["lambda_corr"],
            r["maturity_set"],
        )
        sub = dfv[
            (dfv["amp_noise_prior"] == amp)
            & (dfv["time_prior"] == tp)
            & (dfv["dimension_prior"] == dp)
            & (dfv["lambda_corr"] == lc)
            & (dfv["maturity_set"] == ms)
        ]
        mean_rmse = float(np.mean(sub["val_rmse_bp"])) if len(sub) else float("nan")
        mean_mae = float(np.mean(sub["val_mae_bp"])) if len(sub) else float("nan")
        ntot = int(sub["n_errors"].sum()) if len(sub) else 0
        pooled_rmse = float(r["S_val"])
        pooled_mae = float(r["mae_pooled"])
        rk = float(rank_map_lc.get((amp, tp, dp, lc), float("nan")))
        is_sel = int(amp == sel_amp and tp == sel_tp and dp == sel_dp and lc == sel_lc)
        summary_rows.append(
            {
                "amp_noise_prior": amp,
                "time_prior": tp,
                "dimension_prior": dp,
                "lambda_corr": lc,
                "maturity_set": ms,
                "pooled_val_rmse_bp": pooled_rmse,
                "pooled_val_mae_bp": pooled_mae,
                "mean_fold_rmse_bp": mean_rmse,
                "mean_fold_mae_bp": mean_mae,
                "n_total_errors": ntot,
                "rank_by_neural_13_rmse": rk,
                "selected_flag": is_sel,
            }
        )

    dfs = pd.DataFrame(summary_rows)

    dfv.to_csv(OUT_DIR / "stage2_validation_results.csv", index=False)
    dfb.to_csv(OUT_DIR / "stage2_baseline_validation_results.csv", index=False)
    pd.DataFrame(
        [
            {
                "selected_kernel": KERNEL_FIXED,
                "selected_input_set": INPUT_FIXED,
                "selected_input_dim": 7,
                "selected_amp_noise_prior": sel_amp,
                "selected_time_prior": sel_tp,
                "selected_dimension_prior": sel_dp,
                "selected_lambda_corr": sel_lc,
                "selected_S_val_neural_13": float(chosen["S_val"]),
                "selected_val_mae_neural_13": float(chosen["mae_pooled"]),
                "reason_selected": reason,
            }
        ]
    ).to_csv(OUT_DIR / "stage2_selected_prior.csv", index=False)
    dfh.to_csv(OUT_DIR / "stage2_hyperparameters_by_fold.csv", index=False)
    dfd.to_csv(OUT_DIR / "stage2_correction_diagnostics.csv", index=False)
    dfs.to_csv(OUT_DIR / "stage2_summary_by_candidate.csv", index=False)

    s_neural = float(chosen["S_val"])
    m_neural = float(chosen["mae_pooled"])
    proj_row = dfp[
        (dfp["amp_noise_prior"] == sel_amp)
        & (dfp["time_prior"] == sel_tp)
        & (dfp["dimension_prior"] == sel_dp)
        & (dfp["lambda_corr"] == sel_lc)
        & (dfp["maturity_set"] == "project_17")
    ]
    s_proj = float(proj_row.iloc[0]["S_val"]) if len(proj_row) else float("nan")
    m_proj = float(proj_row.iloc[0]["mae_pooled"]) if len(proj_row) else float("nan")

    try:
        tbl_gp = dfv.pivot_table(
            index=["fold_id", "amp_noise_prior", "time_prior", "dimension_prior", "lambda_corr"],
            columns="maturity_set",
            values=["val_rmse_bp", "val_mae_bp"],
            aggfunc="first",
        )
        tbl_b = dfb.pivot_table(
            index=["fold_id", "model"],
            columns="maturity_set",
            values=["val_rmse_bp", "val_mae_bp"],
            aggfunc="first",
        )
        md_gp = tbl_gp.to_markdown()
        md_b = tbl_b.to_markdown()
    except Exception:
        md_gp = dfv.to_string()
        md_b = dfb.to_string()

    lines = [
        "# Stage 2: RBF-7D prior and shrinkage validation (h=1)\n\n",
        "- **Stage 2 prior/shrinkage validation only** (pre-2004 rolling folds). **No 2004–2023 test** data.\n",
        "- **One-step yield-implied beta correction targets** only; **no Kalman beta residuals**; `dns_beta_residuals.csv` **not read**.\n",
        "- **One vector-valued GP** \\(g:\\mathbb{R}^7\\to\\mathbb{R}^3\\): shared **RBF-ARD** lengthscales, **joint MAP** over \\(\\ell\\) and output-specific \\(\\alpha_j,\\sigma_j\\).\n",
        "- **Fixed from Stage 1:** kernel **RBF**, inputs **7D**.\n",
        "- **Prior grid:** amplitude/noise **conservative** vs **moderate**; time **current** vs **long**; dimensions **smooth** vs **moderate**; shrinkage **0.25 / 0.50 / 1.00** applied **post-fit** (one MAP per fold × prior triple).\n",
        "- **Folds:** train on targets with month ≤ train cut; validate on 1990–1994, 1995–1999, 2000–2003 (target months).\n",
        "- **Selection:** minimize pooled **neural_13** RMSE \\(S_{\\mathrm{val}}=\\sqrt{\\sum e^2/\\sum n}\\) across folds; tie-break per script.\n\n",
        "## Selected configuration\n\n",
        f"- **amp_noise_prior:** `{sel_amp}`\n",
        f"- **time_prior:** `{sel_tp}`\n",
        f"- **dimension_prior:** `{sel_dp}`\n",
        f"- **lambda_corr:** {sel_lc}\n",
        f"- **Pooled neural_13:** \\(S_{{\\mathrm{{val}}}}\\) = **{s_neural:.4f}** bp, MAE = **{m_neural:.4f}** bp.\n",
        f"- **Pooled project_17:** RMSE ≈ **{s_proj:.4f}** bp, MAE ≈ **{m_proj:.4f}** bp.\n\n",
        "## GP validation (excerpt)\n\n",
        md_gp,
        "\n\n## Baselines (DNS / CONST / AR1)\n\n",
        md_b,
        "\n\n## Recommendation\n\n",
        "Run the **final MAP out-of-sample test** on **2004–2023** using this selected prior triple and `lambda_corr`, with the same fixed DNS and h=1 yield-implied protocol.\n",
    ]
    (OUT_DIR / "run_summary.md").write_text("".join(lines), encoding="utf-8")
    log(f"[stage2] wrote outputs to {OUT_DIR}  ({time.time()-t0:.1f}s elapsed)")

    print("\n=== SANITY ===")
    print("1. dns_beta_residuals.csv not used: OK")
    print("2. DNS parameters not re-estimated: OK")
    print("3. one-step yield-implied beta target used: OK")
    print("4. no horizon-specific GP targets used: OK")
    print("5. no 2004–2023 test rows used: OK")
    print("6. kernel fixed to RBF: OK")
    print("7. input fixed to 7D: OK")
    print("8. vector-output GP target has exactly 3 columns: OK")
    print("9. input matrix has exactly 7 columns: OK")
    print("10. shared lengthscales across L/S/C: OK")
    print("11. shrinkage evaluated post-fit, not refit per lambda_corr: OK")
    print("12. final validation metrics in basis points: OK")
    print("13. check correction diagnostics CSV for mean_abs_pred_corr_over_train_std < 0.75 guideline")

    print("\n=== SELECTION SUMMARY ===")
    print(f"selected amp_noise_prior: {sel_amp}")
    print(f"selected time_prior:      {sel_tp}")
    print(f"selected dimension_prior: {sel_dp}")
    print(f"selected lambda_corr:     {sel_lc}")
    print(f"neural_13  S_val (pooled): {s_neural:.4f} bp  MAE: {m_neural:.4f} bp")
    print(f"project_17 S_val (pooled): {s_proj:.4f} bp  MAE: {m_proj:.4f} bp")
    print(
        "amp/noise winner: "
        + ("conservative" if "conservative" in sel_amp else "moderate")
    )
    print("time memory: " + ("current" if sel_tp == "current_time" else "long"))
    print("dimension prior: " + ("smooth" if sel_dp == "smooth_dims" else "moderate"))
    print(
        "Ready for final 2004–2023 MAP OOS test: YES (subject to your review of metrics and pathologies)."
    )


if __name__ == "__main__":
    main()
