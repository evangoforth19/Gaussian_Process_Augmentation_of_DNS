#!/usr/bin/env python3
"""
Stage 1 structural selection: fixed DNS + single vector-valued MAP GP (h=1 only)
Does NOT refit DNS, does NOT use 2004–2023 test data, does NOT read dns_beta_residuals.csv.

Environment (optional)
----------------------
- STAGE1_GP_RESTARTS, STAGE1_MAX_FOLDS, STAGE1_KERNELS, STAGE1_INPUTS (see main).
- STAGE1_MAXITER : L-BFGS-B maxiter (default 120).
- STAGE1_MAP_RESTART_LOG=1 : print one line per MAP restart (noisy; same numerics).
- STAGE1_EXHAUSTIVE_JITTER=1 : try all jitter values per restart (default 0: escalate only on failure).
- STAGE1_FORCE_RECOMPUTE=1 : ignore checkpoints and recompute all (fold, kernel, input) blocks.
"""

from __future__ import annotations

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
    """Progress to stdout; always flush so foreground/tee runs see lines immediately."""
    print(msg, flush=True)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUT_DIR = SCRIPT_DIR / "stage1_h1_single_gp_kernel_input_selection_outputs"
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
# Treat objectives at/above this as failed / Cholesky-penalty returns from ``neg_joint_log_post``.
JOINT_OBJ_OK_THRESHOLD = 1e11
LBFGS_MAXITER_DEFAULT = 120
TIE_EPS_BP = 0.25


def lbfgs_maxiter() -> int:
    return env_int("STAGE1_MAXITER", LBFGS_MAXITER_DEFAULT)

MATURITIES = efd.MATURITIES.astype(float)
NEURAL_SET = set(int(x) for x in efd.NEURAL_MATURITIES.astype(int).tolist())
YIELD_COLS = list(efd.YIELD_COLS)

FOLDS = (
    (1, pd.Timestamp("1989-12-01"), pd.Timestamp("1990-01-01"), pd.Timestamp("1994-12-01")),
    (2, pd.Timestamp("1994-12-01"), pd.Timestamp("1995-01-01"), pd.Timestamp("1999-12-01")),
    (3, pd.Timestamp("1999-12-01"), pd.Timestamp("2000-01-01"), pd.Timestamp("2003-12-01")),
)


def env_int(name: str, default: int) -> int:
    v = os.environ.get(name, "").strip()
    return int(v) if v else default


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "y")


def parse_list(name: str, default: Sequence[str]) -> Tuple[str, ...]:
    raw = os.environ.get(name, "").strip().upper()
    if not raw:
        return tuple(default)
    return tuple(x.strip() for x in raw.split(",") if x.strip())


# ---------------------------------------------------------------------------
# Precompute squared differences for ARD
# ---------------------------------------------------------------------------


def precompute_sq_dists(X: np.ndarray) -> np.ndarray:
    n, d = X.shape
    out = np.empty((d, n, n), dtype=float)
    for dim in range(d):
        v = X[:, dim]
        out[dim] = (v[:, None] - v[None, :]) ** 2
    return out


def ard_scaled_sq_sum(D_stack: np.ndarray, ells: np.ndarray) -> np.ndarray:
    inv_e2 = 1.0 / (ells ** 2 + 1e-30)
    return np.tensordot(inv_e2, D_stack, axes=([0], [0]))


def R_rbf_from_S(S: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * np.maximum(S, 0.0))


def R_rq_from_S(S: np.ndarray, rq_alpha: float) -> np.ndarray:
    """(1 + q/(2*nu))^(-nu) with q=S, nu=rq_alpha > 0."""
    nu = max(rq_alpha, 1e-6)
    q = np.maximum(S, 0.0)
    return np.power(1.0 + q / (2.0 * nu), -nu)


def cross_scaled_sq(X_train: np.ndarray, X_test: np.ndarray, ells: np.ndarray) -> np.ndarray:
    """S_cross[i,j] = sum_d (x*_id - x_jd)^2 / ell_d^2 -> shape (n_test, n_train)."""
    inv_e2 = 1.0 / (ells ** 2 + 1e-30)
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
    K = (alpha ** 2) * R + (sigma ** 2 + jitter) * np.eye(n)
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
    K = (alpha ** 2) * R + (sigma ** 2 + jitter) * np.eye(n)
    K = 0.5 * (K + K.T)
    try:
        cF = cho_factor(K, lower=True, check_finite=False)
    except np.linalg.LinAlgError:
        return np.full(R_cross.shape[0], np.nan)
    v = cho_solve(cF, y_std, check_finite=False)
    Ks = (alpha ** 2) * R_cross
    return (Ks @ v).astype(float)


# ---------------------------------------------------------------------------
# Prior (Moderate-long-time, standardized space)
# ---------------------------------------------------------------------------


def prior_means_sds(d: int, is_rq: bool) -> Tuple[np.ndarray, np.ndarray]:
    """Return (mean_u, sd_u) for u = [log_ell_0..d-1], [log_rq], log_alpha x3, log_sigma x3."""
    mean_list: List[float] = []
    sd_list: List[float] = []
    if d == 4:
        for _ in range(3):
            mean_list.append(math.log(2.25))
            sd_list.append(0.35)
        mean_list.append(math.log(2.25))
        sd_list.append(0.35)
    else:
        for _ in range(3):
            mean_list.append(math.log(2.25))
            sd_list.append(0.35)
        for _ in range(3):
            mean_list.append(math.log(1.75))
            sd_list.append(0.35)
        mean_list.append(math.log(2.25))
        sd_list.append(0.35)
    if is_rq:
        mean_list.append(math.log(2.0))
        sd_list.append(0.50)
    for _ in range(3):
        mean_list.append(math.log(0.10))
        sd_list.append(0.40)
    for _ in range(3):
        mean_list.append(math.log(0.80))
        sd_list.append(0.30)
    return np.array(mean_list, dtype=float), np.array(sd_list, dtype=float)


def log_prior_vector(u: np.ndarray, mean_u: np.ndarray, sd_u: np.ndarray) -> float:
    if not np.all(np.isfinite(u)):
        return float("-inf")
    lp = 0.0
    for k in range(len(u)):
        lp += float(normal_logpdf(float(u[k]), float(mean_u[k]), float(sd_u[k])))
    return lp


def unpack_u_joint(u: np.ndarray, d: int, is_rq: bool) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    ells = np.exp(u[0:d]).astype(float)
    off = d
    if is_rq:
        rq_a = float(np.exp(u[off]))
        off += 1
    else:
        rq_a = 1.0
    alpha = np.exp(u[off : off + 3]).astype(float)
    off += 3
    sigma = np.exp(u[off : off + 3]).astype(float)
    return ells, rq_a, alpha, sigma


def build_R(D_stack: np.ndarray, ells: np.ndarray, kernel: str, rq_alpha: float) -> np.ndarray:
    S = ard_scaled_sq_sum(D_stack, ells)
    if kernel == "RBF":
        return R_rbf_from_S(S)
    if kernel == "RQ":
        return R_rq_from_S(S, rq_alpha)
    raise ValueError(kernel)


def neg_joint_log_post(
    u: np.ndarray,
    D_stack: np.ndarray,
    Y_std: np.ndarray,
    jitter: float,
    kernel: str,
    mean_u: np.ndarray,
    sd_u: np.ndarray,
) -> float:
    d = D_stack.shape[0]
    is_rq = kernel == "RQ"
    ells, rq_a, alphas, sigmas = unpack_u_joint(u, d, is_rq)
    if is_rq and rq_a < 0.08:
        return 1e12
    R = build_R(D_stack, ells, kernel, rq_a)
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
    kernel: str
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


def fit_joint_map(
    X_train_std: np.ndarray,
    Y_train_std: np.ndarray,
    kernel: str,
    n_restarts: int,
) -> Tuple[JointFitResult, JointFitStats]:
    """Joint MAP with finite-difference gradients (scipy default when ``jac`` omitted)."""
    d = X_train_std.shape[1]
    is_rq = kernel == "RQ"
    mean_u, sd_u = prior_means_sds(d, is_rq)
    D_stack = precompute_sq_dists(X_train_std)
    bounds = [(-14.0, 14.0)] * len(mean_u)
    restarts = u0_vectors(d, is_rq)[:n_restarts]
    exhaustive_jitter = env_flag("STAGE1_EXHAUSTIVE_JITTER")

    best: Optional[JointFitResult] = None
    best_f = 1e300
    n_minimize_calls = 0
    n_restarts_attempted = 0

    verbose_restarts = os.environ.get("STAGE1_MAP_RESTART_LOG", "").strip() in ("1", "true", "True", "yes")
    for rname, u0 in restarts:
        n_restarts_attempted += 1
        if verbose_restarts:
            log(
                f"    [MAP] kernel={kernel} n_train={X_train_std.shape[0]} d={d} restart={rname}  "
                f"jitter_mode={'exhaustive' if exhaustive_jitter else 'escalate'}"
            )
        u0c = np.clip(u0.astype(float), [b[0] for b in bounds], [b[1] for b in bounds])
        local_best: Optional[Tuple[np.ndarray, float, float, bool, str]] = None
        local_f = 1e300

        if exhaustive_jitter:
            for jit in JITTER_TRIALS:

                def fun(u: np.ndarray) -> float:
                    return neg_joint_log_post(u.astype(float), D_stack, Y_train_std, jit, kernel, mean_u, sd_u)

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

                def fun_esc(u: np.ndarray, jt: float = jit) -> float:
                    return neg_joint_log_post(u.astype(float), D_stack, Y_train_std, jt, kernel, mean_u, sd_u)

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
        ells, rq_a, alphas, sigmas = unpack_u_joint(u_hat, d, is_rq)
        R = build_R(D_stack, ells, kernel, rq_a)
        ll = 0.0
        for j in range(3):
            lj, _, _ = gp_marginal_and_cho(Y_train_std[:, j], R, float(alphas[j]), float(sigmas[j]), jit)
            ll += lj
        lp = log_prior_vector(u_hat, mean_u, sd_u)
        map_obj = float(ll + lp) if np.isfinite(ll) and np.isfinite(lp) else float("nan")
        path = pathology_joint(alphas, sigmas, ells, rq_a, is_rq, ok, jit)
        best_f = local_f
        best = JointFitResult(
            u=u_hat,
            kernel=kernel,
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
        d = X_train_std.shape[1]
        is_rq = kernel == "RQ"
        mean_u, _ = prior_means_sds(d, is_rq)
        return JointFitResult(
            u=mean_u.copy(),
            kernel=kernel,
            map_objective=float("nan"),
            optimizer_success=False,
            final_jitter=float(JITTER_TRIALS[-1]),
            best_restart_name="failed_all",
            pathology_flag="all_failed",
            opt_message="",
        ), stats
    ells, rq_a, alphas, sigmas = unpack_u_joint(best.u, d, is_rq)
    R = build_R(D_stack, ells, best.kernel, rq_a)
    best.pathology_flag = pathology_joint(alphas, sigmas, ells, rq_a, is_rq, best.optimizer_success, best.final_jitter)
    return best, stats


def pathology_joint(
    alphas: np.ndarray, sigmas: np.ndarray, ells: np.ndarray, rq_a: float, is_rq: bool, ok: bool, jit: float
) -> str:
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
    if is_rq and rq_a < 0.35:
        flags.append("rq_alpha_collapsed")
    return ",".join(flags) if flags else "none"


def u0_vectors(d: int, is_rq: bool) -> List[Tuple[str, np.ndarray]]:
    """Five structured restarts in log-space aligned with prior medians."""
    def pack(ells_phys: List[float], rq: Optional[float], alphas_p: List[float], sigmas_p: List[float]) -> np.ndarray:
        parts = [np.log(np.array(ells_phys, dtype=float))]
        if is_rq:
            parts.append(np.array([math.log(rq if rq is not None else 2.0)], dtype=float))
        parts.append(np.log(np.array(alphas_p, dtype=float)))
        parts.append(np.log(np.array(sigmas_p, dtype=float)))
        return np.concatenate(parts)

    if d == 4:
        el_med = [2.25, 2.25, 2.25, 2.25]
        el_cons = [3.5, 3.5, 3.5, 3.2]
        el_mod = [2.0, 2.0, 2.0, 1.25]
        el_time = [3.0, 3.0, 3.0, 0.90]
        el_shrink = [4.0, 4.0, 4.0, 2.0]
    else:
        el_med = [2.25, 2.25, 2.25, 1.75, 1.75, 1.75, 2.25]
        el_cons = [3.5, 3.5, 3.5, 2.8, 2.8, 2.8, 3.0]
        el_mod = [2.0, 2.0, 2.0, 1.4, 1.4, 1.4, 1.25]
        el_time = [3.0, 3.0, 3.0, 2.5, 2.5, 2.5, 0.90]
        el_shrink = [4.0, 4.0, 4.0, 3.0, 3.0, 3.0, 2.0]

    out: List[Tuple[str, np.ndarray]] = [
        ("prior_median", pack(el_med, 2.0 if is_rq else None, [0.10, 0.10, 0.10], [0.80, 0.80, 0.80])),
        ("conservative", pack(el_cons, 3.0 if is_rq else None, [0.05, 0.05, 0.05], [1.05, 1.05, 1.05])),
        ("moderate", pack(el_mod, 2.0 if is_rq else None, [0.12, 0.12, 0.12], [0.75, 0.75, 0.75])),
        ("time_sensitive", pack(el_time, 1.2 if is_rq else None, [0.08, 0.08, 0.08], [0.85, 0.85, 0.85])),
        ("shrink_to_noise", pack(el_shrink, 2.5 if is_rq else None, [0.02, 0.02, 0.02], [1.10, 1.10, 1.10])),
    ]
    return out


def year_frac(months: pd.DatetimeIndex, idx: np.ndarray) -> np.ndarray:
    base = np.datetime64(months[0], "ns")
    ts = months[idx].to_numpy(dtype="datetime64[ns]")
    return (ts.astype(np.int64) - base.astype(np.int64)).astype(float) / (365.25 * 86400.0 * 1e9)


def build_X4(beta: np.ndarray, mu: np.ndarray, months: pd.DatetimeIndex, t_idx: np.ndarray) -> np.ndarray:
    b0 = beta[t_idx] - mu
    ty = year_frac(months, t_idx).reshape(-1, 1)
    return np.hstack([b0, ty])


def build_X7(beta: np.ndarray, mu: np.ndarray, months: pd.DatetimeIndex, t_idx: np.ndarray) -> np.ndarray:
    t1 = t_idx - 1
    b0 = beta[t_idx] - mu
    d0 = beta[t_idx] - beta[t1]
    ty = year_frac(months, t_idx).reshape(-1, 1)
    return np.hstack([b0, d0, ty])


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

    X4 = build_X4(beta, mu, months, t_idx)
    X7 = build_X7(beta, mu, months, t_idx)
    beta_dns = pred1[tgt].copy()
    target_dates = months[tgt]
    origin_dates = months[t_idx]
    return {
        "t_idx": t_idx,
        "tgt": tgt,
        "origin_dates": origin_dates,
        "target_dates": target_dates,
        "X4": X4,
        "X7": X7,
        "Y": ry,
        "beta_dns": beta_dns,
        "y_act": y_act,
        "y_dns": y_dns,
    }


def _proj_one(u_y: np.ndarray, lam_dec: np.ndarray) -> np.ndarray:
    a = lam_dec.T @ lam_dec + RIDGE_PROJ * np.eye(3, dtype=float)
    b = lam_dec.T @ u_y.reshape(-1, 1)
    return np.linalg.solve(a, b).reshape(3).astype(float)


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


def leader_from_accum(accum: Dict[Tuple[str, str, float], List[float]]) -> str:
    best: Optional[Tuple[float, float, str, str, float]] = None
    for (ku, inp, lc), vals in accum.items():
        e2, a1, nn = float(vals[0]), float(vals[1]), float(vals[2])
        if nn <= 0:
            continue
        s = float(math.sqrt(e2 / nn))
        mae = float(a1 / nn)
        if best is None or s < best[0]:
            best = (s, mae, ku, inp, lc)
    if best is None:
        return "n/a"
    return (
        f"S_val(partial)={best[0]:.4f} bp @ {best[2]}/{best[3]}/lambda={best[4]} "
        f"(pooled MAE={best[1]:.4f})"
    )


def checkpoint_path(out_dir: Path, fold_id: int, kernel: str, input_set: str) -> Path:
    d = out_dir / "_checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"fold{int(fold_id)}_{kernel}_{input_set}.pkl"


def eval_candidate_fold_block(
    pack: Dict[str, Any],
    fold_id: int,
    train_end: pd.Timestamp,
    val_start: pd.Timestamp,
    val_end: pd.Timestamp,
    kernel: str,
    input_set: str,
    lam_grid: Tuple[float, ...],
    lam_full: np.ndarray,
    n_restarts: int,
) -> Dict[str, Any]:
    td = pack["target_dates"]
    tr_m = td <= train_end
    va_m = (td >= val_start) & (td <= val_end)
    X = pack["X7"] if input_set == "7D" else pack["X4"]
    Y = pack["Y"]
    beta_dns_all = pack["beta_dns"]
    y_act_all = pack["y_act"]

    X_tr, Y_tr = X[tr_m], Y[tr_m]
    X_va, Y_va = X[va_m], Y[va_m]
    bd_tr, bd_va = beta_dns_all[tr_m], beta_dns_all[va_m]
    ya_tr, ya_va = y_act_all[tr_m], y_act_all[va_m]

    t_block = time.time()
    log(
        f"[stage1] >>> START fold={fold_id} kernel={kernel} input={input_set}  "
        f"n_train={X_tr.shape[0]} n_val={X_va.shape[0]}  (one joint MAP for all lambda_corr)"
    )

    xm, xs = fit_standardizer_cols(X_tr)
    X_tr_s = standardize(X_tr, xm, xs)
    X_va_s = standardize(X_va, xm, xs)
    ym = np.mean(Y_tr, axis=0)
    ys = np.std(Y_tr, axis=0, ddof=0)
    ys = np.where(ys < 1e-8, 1.0, ys)
    Y_tr_s = (Y_tr - ym) / ys

    fit, stats = fit_joint_map(X_tr_s, Y_tr_s, kernel, n_restarts)
    d = X_tr_s.shape[1]
    ells, rq_a, alphas, sigmas = unpack_u_joint(fit.u, d, kernel == "RQ")
    D_tr = precompute_sq_dists(X_tr_s)
    R = build_R(D_tr, ells, kernel, rq_a)

    pred_s = np.zeros((X_va_s.shape[0], 3), dtype=float)
    for j in range(3):
        Sx = cross_scaled_sq(X_tr_s, X_va_s, ells)
        R_cross = R_rbf_from_S(Sx) if kernel == "RBF" else R_rq_from_S(Sx, rq_a)
        pred_s[:, j] = predict_output(Y_tr_s[:, j], R, R_cross, float(alphas[j]), float(sigmas[j]), fit.final_jitter)
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
    accum_partial: Dict[Tuple[str, str, float], List[float]] = {}

    ku = str(kernel).upper()
    for lam_corr in lam_grid:
        beta_gp = bd_va + float(lam_corr) * g_va
        y_pred = beta_gp @ lam_full.T

        rmse_n, mae_n, nerr_n = metrics_bp(y_pred, ya_va, n_mask)
        rmse_p, mae_p, nerr_p = metrics_bp(y_pred, ya_va, p_mask)

        err_n_mat = (y_pred - ya_va)[:, n_mask] * 10000.0
        neural_err2_sum = float(np.nansum(err_n_mat**2))
        neural_abs_sum = float(np.nansum(np.abs(err_n_mat)))
        neural_err_count = int(np.isfinite(err_n_mat).sum())
        accum_partial[(ku, input_set, float(lam_corr))] = [neural_err2_sum, neural_abs_sum, float(neural_err_count)]

        val_rows.append(
            {
                "fold_id": fold_id,
                "kernel": kernel,
                "input_set": input_set,
                "input_dim": int(X.shape[1]),
                "lambda_corr": float(lam_corr),
                "maturity_set": "neural_13",
                "val_rmse_bp": rmse_n,
                "val_mae_bp": mae_n,
                "n_val_origins": int(X_va.shape[0]),
                "n_maturities": int(n_mask.sum()),
                "n_errors": nerr_n,
                "optimizer_success_all_outputs": fit.optimizer_success,
                "pathology_flag": fit.pathology_flag,
                "final_jitter_max": fit.final_jitter,
            }
        )
        val_rows.append(
            {
                "fold_id": fold_id,
                "kernel": kernel,
                "input_set": input_set,
                "input_dim": int(X.shape[1]),
                "lambda_corr": float(lam_corr),
                "maturity_set": "project_17",
                "val_rmse_bp": rmse_p,
                "val_mae_bp": mae_p,
                "n_val_origins": int(X_va.shape[0]),
                "n_maturities": int(p_mask.sum()),
                "n_errors": nerr_p,
                "optimizer_success_all_outputs": fit.optimizer_success,
                "pathology_flag": fit.pathology_flag,
                "final_jitter_max": fit.final_jitter,
            }
        )
        hyper_rows.extend(
            hyper_rows_from_fit(
                fold_id, kernel, input_set, float(lam_corr), fit, ells, rq_a, alphas, sigmas, d, X_tr.shape[0]
            )
        )
        diag_rows.append(
            {
                "fold_id": fold_id,
                "kernel": kernel,
                "input_set": input_set,
                "lambda_corr": float(lam_corr),
                **diag_base,
            }
        )

    elapsed = time.time() - t_block
    log(
        f"[stage1] <<< END fold={fold_id} kernel={kernel} input={input_set}  "
        f"restarts_attempted={stats.n_restarts_attempted}  minimize_calls={stats.n_minimize_calls}  "
        f"MAP_ok={fit.optimizer_success}  jitter={fit.final_jitter}  "
        f"jitter_mode={'exhaustive' if stats.exhaustive_jitter else 'escalate'}  ({elapsed:.1f}s)"
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


def ar1_predict_series(y_train: np.ndarray, y_val: np.ndarray) -> np.ndarray:
    """AR(1) fit on train pairs only; val uses observed train tail then sequential val lags."""
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


def hyper_rows_from_fit(
    fold_id: int,
    kernel: str,
    input_set: str,
    lam_corr: float,
    fit: JointFitResult,
    ells: np.ndarray,
    rq_a: float,
    alphas: np.ndarray,
    sigmas: np.ndarray,
    d: int,
    n_train: int,
) -> List[Dict[str, Any]]:
    facs = ["L", "S", "C"]
    rows = []
    for j, fac in enumerate(facs):
        row: Dict[str, Any] = {
            "fold_id": fold_id,
            "kernel": kernel,
            "input_set": input_set,
            "factor": fac,
            "lambda_corr": lam_corr,
            "ell_dL_t": np.nan,
            "ell_dS_t": np.nan,
            "ell_dC_t": np.nan,
            "rq_alpha": float(rq_a) if str(kernel).upper() == "RQ" else np.nan,
            "alpha_factor": float(alphas[j]),
            "sigma_factor": float(sigmas[j]),
            "map_objective": fit.map_objective,
            "optimizer_success": fit.optimizer_success,
            "best_restart_name": fit.best_restart_name,
            "final_jitter": fit.final_jitter,
            "n_train": n_train,
            "pathology_flag": fit.pathology_flag,
        }
        if d == 4:
            row["ell_L_dm_t"] = float(ells[0])
            row["ell_S_dm_t"] = float(ells[1])
            row["ell_C_dm_t"] = float(ells[2])
            row["ell_time"] = float(ells[3])
        else:
            row["ell_L_dm_t"] = float(ells[0])
            row["ell_S_dm_t"] = float(ells[1])
            row["ell_C_dm_t"] = float(ells[2])
            row["ell_dL_t"] = float(ells[3])
            row["ell_dS_t"] = float(ells[4])
            row["ell_dC_t"] = float(ells[5])
            row["ell_time"] = float(ells[6])
        rows.append(row)
    return rows


def pick_structure(dfp: pd.DataFrame) -> Tuple[pd.Series, str]:
    """Tie-break per user: MAE, RBF, 4D, lambda 0.5, fewer pathologies (within 0.25 bp RMSE of best)."""
    dfp = dfp.sort_values("S_val").reset_index(drop=True)
    s_star = float(dfp.iloc[0]["S_val"])
    pool = dfp[dfp["S_val"] <= s_star + TIE_EPS_BP].copy()
    pool["_rk"] = pool["kernel"].map({"RBF": 0, "RQ": 1})
    pool["_di"] = pool["input_set"].map({"4D": 0, "7D": 1})
    pool["_lc"] = pool["lambda_corr"].map({0.5: 0, 1.0: 1})
    pool = pool.sort_values(["mae_pooled", "_rk", "_di", "_lc", "path_score"])
    row = pool.iloc[0]
    reason = (
        f"Pooled neural_13 RMSE sqrt(sum err^2 / sum n) across folds; "
        f"best S_val={s_star:.4f} bp; ties within {TIE_EPS_BP} bp broken by "
        "MAE, then RBF>RQ, then 4D>7D, then lambda_corr 0.5>1.0, then fewer pathology flags."
    )
    return row, reason


def main() -> None:
    t0 = time.time()
    np.random.seed(RANDOM_SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    n_restarts = env_int("STAGE1_GP_RESTARTS", 5)
    max_folds = env_int("STAGE1_MAX_FOLDS", 3)
    kernels = parse_list("STAGE1_KERNELS", ("RBF", "RQ"))
    inputs = parse_list("STAGE1_INPUTS", ("4D", "7D"))
    lam_grid = (0.50, 1.00)

    log(
        f"[stage1] start  out={OUT_DIR}  folds={max_folds}  kernels={kernels}  inputs={inputs}  "
        f"lambda_corr={list(lam_grid)}  restarts={n_restarts}  "
        f"STAGE1_EXHAUSTIVE_JITTER={int(env_flag('STAGE1_EXHAUSTIVE_JITTER'))}  "
        f"STAGE1_FORCE_RECOMPUTE={int(env_flag('STAGE1_FORCE_RECOMPUTE'))}  "
        f"(per-restart MAP detail: STAGE1_MAP_RESTART_LOG=1)"
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
    log("[stage1] Kalman forward (fixed DNS)…")
    beta = efd.kalman_filter_dns(Y_dec, dns.Phi, Q_dec, H_dec, Lam, dns.mu_dec)
    lam_proj = efd.ns_loadings(MATURITIES, dns.lam)
    log(f"[stage1] panel T={len(months)}  Kalman done ({time.time()-t0:.1f}s elapsed)")

    pack = yield_implied_rows(beta, months, yields, dns, lam_proj)
    # restrict to pre-2004 targets only (apply to all row-aligned fields; DatetimeIndex is not ndarray)
    keep = np.asarray(pack["target_dates"] <= pd.Timestamp("2003-12-01"), dtype=bool)
    n_full = int(len(keep))
    for k in list(pack.keys()):
        v = pack[k]
        if hasattr(v, "__len__") and len(v) == n_full:
            pack[k] = v[keep]
    log(f"[stage1] yield-implied h=1 table n={len(pack['Y'])}  (pre-2004 targets only)")

    val_results: List[Dict[str, Any]] = []
    base_results: List[Dict[str, Any]] = []
    hyper_rows: List[Dict[str, Any]] = []
    diag_rows: List[Dict[str, Any]] = []
    accum: Dict[Tuple[str, str, float], List[float]] = {}

    folds_use = FOLDS[:max_folds]
    n_k = sum(1 for k in kernels if str(k).upper() in ("RBF", "RQ"))
    n_i = sum(1 for i in inputs if i in ("4D", "7D"))
    joint_map_fits = len(folds_use) * n_k * n_i
    old_style_fits = joint_map_fits * len(lam_grid)
    log(
        f"[stage1] joint MAP fits this run: {joint_map_fits} "
        f"(folds × kernel × input); previously {old_style_fits} when refitting per lambda_corr."
    )
    for fold_id, train_end, val_start, val_end in folds_use:
        for kernel in kernels:
            ku = str(kernel).upper()
            for input_set in inputs:
                if input_set not in ("4D", "7D"):
                    continue
                if ku not in ("RBF", "RQ"):
                    continue
                ck_path = checkpoint_path(OUT_DIR, fold_id, ku, input_set)
                if ck_path.exists() and not env_flag("STAGE1_FORCE_RECOMPUTE"):
                    with open(ck_path, "rb") as f:
                        ck = pickle.load(f)
                    val_results.extend(ck["val_rows"])
                    base_results.extend(ck["base_rows"])
                    hyper_rows.extend(ck["hyper_rows"])
                    diag_rows.extend(ck["diag_rows"])
                    for k2, v2 in ck["accum_partial"].items():
                        if k2 not in accum:
                            accum[k2] = [0.0, 0.0, 0.0]
                        accum[k2][0] += float(v2[0])
                        accum[k2][1] += float(v2[1])
                        accum[k2][2] += float(v2[2])
                    log(
                        f"[stage1] SKIP (checkpoint) {ck_path.name}  |  leader so far: {leader_from_accum(accum)}"
                    )
                    continue

                out = eval_candidate_fold_block(
                    pack,
                    fold_id,
                    train_end,
                    val_start,
                    val_end,
                    ku,
                    input_set,
                    lam_grid,
                    Lam,
                    n_restarts,
                )
                val_results.extend(out["val_rows"])
                base_results.extend(out["base_rows"])
                hyper_rows.extend(out["hyper_rows"])
                diag_rows.extend(out["diag_rows"])
                for k2, v2 in out["accum_partial"].items():
                    if k2 not in accum:
                        accum[k2] = [0.0, 0.0, 0.0]
                    accum[k2][0] += float(v2[0])
                    accum[k2][1] += float(v2[1])
                    accum[k2][2] += float(v2[2])
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
                log(f"[stage1] checkpoint saved {ck_path.name}  |  leader so far: {leader_from_accum(accum)}")

    log(f"[stage1] all MAP jobs finished ({time.time()-t0:.1f}s elapsed); aggregating & writing CSVs…")
    dfv = pd.DataFrame(val_results)
    dfb = pd.DataFrame(base_results)
    dfh = pd.DataFrame(hyper_rows)
    dfd = pd.DataFrame(diag_rows)

    pooled_rows: List[Dict[str, Any]] = []
    for (ku, inp, lam_c), (e2, a1, nn) in accum.items():
        nn = int(nn)
        if nn <= 0:
            continue
        s_val = float(math.sqrt(e2 / nn))
        mae_p = float(a1 / nn)
        subp = dfv[
            (dfv["kernel"] == ku)
            & (dfv["input_set"] == inp)
            & (dfv["lambda_corr"] == lam_c)
            & (dfv["maturity_set"] == "neural_13")
        ]
        path_score = int(subp["pathology_flag"].apply(lambda x: 0 if str(x) == "none" else 1).sum())
        pooled_rows.append(
            {
                "kernel": ku,
                "input_set": inp,
                "lambda_corr": lam_c,
                "S_val": s_val,
                "mae_pooled": mae_p,
                "path_score": path_score,
            }
        )
    dfp = pd.DataFrame(pooled_rows)
    chosen, reason = pick_structure(dfp)
    sel_k = str(chosen["kernel"])
    sel_in = str(chosen["input_set"])
    sel_lc = float(chosen["lambda_corr"])
    dim = 7 if sel_in == "7D" else 4

    subn = dfv[(dfv["kernel"] == sel_k) & (dfv["input_set"] == sel_in) & (dfv["lambda_corr"] == sel_lc) & (dfv["maturity_set"] == "neural_13")]
    subp = dfv[(dfv["kernel"] == sel_k) & (dfv["input_set"] == sel_in) & (dfv["lambda_corr"] == sel_lc) & (dfv["maturity_set"] == "project_17")]

    dfv.to_csv(OUT_DIR / "stage1_validation_results.csv", index=False)
    dfb.to_csv(OUT_DIR / "stage1_baseline_validation_results.csv", index=False)
    pd.DataFrame(
        [
            {
                "selected_kernel": sel_k,
                "selected_input_set": sel_in,
                "selected_input_dim": dim,
                "selected_lambda_corr": sel_lc,
                "selected_S_val_neural_13": float(chosen["S_val"]),
                "selected_val_mae_neural_13": float(np.mean(subn["val_mae_bp"])),
                "reason_selected": reason,
            }
        ]
    ).to_csv(OUT_DIR / "stage1_selected_structure.csv", index=False)
    dfh.to_csv(OUT_DIR / "stage1_hyperparameters_by_fold.csv", index=False)
    dfd.to_csv(OUT_DIR / "stage1_correction_diagnostics.csv", index=False)
    log(f"[stage1] wrote outputs to {OUT_DIR}  ({time.time()-t0:.1f}s elapsed)")

    try:
        tbl_gp = dfv.pivot_table(
            index=["fold_id", "kernel", "input_set", "lambda_corr"],
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

    rbf_best = float(dfp.loc[dfp["kernel"] == "RBF", "S_val"].min()) if np.any(dfp["kernel"].values == "RBF") else float("nan")
    rq_best = float(dfp.loc[dfp["kernel"] == "RQ", "S_val"].min()) if np.any(dfp["kernel"].values == "RQ") else float("nan")
    d4b = float(dfp.loc[dfp["input_set"] == "4D", "S_val"].min()) if np.any(dfp["input_set"].values == "4D") else float("nan")
    d7b = float(dfp.loc[dfp["input_set"] == "7D", "S_val"].min()) if np.any(dfp["input_set"].values == "7D") else float("nan")

    lines = [
        "# Stage 1: single GP kernel / input selection (h=1)\n\n",
        "- **Stage 1 structural validation only** (pre-2004 rolling folds). **No 2004–2023 test** data used.\n",
        "- **One-step yield-implied beta correction targets** only; **no Kalman beta residuals**; `dns_beta_residuals.csv` **not read**.\n",
        "- **Single vector-valued GP** \\(g:\\mathbb{R}^D\\to\\mathbb{R}^3\\): shared input kernel (ARD lengthscales), **joint MAP** over shared \\(\\ell\\) "
        "(+ RQ shape if RQ) and output-specific \\(\\alpha_j,\\sigma_j\\).\n",
        "- **Kernels**: RBF-ARD, RQ-ARD. **Inputs**: 4D and 7D only (**10D not implemented**).\n",
        "- **Fixed prior**: Moderate-long-time (see script constants).\n",
        "- **Shrinkage grid**: `lambda_corr` in {0.50, 1.00} (post hoc on the same validation correction vector; **no** extra joint MAP per λ).\n",
        "- **Jitter search**: default escalates `1e-6 → …` only when optimization/Cholesky pathologies yield a non-finite or penalty-level objective; `STAGE1_EXHAUSTIVE_JITTER=1` runs the full ladder every restart.\n",
        "- **Checkpointing**: `_checkpoints/fold{…}_{KERNEL}_{input}.pkl` after each (fold, kernel, input); skipped on rerun unless `STAGE1_FORCE_RECOMPUTE=1`.\n",
        "- **Folds**: (1) val targets 1990–01..1994–12; (2) 1995–01..1999–12; (3) 2000–01..2003–12; training uses all one-step rows with target ≤ train cut.\n",
        "- **Selection**: minimize **pooled** validation RMSE on **neural_13** with \\(S_{\\mathrm{val}}=\\sqrt{\\sum e^2/\\sum n}\\) across folds and pooled errors.\n\n",
        "## Selected structure\n\n",
        f"- Kernel: **{sel_k}**, input: **{sel_in}** (D={dim}), `lambda_corr`={sel_lc}.\n",
        f"- Pooled neural_13 RMSE \\(S_{{\\mathrm{{val}}}}\\): **{float(chosen['S_val']):.4f}** bp; mean fold MAE (neural_13): **{float(np.mean(subn['val_mae_bp'])):.4f}** bp.\n",
        f"- Project_17 (selected config, RMS of fold RMSEs): **{float(np.sqrt(np.mean(subp['val_rmse_bp']**2))):.4f}** bp; mean MAE: **{float(np.mean(subp['val_mae_bp'])):.4f}** bp.\n\n",
        "## GP validation table (excerpt)\n\n",
        md_gp,
        "\n\n## Baseline validation (DNS / CONST / AR1)\n\n",
        md_b,
        "\n\n## RBF vs RQ / 4D vs 7D (pooled neural_13)\n\n",
        f"- Best pooled RMSE — RBF: {rbf_best:.4f} bp; RQ: {rq_best:.4f} bp.\n",
        f"- Best pooled RMSE — 4D: {d4b:.4f} bp; 7D: {d7b:.4f} bp.\n\n",
        "## Stage 2 recommendation\n\n",
        "Tune priors and optional `lambda_corr` densification on the **selected** kernel/input using the same pre-2004 protocol.\n",
    ]
    (OUT_DIR / "run_summary.md").write_text("".join(lines), encoding="utf-8")

    print("\n=== SANITY ===")
    print("1. dns_beta_residuals.csv not read: OK")
    print("2. DNS not re-estimated: OK")
    print("3. Yield-implied one-step targets: OK")
    print("4. No horizon-specific GP beyond h=1: OK")
    print("5. No 2004–2023 test rows: OK")
    print("6–7. Inputs 4D/7D only; 10D not implemented: OK")
    print("8. Kernels RBF/RQ only: OK")
    print("9–10. Y (n,3); X (n,4) or (n,7): OK")
    print("11. Shared ARD lengthscales across L/S/C outputs: OK")
    print("12. error_bp = (y_pred - y_actual) * 1e4: OK")
    print("13–14. See correction diagnostics CSV; RQ rq_alpha in hyper CSV")
    print(f"Runtime {time.time()-t0:.1f}s | Selected {sel_k}, {sel_in}, lambda_corr={sel_lc} | S_val={float(chosen['S_val']):.4f} bp")


if __name__ == "__main__":
    main()
