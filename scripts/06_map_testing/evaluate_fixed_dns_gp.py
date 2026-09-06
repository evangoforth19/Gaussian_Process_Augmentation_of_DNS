#!/usr/bin/env python3
"""
evaluate_fixed_dns_gp.py

Fixed-DNS Gaussian-Process correction evaluation, comparable in metric
format to the DLNS / DLNS-X neural-network DNS paper.

Design (important)
------------------
- DNS is NOT re-estimated.  We use the published pre-2004 DNS parameters
  from
      Kalman Filter/Original Macro + DNS Filter/dns_fitted_params_labeled.csv
  for the entire experiment.  Units are aligned by dividing mu by 100:
      mu_decimal = mu_percent / 100        (per project convention)
      Phi, lambda  : unchanged
      yields, betas: decimal units internally
      errors       : reported in basis points = (y_pred_dec - y_actual_dec)*1e4

- The Kalman filter is run *forward* with these fixed parameters over the
  entire 1972-01..2025-12 yield panel to extract *filtered* states.  Only
  filtered factors are used (never smoothed).

- For each horizon h in {1, 3, 6, 12}:
      DNS h-step factor forecast:   beta_dns_{t+h|t} = mu + Phi^h(beta_t - mu)
      Residual target            :  R_{t,h} = beta_{t+h} - beta_dns_{t+h|t}

  Training set : origins t with t+h <= 2003-12  (no leak)
  Test origins : 2004-01..2023-12 with t+h <= 2025-12

- Models compared
    1. Plain DNS                              (DNS)
    2. DNS + constant residual correction     (DNS+CONST)
    3. DNS + AR(1) residual correction        (DNS+AR1)
    4. DNS + MAP GP correction                (DNS+GP_MAP)
    5. DNS + fully Bayesian GP correction     (DNS+GP_BAYES)
       Implemented via adaptive random-walk Metropolis on the standardised
       residuals (same kernel/prior as MAP).  The fully Bayesian GP is
       run only if FULLBAYES_ENABLE=1 (default on).  Each horizon gets its
       own MCMC chain; lightweight defaults are used to keep total runtime
       moderate (set FULLBAYES_N_DRAWS / FULLBAYES_BURN_IN env vars to
       override).

- DOES NOT read Gaussian Processes/dns_beta_residuals.csv at any point.

Outputs (Gaussian Processes/fixed_dns_gp_clean_outputs/):
    pooled_yield_metrics_bp.csv
    maturity_yield_metrics_bp.csv
    pooled_beta_metrics.csv
    run_summary.md
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve, solve_discrete_lyapunov

from plain_dns_gp_correction import JointPlainDNSGP, matern52_ard


# ============================================================
# Paths and constants
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PANEL_CSV = PROJECT_ROOT / "Macro" / "master_macro_dns_panel.csv"
DNS_PARAMS_CSV = (
    PROJECT_ROOT
    / "Kalman Filter"
    / "Original Macro + DNS Filter"
    / "dns_fitted_params_labeled.csv"
)
OUT_DIR = SCRIPT_DIR / "fixed_dns_gp_clean_outputs"

YIELD_COLS: List[str] = [
    "y3", "y6", "y9", "y12", "y15", "y18", "y21", "y24",
    "y30", "y36", "y48", "y60", "y72", "y84", "y96", "y108", "y120",
]
MATURITIES = np.array(
    [3, 6, 9, 12, 15, 18, 21, 24, 30, 36, 48, 60, 72, 84, 96, 108, 120],
    dtype=float,
)
NEURAL_MATURITIES = np.array(
    [3, 6, 9, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120], dtype=float,
)

HORIZONS: Tuple[int, ...] = (1, 3, 6, 12)
TRAIN_END = pd.Timestamp("2003-12-01")
FORECAST_ORIGIN_START = pd.Timestamp("2004-01-01")
FORECAST_ORIGIN_END = pd.Timestamp("2023-12-01")

RANDOM_SEED = 7300

# MAP GP hyper-prior settings (same as plain_dns_gp_correction defaults)
MAP_TAU_SCALE = 0.5
MAP_LS_LOG_MEAN = math.log(1.25)
MAP_LS_LOG_SD = 0.35
MAP_NOISE_SCALE = 0.25
GP_MAXITER = 400
GP_N_RESTARTS = 3
GP_JITTER = 1e-6

# Fully Bayesian GP settings
FULLBAYES_ENABLE = int(os.environ.get("FULLBAYES_ENABLE", "1"))
FB_N_DRAWS = int(os.environ.get("FULLBAYES_N_DRAWS", "4000"))
FB_BURN_IN = int(os.environ.get("FULLBAYES_BURN_IN", "1500"))
FB_THIN = int(os.environ.get("FULLBAYES_THIN", "2"))
FB_PROP_ADAPT_INTERVAL = 100
FB_PRINT_EVERY = 500
FB_TARGET_ACC_LOW = 0.20
FB_TARGET_ACC_HIGH = 0.35


# ============================================================
# DNS parameter loading (fixed pre-2004)
# ============================================================

@dataclass
class FixedDNS:
    lam: float
    mu_dec: np.ndarray           # (3,)
    Phi: np.ndarray              # (3, 3)
    Q_pct: np.ndarray            # (3, 3) in percent-scale state innovation covariance
    H_pct: np.ndarray            # (17, 17) measurement noise covariance in percent
    mu_pct: np.ndarray           # original mu in percent (for filter run)
    source_csv: Path


def load_fixed_dns_params(path: Path) -> FixedDNS:
    df = pd.read_csv(path)
    pmap = dict(zip(df["parameter"], df["value"]))
    lam = float(np.exp(pmap["log_lambda"]))
    mu_pct = np.array([pmap["mu_L"], pmap["mu_S"], pmap["mu_C"]])
    Phi = np.array([
        [pmap["Phi_L_L"], pmap["Phi_L_S"], pmap["Phi_L_C"]],
        [pmap["Phi_S_L"], pmap["Phi_S_S"], pmap["Phi_S_C"]],
        [pmap["Phi_C_L"], pmap["Phi_C_S"], pmap["Phi_C_C"]],
    ])
    q = np.array([
        pmap["Q_root_00"], pmap["Q_root_01"], pmap["Q_root_02"],
        pmap["Q_root_11"], pmap["Q_root_12"], pmap["Q_root_22"],
    ])
    U = np.array([
        [q[0], q[1], q[2]],
        [0.0, q[3], q[4]],
        [0.0, 0.0, q[5]],
    ])
    Q = U @ U.T
    h_std = np.array([pmap[f"H_std_{c}"] for c in YIELD_COLS])
    H = np.diag(h_std ** 2)
    return FixedDNS(
        lam=lam,
        mu_dec=mu_pct / 100.0,
        Phi=Phi,
        Q_pct=Q,
        H_pct=H,
        mu_pct=mu_pct,
        source_csv=path,
    )


# ============================================================
# Nelson-Siegel loadings + Kalman filter (filtering only, fixed params)
# ============================================================

def ns_loadings(maturities: np.ndarray, lam: float) -> np.ndarray:
    x = lam * maturities
    c1 = np.ones_like(maturities, dtype=float)
    c2 = (1.0 - np.exp(-x)) / x
    c3 = c2 - np.exp(-x)
    return np.column_stack([c1, c2, c3])


def kalman_filter_dns(
    Y: np.ndarray,
    Phi: np.ndarray,
    Q: np.ndarray,
    H: np.ndarray,
    Lambda: np.ndarray,
    mu: np.ndarray,
) -> np.ndarray:
    """Plain Kalman filter for the DNS state-space model in level form.

    Returns filtered states a_tt of shape (T, K).
    """
    T, N = Y.shape
    K = Phi.shape[0]

    z = np.zeros(K)
    try:
        P = solve_discrete_lyapunov(Phi, Q)
    except Exception:
        P = 100.0 * np.eye(K)

    a_tt = np.empty((T, K), dtype=float)
    for t in range(T):
        z_pred = Phi @ z
        P_pred = Phi @ P @ Phi.T + Q

        y_centered = Y[t] - Lambda @ mu - Lambda @ z_pred
        S = Lambda @ P_pred @ Lambda.T + H
        S = 0.5 * (S + S.T)

        PLT = P_pred @ Lambda.T
        K_gain = np.linalg.solve(S, PLT.T).T

        z = z_pred + K_gain @ y_centered
        P = P_pred - K_gain @ Lambda @ P_pred
        a_tt[t] = z + mu
    return a_tt


# ============================================================
# Direct-h residuals
# ============================================================

def matrix_power_int(M: np.ndarray, h: int) -> np.ndarray:
    out = np.eye(M.shape[0])
    for _ in range(h):
        out = out @ M
    return out


@dataclass
class HorizonData:
    h: int
    train_X: np.ndarray            # (n_train, 3) beta_t (decimal) for training origins
    train_Y: np.ndarray            # (n_train, 3) residual targets
    train_origin_dates: pd.DatetimeIndex
    train_target_dates: pd.DatetimeIndex
    test_X: np.ndarray             # (n_test, 3) beta_t0 (decimal) for OOS origins
    test_Y: np.ndarray             # (n_test, 3) realised residuals at target month
    test_origin_dates: pd.DatetimeIndex
    test_target_dates: pd.DatetimeIndex
    test_beta_dns: np.ndarray      # (n_test, 3) DNS h-step beta forecast (decimal)
    test_beta_actual: np.ndarray   # (n_test, 3) filtered beta at target month (decimal)
    test_y_actual: np.ndarray      # (n_test, 17) observed yields (decimal) at target
    Phi_h: np.ndarray


def build_horizon_data(
    beta_filt_dec: np.ndarray,
    months: pd.DatetimeIndex,
    yields_dec: np.ndarray,
    dns: FixedDNS,
    h: int,
) -> HorizonData:
    Phi_h = matrix_power_int(dns.Phi, h)
    T = len(months)
    mu = dns.mu_dec

    pred = np.full_like(beta_filt_dec, np.nan)
    actual = np.full_like(beta_filt_dec, np.nan)
    for t in range(T - h):
        beta_t = beta_filt_dec[t]
        pred[t + h] = mu + Phi_h @ (beta_t - mu)
        actual[t + h] = beta_filt_dec[t + h]
    resid = actual - pred

    origin_idx = np.arange(0, T - h)
    target_idx = origin_idx + h
    origin_dates = months[origin_idx]
    target_dates = months[target_idx]

    train_mask = target_dates <= TRAIN_END
    test_mask = (
        (origin_dates >= FORECAST_ORIGIN_START)
        & (origin_dates <= FORECAST_ORIGIN_END)
    )

    return HorizonData(
        h=h,
        train_X=beta_filt_dec[origin_idx][train_mask].copy(),
        train_Y=resid[target_idx][train_mask].copy(),
        train_origin_dates=origin_dates[train_mask],
        train_target_dates=target_dates[train_mask],
        test_X=beta_filt_dec[origin_idx][test_mask].copy(),
        test_Y=resid[target_idx][test_mask].copy(),
        test_origin_dates=origin_dates[test_mask],
        test_target_dates=target_dates[test_mask],
        test_beta_dns=pred[target_idx][test_mask].copy(),
        test_beta_actual=beta_filt_dec[target_idx][test_mask].copy(),
        test_y_actual=yields_dec[target_idx][test_mask].copy(),
        Phi_h=Phi_h,
    )


# ============================================================
# AR(1) baseline (per factor)
# ============================================================

@dataclass
class AR1Coef:
    a: float
    c: float
    mu_r: float


def fit_ar1(series: np.ndarray) -> AR1Coef:
    r = np.asarray(series, dtype=float)
    if len(r) < 3:
        return AR1Coef(a=0.0, c=0.0, mu_r=float(np.mean(r)) if len(r) else 0.0)
    y = r[1:]
    x = r[:-1]
    X = np.column_stack([x, np.ones_like(x)])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    a = float(coef[0])
    c = float(coef[1])
    mu_r = c / (1.0 - a) if abs(1.0 - a) > 1e-9 else float(np.mean(r))
    return AR1Coef(a=a, c=c, mu_r=mu_r)


def ar1_iterate(coef: AR1Coef, last_r: float, h_gap: int) -> float:
    """Iterate AR(1) h_gap steps from last_r."""
    if h_gap <= 0:
        return float(last_r)
    return float(coef.a ** h_gap * (last_r - coef.mu_r) + coef.mu_r)


# ============================================================
# Fully Bayesian GP (Adaptive RWM)
# ============================================================

# Standardiser (local copy to avoid coupling to import order)
@dataclass
class Standardizer:
    mean_: np.ndarray
    std_: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean_) / self.std_


def fit_standardizer(x: np.ndarray) -> Standardizer:
    m = np.mean(x, axis=0)
    s = np.std(x, axis=0, ddof=0)
    s = np.where(s < 1e-8, 1.0, s)
    return Standardizer(mean_=m, std_=s)


def _log_halfnormal(logx: np.ndarray, scale: float) -> np.ndarray:
    x = np.exp(logx)
    return (
        math.log(math.sqrt(2.0 / math.pi))
        - math.log(scale)
        - 0.5 * (x / scale) ** 2
        + logx
    )


def _log_normal(z: np.ndarray, mean: float, sd: float) -> np.ndarray:
    return -0.5 * math.log(2.0 * math.pi) - math.log(sd) - 0.5 * ((z - mean) / sd) ** 2


def _fb_unpack(theta: np.ndarray) -> Dict[str, np.ndarray]:
    idx = 0
    log_tau = float(theta[idx]); idx += 1
    log_lam = theta[idx:idx + 3]; idx += 3
    log_ls = theta[idx:idx + 9].reshape(3, 3); idx += 9
    log_sig = theta[idx:idx + 3]
    tau = float(np.exp(log_tau))
    lam = np.exp(log_lam)
    alpha = tau * lam
    ls = np.exp(log_ls)
    sig = np.exp(log_sig)
    return {
        "log_tau": log_tau,
        "log_lam": log_lam,
        "log_ls": log_ls,
        "log_sig": log_sig,
        "tau": tau,
        "alpha": alpha,
        "ls": ls,
        "sig": sig,
    }


def _fb_log_post(
    theta: np.ndarray,
    X_std: np.ndarray,
    Y_std: np.ndarray,
    jitter: float,
) -> float:
    if not np.all(np.isfinite(theta)):
        return -np.inf
    p = _fb_unpack(theta)
    if np.any(p["sig"] < 1e-8) or np.any(p["alpha"] < 1e-12):
        return -np.inf

    lp = 0.0
    lp += float(_log_halfnormal(np.array([p["log_tau"]]), MAP_TAU_SCALE)[0])
    lp += float(np.sum(_log_halfnormal(p["log_lam"], 1.0)))
    lp += float(np.sum(_log_halfnormal(p["log_sig"], MAP_NOISE_SCALE)))
    lp += float(np.sum(_log_normal(p["log_ls"].reshape(-1), MAP_LS_LOG_MEAN, MAP_LS_LOG_SD)))
    if not np.isfinite(lp):
        return -np.inf

    n = X_std.shape[0]
    for j in range(3):
        K = (p["alpha"][j] ** 2) * matern52_ard(X_std, X_std, p["ls"][j])
        K = K + ((p["sig"][j] ** 2) + jitter) * np.eye(n)
        try:
            cF = cho_factor(K, lower=True, check_finite=False)
        except np.linalg.LinAlgError:
            return -np.inf
        a_y = cho_solve(cF, Y_std[:, j], check_finite=False)
        logdet = 2.0 * np.sum(np.log(np.diag(cF[0])))
        lp += -0.5 * float(Y_std[:, j] @ a_y) - 0.5 * logdet - 0.5 * n * math.log(2.0 * math.pi)
        if not np.isfinite(lp):
            return -np.inf
    return lp


@dataclass
class FBChain:
    samples: np.ndarray
    acc_rate: float
    prop_scale: float


def run_adaptive_rwm(
    theta0: np.ndarray,
    X_std: np.ndarray,
    Y_std: np.ndarray,
    n_draws: int,
    burn_in: int,
    thin: int,
    jitter: float,
    seed: int,
) -> FBChain:
    rng = np.random.default_rng(seed)
    z = theta0.copy()
    dim = len(z)
    lp = _fb_log_post(z, X_std, Y_std, jitter)
    if not np.isfinite(lp):
        raise RuntimeError("Initial log-posterior is not finite for FB GP chain.")
    samples: List[np.ndarray] = []
    accepts = 0
    total = 0
    prop_scale = 0.08
    for it in range(1, n_draws + 1):
        prop = z + prop_scale * rng.standard_normal(dim)
        lp_new = _fb_log_post(prop, X_std, Y_std, jitter)
        total += 1
        if np.isfinite(lp_new) and (lp_new - lp > 0 or np.log(rng.random()) < (lp_new - lp)):
            z = prop
            lp = lp_new
            accepts += 1
        if it <= burn_in and it % FB_PROP_ADAPT_INTERVAL == 0:
            acc = accepts / total
            if acc < FB_TARGET_ACC_LOW:
                prop_scale *= 0.85
            elif acc > FB_TARGET_ACC_HIGH:
                prop_scale *= 1.12
            prop_scale = float(np.clip(prop_scale, 1e-4, 1.0))
        if it > burn_in and (it - burn_in) % thin == 0:
            samples.append(z.copy())
        if it % FB_PRINT_EVERY == 0:
            print(
                f"    [FB] iter={it:5d}  lp={lp:9.2f}  acc={accepts/total:.3f}  "
                f"prop={prop_scale:.4f}",
                flush=True,
            )
    arr = np.stack(samples, axis=0) if samples else np.empty((0, dim))
    return FBChain(samples=arr, acc_rate=accepts / max(total, 1), prop_scale=prop_scale)


def fb_predict_mean(
    samples: np.ndarray,
    X_train_std: np.ndarray,
    Y_train_std: np.ndarray,
    X_test_std: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    jitter: float,
) -> np.ndarray:
    """Posterior predictive mean in *original* residual units.

    Returns array of shape (n_test, 3).
    """
    n_s = samples.shape[0]
    n_train = X_train_std.shape[0]
    n_test = X_test_std.shape[0]
    accum = np.zeros((n_test, 3))
    for s_idx in range(n_s):
        p = _fb_unpack(samples[s_idx])
        for j in range(3):
            K = (p["alpha"][j] ** 2) * matern52_ard(X_train_std, X_train_std, p["ls"][j])
            K = K + ((p["sig"][j] ** 2) + jitter) * np.eye(n_train)
            try:
                cF = cho_factor(K, lower=True, check_finite=False)
            except np.linalg.LinAlgError:
                continue
            Ks = (p["alpha"][j] ** 2) * matern52_ard(X_test_std, X_train_std, p["ls"][j])
            mu_std = Ks @ cho_solve(cF, Y_train_std[:, j], check_finite=False)
            accum[:, j] += mu_std * y_std[j] + y_mean[j]
    accum /= max(n_s, 1)
    return accum


# ============================================================
# Pipeline driver
# ============================================================

@dataclass
class HorizonModels:
    h: int
    rbar: np.ndarray
    ar1: List[AR1Coef]  # one per factor
    map_gp: Optional[JointPlainDNSGP]
    fb_samples: Optional[np.ndarray]
    fb_acc_rate: Optional[float]
    fb_X_std: Optional[np.ndarray]
    fb_Y_std: Optional[np.ndarray]
    fb_x_scaler: Optional[Standardizer]
    fb_y_scaler: Optional[Standardizer]


def train_models_for_horizon(hd: HorizonData, run_fb: bool) -> HorizonModels:
    rbar = hd.train_Y.mean(axis=0)
    ar1 = [fit_ar1(hd.train_Y[:, j]) for j in range(3)]

    # MAP GP using existing class
    np.random.seed(RANDOM_SEED + hd.h)
    map_gp = JointPlainDNSGP(
        tau_scale=MAP_TAU_SCALE,
        ls_log_mean=MAP_LS_LOG_MEAN,
        ls_log_sd=MAP_LS_LOG_SD,
        noise_scale=MAP_NOISE_SCALE,
        jitter=GP_JITTER,
        maxiter=GP_MAXITER,
        n_restarts=GP_N_RESTARTS,
        random_seed=RANDOM_SEED + hd.h,
    )
    try:
        map_gp.fit(hd.train_X, hd.train_Y)
    except Exception as exc:
        print(f"  [WARN] MAP GP fit failed for h={hd.h}: {exc}")
        map_gp = None

    fb_samples = None
    fb_acc = None
    fb_X_std = fb_Y_std = None
    fb_x_scaler = fb_y_scaler = None
    if run_fb and map_gp is not None:
        try:
            fb_x_scaler = fit_standardizer(hd.train_X)
            fb_y_scaler = fit_standardizer(hd.train_Y)
            fb_X_std = fb_x_scaler.transform(hd.train_X)
            fb_Y_std = fb_y_scaler.transform(hd.train_Y)

            p = map_gp.fitted_params_
            theta0 = np.concatenate([
                np.array([p["log_tau"]]),
                p["log_lambda"].astype(float),
                p["log_ls"].reshape(-1),
                p["log_noise"].astype(float),
            ])
            print(
                f"  [FB] Starting Adaptive RWM for h={hd.h}  "
                f"(n_train={fb_X_std.shape[0]}, draws={FB_N_DRAWS}, burn={FB_BURN_IN})"
            )
            t0 = time.time()
            chain = run_adaptive_rwm(
                theta0=theta0,
                X_std=fb_X_std,
                Y_std=fb_Y_std,
                n_draws=FB_N_DRAWS,
                burn_in=FB_BURN_IN,
                thin=FB_THIN,
                jitter=GP_JITTER,
                seed=RANDOM_SEED + 100 * hd.h,
            )
            elapsed = time.time() - t0
            fb_samples = chain.samples
            fb_acc = chain.acc_rate
            print(
                f"  [FB] h={hd.h} chain done in {elapsed:.1f}s; "
                f"kept {fb_samples.shape[0]} samples, acc={fb_acc:.3f}"
            )
        except Exception as exc:
            print(f"  [WARN] Fully Bayesian GP failed for h={hd.h}: {exc}")
            fb_samples = None

    return HorizonModels(
        h=hd.h,
        rbar=rbar,
        ar1=ar1,
        map_gp=map_gp,
        fb_samples=fb_samples,
        fb_acc_rate=fb_acc,
        fb_X_std=fb_X_std,
        fb_Y_std=fb_Y_std,
        fb_x_scaler=fb_x_scaler,
        fb_y_scaler=fb_y_scaler,
    )


def predict_corrections(
    hd: HorizonData,
    mods: HorizonModels,
) -> Dict[str, np.ndarray]:
    """For each test origin, produce the predicted residual correction
    array (n_test, 3) for each correction model."""
    n_test = hd.test_X.shape[0]

    # constant
    const_pred = np.tile(mods.rbar, (n_test, 1))

    # AR(1) per factor, iterated h steps from last training residual
    last_train_target = hd.train_Y[-1]
    ar1_pred = np.empty((n_test, 3))
    for j in range(3):
        coef = mods.ar1[j]
        for i in range(n_test):
            t0 = hd.test_origin_dates[i]
            last_origin = hd.train_origin_dates[-1]
            gap = (t0.year - last_origin.year) * 12 + (t0.month - last_origin.month)
            gap = max(gap, 0)
            ar1_pred[i, j] = ar1_iterate(coef, last_train_target[j], gap)

    # MAP GP
    if mods.map_gp is not None:
        map_pred = mods.map_gp.predict_residual_mean(hd.test_X)
    else:
        map_pred = np.zeros_like(const_pred)

    # Fully Bayesian GP posterior predictive mean
    if mods.fb_samples is not None and mods.fb_samples.shape[0] > 0:
        X_test_std = mods.fb_x_scaler.transform(hd.test_X)
        fb_pred = fb_predict_mean(
            samples=mods.fb_samples,
            X_train_std=mods.fb_X_std,
            Y_train_std=mods.fb_Y_std,
            X_test_std=X_test_std,
            y_mean=mods.fb_y_scaler.mean_,
            y_std=mods.fb_y_scaler.std_,
            jitter=GP_JITTER,
        )
    else:
        fb_pred = None

    return {
        "CONST": const_pred,
        "AR1": ar1_pred,
        "GP_MAP": map_pred,
        "GP_BAYES": fb_pred,
    }


# ============================================================
# Metric aggregation
# ============================================================

def compute_yield_errors(
    hd: HorizonData,
    dns: FixedDNS,
    corrections: Dict[str, np.ndarray],
) -> pd.DataFrame:
    """Returns per (origin, target, model, maturity) error frame."""
    Lambda_dec = ns_loadings(MATURITIES, dns.lam)
    rows: List[Dict[str, object]] = []

    beta_dns = hd.test_beta_dns
    y_actual = hd.test_y_actual

    model_betas: Dict[str, np.ndarray] = {"DNS": beta_dns}
    for name, corr in corrections.items():
        if corr is None:
            continue
        model_betas[f"DNS+{name}"] = beta_dns + corr

    for model_name, betas in model_betas.items():
        y_pred = betas @ Lambda_dec.T  # (n_test, 17) decimal
        err_bp = (y_pred - y_actual) * 1.0e4
        for i in range(hd.test_X.shape[0]):
            t0 = hd.test_origin_dates[i]
            tgt = hd.test_target_dates[i]
            for j, m in enumerate(MATURITIES.astype(int)):
                rows.append({
                    "forecast_origin": t0,
                    "target_month": tgt,
                    "horizon": hd.h,
                    "model": model_name,
                    "maturity_months": int(m),
                    "y_actual_decimal": float(y_actual[i, j]),
                    "y_pred_decimal": float(y_pred[i, j]),
                    "error_bp": float(err_bp[i, j]),
                })
    return pd.DataFrame(rows)


def compute_beta_errors(
    hd: HorizonData,
    corrections: Dict[str, np.ndarray],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    beta_dns = hd.test_beta_dns
    beta_actual = hd.test_beta_actual

    model_betas: Dict[str, np.ndarray] = {"DNS": beta_dns}
    for name, corr in corrections.items():
        if corr is None:
            continue
        model_betas[f"DNS+{name}"] = beta_dns + corr

    for model_name, betas in model_betas.items():
        for i in range(hd.test_X.shape[0]):
            t0 = hd.test_origin_dates[i]
            tgt = hd.test_target_dates[i]
            for j, fac in enumerate(["L", "S", "C"]):
                rows.append({
                    "forecast_origin": t0,
                    "target_month": tgt,
                    "horizon": hd.h,
                    "model": model_name,
                    "factor": fac,
                    "beta_actual": float(beta_actual[i, j]),
                    "beta_pred": float(betas[i, j]),
                    "beta_error": float(betas[i, j] - beta_actual[i, j]),
                })
    return pd.DataFrame(rows)


def pooled_yield_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mat_set_name, mats in [
        ("project_17", MATURITIES.astype(int).tolist()),
        ("neural_13", NEURAL_MATURITIES.astype(int).tolist()),
    ]:
        sub = df[df["maturity_months"].isin(mats)]
        for (model, h), grp in sub.groupby(["model", "horizon"]):
            errs = grp["error_bp"].to_numpy()
            rows.append({
                "model": model,
                "horizon": int(h),
                "maturity_set": mat_set_name,
                "pooled_rmse_bp": float(np.sqrt(np.mean(errs ** 2))),
                "pooled_mae_bp": float(np.mean(np.abs(errs))),
                "n_origins": int(grp["forecast_origin"].nunique()),
                "n_maturities": int(grp["maturity_months"].nunique()),
                "n_errors": int(len(errs)),
            })
    return pd.DataFrame(rows).sort_values(
        ["maturity_set", "model", "horizon"]
    ).reset_index(drop=True)


def maturity_yield_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    neural_set = set(NEURAL_MATURITIES.astype(int).tolist())
    for (model, h, m), grp in df.groupby(["model", "horizon", "maturity_months"]):
        errs = grp["error_bp"].to_numpy()
        mat_set = "neural_13" if int(m) in neural_set else "project_only"
        rows.append({
            "model": model,
            "horizon": int(h),
            "maturity_months": int(m),
            "maturity_set": mat_set,
            "rmse_bp": float(np.sqrt(np.mean(errs ** 2))),
            "mae_bp": float(np.mean(np.abs(errs))),
            "n_obs": int(len(errs)),
        })
    return pd.DataFrame(rows).sort_values(
        ["model", "horizon", "maturity_months"]
    ).reset_index(drop=True)


def pooled_beta_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, h), grp in df.groupby(["model", "horizon"]):
        errs = grp["beta_error"].to_numpy()
        rows.append({
            "model": model,
            "horizon": int(h),
            "pooled_beta_rmse": float(np.sqrt(np.mean(errs ** 2))),
            "pooled_beta_mae": float(np.mean(np.abs(errs))),
            "n_obs": int(len(errs)),
        })
    return pd.DataFrame(rows).sort_values(["model", "horizon"]).reset_index(drop=True)


# ============================================================
# Sanity checks
# ============================================================

def sanity_checks(
    dns: FixedDNS,
    beta_filt_dec: np.ndarray,
    horizon_datasets: Dict[int, HorizonData],
    yield_df: pd.DataFrame,
) -> List[Tuple[str, bool, str]]:
    checks: List[Tuple[str, bool, str]] = []
    checks.append((
        "broken_residuals_file_not_used",
        True,
        "Script does not read Gaussian Processes/dns_beta_residuals.csv.",
    ))
    checks.append((
        "dns_parameters_not_reestimated",
        True,
        f"DNS parameters loaded from fixed file: {dns.source_csv}.",
    ))

    L_mean = float(np.mean(beta_filt_dec[:, 0]))
    cond_units = 0.005 < L_mean < 0.25
    checks.append((
        "beta_factor_units_decimal",
        cond_units,
        f"Mean filtered L = {L_mean:.4f} (should be 0.02..0.12 decimal).",
    ))

    resid_lines: List[str] = []
    cond_resid_ok = True
    for h, hd in horizon_datasets.items():
        for j, fac in enumerate(["L", "S", "C"]):
            r = hd.train_Y[:, j]
            mean_ = float(np.mean(r))
            std_ = float(np.std(r) + 1e-12)
            ratio = abs(mean_) / std_
            resid_lines.append(f"  h={h} {fac}: mean={mean_:+.6f} std={std_:.6f} |mean|/std={ratio:.4f}")
            if ratio > 1.0:
                cond_resid_ok = False
    resid_msg = "Training residual stats per (horizon, factor):\n" + "\n".join(resid_lines)
    checks.append(("residual_stats_no_unit_mismatch", cond_resid_ok, resid_msg))

    dns_h1 = yield_df[(yield_df["model"] == "DNS") & (yield_df["horizon"] == 1)]
    rmse_h1 = float(np.sqrt(np.mean(dns_h1["error_bp"].to_numpy() ** 2)))
    cond_rmse = 10.0 < rmse_h1 < 800.0
    checks.append((
        "dns_h1_rmse_plausible_bp",
        cond_rmse,
        f"Plain DNS h=1 pooled RMSE = {rmse_h1:.1f} bp (10..800 plausible).",
    ))

    # train chronology: no training target after 2003-12
    chrono_ok = True
    chrono_msg = "All training targets satisfy target_month <= 2003-12."
    for h, hd in horizon_datasets.items():
        if hd.train_target_dates.max() > TRAIN_END:
            chrono_ok = False
            chrono_msg = f"h={h}: training target {hd.train_target_dates.max().date()} exceeds 2003-12."
            break
    checks.append(("train_test_chronology", chrono_ok, chrono_msg))

    max_bp = float(yield_df["error_bp"].abs().max())
    checks.append((
        "errors_in_basis_points",
        max_bp < 1e5,
        f"max |error_bp| across the run = {max_bp:.1f} bp.",
    ))

    return checks


# ============================================================
# Run summary
# ============================================================

def write_run_summary(
    out_path: Path,
    dns: FixedDNS,
    pooled_df: pd.DataFrame,
    sanity: List[Tuple[str, bool, str]],
    horizon_datasets: Dict[int, HorizonData],
    horizon_models: Dict[int, HorizonModels],
    fb_ran: bool,
) -> None:
    lines: List[str] = []
    lines.append("# Fixed-DNS GP correction summary\n\n")

    lines.append("## Configuration\n\n")
    lines.append("- **DNS estimation**: NOT re-estimated.  Fixed pre-2004 parameters loaded from "
                 f"`{dns.source_csv.relative_to(PROJECT_ROOT)}`.\n")
    lines.append("- **Unit convention**: mu, beta factors, and yields are kept in decimal units throughout.  "
                 "mu_decimal = mu_percent / 100.  Phi and lambda are unchanged.\n")
    lines.append(f"- **Fixed parameters**: lambda = {dns.lam:.6f}; "
                 f"mu_decimal = ({dns.mu_dec[0]:+.5f}, {dns.mu_dec[1]:+.5f}, {dns.mu_dec[2]:+.5f}); "
                 f"Phi diagonal = ({dns.Phi[0,0]:.4f}, {dns.Phi[1,1]:.4f}, {dns.Phi[2,2]:.4f}).\n")
    lines.append("- **Factor extraction**: Kalman *filtered* states only.  The filter is run forward once over "
                 "the full 1972-01..2025-12 panel with the fixed parameters above.  Smoothed states are never used.\n")
    lines.append("- **Direct horizon-h forecasts**: beta_pred_{t+h|t} = mu + Phi^h (beta_t - mu).  "
                 "Residual targets R_{t,h} = beta_{t+h} - beta_pred_{t+h|t}.\n")
    lines.append("- **Training**: residual examples with target_month <= 2003-12 only.  "
                 "Test forecast origins: 2004-01..2023-12 (subject to availability of target_month in the panel).\n")
    lines.append(f"- **Horizons evaluated**: {sorted(horizon_datasets.keys())} months.\n")
    lines.append("- **Maturity sets**: project_17 = {3,6,9,12,15,18,21,24,30,36,48,60,72,84,96,108,120}; "
                 "neural_13 = {3,6,9,12,24,36,48,60,72,84,96,108,120}.\n")
    lines.append("- **Errors**: reported in basis points = (y_pred_dec - y_actual_dec) * 1e4.\n")
    if fb_ran:
        lines.append(f"- **Fully Bayesian GP**: enabled, adaptive random-walk Metropolis on "
                     f"({FB_N_DRAWS} draws, {FB_BURN_IN} burn-in, thin {FB_THIN}).  Posterior predictive mean "
                     f"used as the point correction.  Acceptance rates per horizon:\n")
        for h, mods in horizon_models.items():
            if mods.fb_acc_rate is not None:
                lines.append(f"  - h={h}: acceptance = {mods.fb_acc_rate:.3f}, kept samples = {mods.fb_samples.shape[0]}\n")
            else:
                lines.append(f"  - h={h}: chain failed; FB result omitted.\n")
    else:
        lines.append("- **Fully Bayesian GP**: SKIPPED (set FULLBAYES_ENABLE=1 to enable).  "
                     "MAP GP results are still reported.\n")

    lines.append("\n## Training-sample residual statistics (decimal units)\n\n")
    lines.append("| horizon | factor | mean | std | |mean|/std | n_train |\n")
    lines.append("|---|---|---|---|---|---|\n")
    for h in sorted(horizon_datasets.keys()):
        hd = horizon_datasets[h]
        for j, fac in enumerate(["L", "S", "C"]):
            r = hd.train_Y[:, j]
            m = float(np.mean(r))
            s = float(np.std(r) + 1e-12)
            lines.append(f"| {h} | {fac} | {m:+.6f} | {s:.6f} | {abs(m)/s:.4f} | {len(r)} |\n")

    def _df_to_md_table(df: pd.DataFrame, index_label: str) -> str:
        cols = list(df.columns)
        head = "| " + index_label + " | " + " | ".join(str(c) for c in cols) + " |"
        sep = "|" + "|".join(["---"] * (len(cols) + 1)) + "|"
        body = []
        for idx, row in df.iterrows():
            cells = [
                f"{v:.2f}" if isinstance(v, (int, float)) and pd.notna(v) else str(v)
                for v in row.tolist()
            ]
            body.append("| " + str(idx) + " | " + " | ".join(cells) + " |")
        return "\n".join([head, sep] + body) + "\n"

    lines.append("\n## Pooled yield RMSE / MAE (basis points)\n\n")
    for mat_set in pooled_df["maturity_set"].unique():
        sub = pooled_df[pooled_df["maturity_set"] == mat_set].copy()
        wide_rmse = sub.pivot(index="model", columns="horizon", values="pooled_rmse_bp")
        wide_mae = sub.pivot(index="model", columns="horizon", values="pooled_mae_bp")
        lines.append(f"### Maturity set: `{mat_set}`\n\n")
        lines.append("Pooled RMSE (bp):\n\n")
        lines.append(_df_to_md_table(wide_rmse.round(2), "model") + "\n")
        lines.append("Pooled MAE (bp):\n\n")
        lines.append(_df_to_md_table(wide_mae.round(2), "model") + "\n")

    sub_neural = pooled_df[pooled_df["maturity_set"] == "neural_13"]

    def _val(model: str, h: int, col: str) -> float:
        r = sub_neural[(sub_neural["model"] == model) & (sub_neural["horizon"] == h)]
        return float(r[col].iloc[0]) if len(r) else float("nan")

    lines.append("## Does MAP GP beat baselines on pooled `neural_13` RMSE?\n\n")
    lines.append("| horizon | MAP GP < DNS | MAP GP < CONST | MAP GP < AR(1) |\n")
    lines.append("|---|---|---|---|\n")
    for h in sorted(horizon_datasets.keys()):
        rmse_dns = _val("DNS", h, "pooled_rmse_bp")
        rmse_const = _val("DNS+CONST", h, "pooled_rmse_bp")
        rmse_ar1 = _val("DNS+AR1", h, "pooled_rmse_bp")
        rmse_gp = _val("DNS+GP_MAP", h, "pooled_rmse_bp")
        beat_dns = rmse_gp < rmse_dns
        beat_const = rmse_gp < rmse_const
        beat_ar1 = rmse_gp < rmse_ar1
        lines.append(
            f"| {h} | {'YES' if beat_dns else 'no'} | "
            f"{'YES' if beat_const else 'no'} | "
            f"{'YES' if beat_ar1 else 'no'} |\n"
        )

    if fb_ran:
        lines.append("\n## Does fully Bayesian GP beat MAP GP on pooled `neural_13` RMSE?\n\n")
        lines.append("| horizon | FB GP RMSE bp | MAP GP RMSE bp | FB GP < MAP GP? |\n")
        lines.append("|---|---|---|---|\n")
        for h in sorted(horizon_datasets.keys()):
            rmse_fb = _val("DNS+GP_BAYES", h, "pooled_rmse_bp")
            rmse_map = _val("DNS+GP_MAP", h, "pooled_rmse_bp")
            lines.append(f"| {h} | {rmse_fb:.2f} | {rmse_map:.2f} | {'YES' if rmse_fb < rmse_map else 'no'} |\n")

    lines.append("\n## Sanity checks\n\n")
    for name, ok, msg in sanity:
        lines.append(f"- **{name}**: {'PASS' if ok else 'FAIL'}.\n{msg}\n\n")

    lines.append("## Notes\n\n")
    lines.append("- DNS parameters are fixed pre-2004; they are not refit recursively, so the OOS yield-curve "
                 "RMSEs incorporate parameter drift as well as residual structure.  This is the design the user "
                 "requested.\n")
    lines.append("- Errors are pooled across forecast origins and across maturities in the same way as in the "
                 "DLNS / DLNS-X neural-network DNS paper.  Maturity-specific tables are provided in "
                 "`maturity_yield_metrics_bp.csv`.\n")

    out_path.write_text("".join(lines), encoding="utf-8")


# ============================================================
# Driver
# ============================================================

def load_panel() -> pd.DataFrame:
    df = pd.read_csv(PANEL_CSV)
    s = df["Month"].astype(str).str.strip()
    monthly_mask = s.str.fullmatch(r"\d{4}-\d{2}")
    s = s.where(~monthly_mask, s + "-01")
    df["Month"] = pd.to_datetime(s)
    df = df.sort_values("Month").drop_duplicates(subset=["Month"]).reset_index(drop=True)
    missing = [c for c in YIELD_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Panel missing yield columns: {missing}")
    return df


def main() -> None:
    np.random.seed(RANDOM_SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading panel: {PANEL_CSV}")
    panel = load_panel()
    print(
        f"Panel shape: {panel.shape}; date range "
        f"{panel['Month'].min().date()} -> {panel['Month'].max().date()}"
    )

    print(f"\nLoading fixed DNS parameters: {DNS_PARAMS_CSV}")
    dns = load_fixed_dns_params(DNS_PARAMS_CSV)
    print(
        f"  lambda     = {dns.lam:.6f}\n"
        f"  mu_pct     = {dns.mu_pct}\n"
        f"  mu_decimal = {dns.mu_dec}\n"
        f"  Phi diag   = {np.diag(dns.Phi)}"
    )

    print("\nRunning Kalman filter forward with fixed params (percent scale)...")
    Y_dec = panel[YIELD_COLS].to_numpy(float)
    Y_pct = Y_dec * 100.0
    Lambda_pct = ns_loadings(MATURITIES, dns.lam)
    t0 = time.time()
    a_tt_pct = kalman_filter_dns(
        Y_pct, dns.Phi, dns.Q_pct, dns.H_pct, Lambda_pct, dns.mu_pct,
    )
    beta_filt_dec = a_tt_pct / 100.0
    print(f"  Filter done in {time.time() - t0:.2f}s. beta_filt_dec shape: {beta_filt_dec.shape}.")
    print(f"  Mean filtered factors (decimal): L={beta_filt_dec[:,0].mean():.4f}, "
          f"S={beta_filt_dec[:,1].mean():.4f}, C={beta_filt_dec[:,2].mean():.4f}")

    months = pd.DatetimeIndex(panel["Month"])
    horizon_datasets: Dict[int, HorizonData] = {}
    print("\nBuilding direct-h residual datasets...")
    for h in HORIZONS:
        hd = build_horizon_data(beta_filt_dec, months, Y_dec, dns, h)
        horizon_datasets[h] = hd
        print(
            f"  h={h:2d}: n_train={hd.train_X.shape[0]:4d} (target<={hd.train_target_dates.max().date()}), "
            f"n_test={hd.test_X.shape[0]:3d} (origins {hd.test_origin_dates.min().date()}..{hd.test_origin_dates.max().date()})"
        )

    print("\nTraining correction models per horizon (constant, AR(1), MAP GP, optionally FB GP)...")
    horizon_models: Dict[int, HorizonModels] = {}
    fb_attempted = bool(FULLBAYES_ENABLE)
    fb_ran_any = False
    for h in HORIZONS:
        print(f"\n  --- horizon h={h} ---")
        t0 = time.time()
        mods = train_models_for_horizon(horizon_datasets[h], run_fb=bool(FULLBAYES_ENABLE))
        elapsed = time.time() - t0
        if mods.map_gp is not None:
            print(f"  h={h}: MAP GP fit OK ({elapsed:.1f}s total).")
        if mods.fb_samples is not None and mods.fb_samples.shape[0] > 0:
            fb_ran_any = True
        horizon_models[h] = mods

    print("\nGenerating forecasts and computing errors...")
    all_yield_rows: List[pd.DataFrame] = []
    all_beta_rows: List[pd.DataFrame] = []
    for h in HORIZONS:
        hd = horizon_datasets[h]
        mods = horizon_models[h]
        corrections = predict_corrections(hd, mods)
        all_yield_rows.append(compute_yield_errors(hd, dns, corrections))
        all_beta_rows.append(compute_beta_errors(hd, corrections))
    yield_df = pd.concat(all_yield_rows, ignore_index=True)
    beta_df = pd.concat(all_beta_rows, ignore_index=True)

    pooled = pooled_yield_metrics(yield_df)
    maturity = maturity_yield_metrics(yield_df)
    pooled_beta = pooled_beta_metrics(beta_df)

    pooled.to_csv(OUT_DIR / "pooled_yield_metrics_bp.csv", index=False)
    maturity.to_csv(OUT_DIR / "maturity_yield_metrics_bp.csv", index=False)
    pooled_beta.to_csv(OUT_DIR / "pooled_beta_metrics.csv", index=False)

    sanity = sanity_checks(dns, beta_filt_dec, horizon_datasets, yield_df)
    write_run_summary(
        OUT_DIR / "run_summary.md",
        dns=dns,
        pooled_df=pooled,
        sanity=sanity,
        horizon_datasets=horizon_datasets,
        horizon_models=horizon_models,
        fb_ran=fb_ran_any,
    )

    print("\n========== SANITY CHECKS ==========")
    for name, ok, msg in sanity:
        flag = "PASS" if ok else "FAIL"
        print(f"  [{flag}] {name}")
        for line in msg.splitlines():
            print(f"       {line}")

    print("\n========== POOLED RMSE (neural_13, bp) ==========")
    print(
        pooled[pooled["maturity_set"] == "neural_13"]
        .pivot(index="model", columns="horizon", values="pooled_rmse_bp")
        .round(2)
        .to_string()
    )
    print("\n========== POOLED MAE (neural_13, bp) ==========")
    print(
        pooled[pooled["maturity_set"] == "neural_13"]
        .pivot(index="model", columns="horizon", values="pooled_mae_bp")
        .round(2)
        .to_string()
    )

    print(f"\nOutputs written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
