#!/usr/bin/env python3
"""Rebuild the Diebold-Li (2006) DNS monthly U.S. Treasury panel from CRSP (1972–present).

Beginning-of-month timing (daily source)
-----------------------------------------
For each calendar month, keep CRSP **daily** rows on the first priced session on or
after the calendar 1st: ``min(CALDT)`` within the month with
``CALDT >= max(month_start, --sample-start)``. ``panel_month`` is always the calendar
month of that quote (YYYY-MM) to prevent label drift.

WRDS **monthly** files usually use month-end ``MCALDT``; use ``--accept-month-end-monthly``
only if you intentionally accept that timing.

See ``dns_panel_date_audit_1972_present.csv`` for month label vs actual quote date.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from tqdm.auto import tqdm

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 200)

_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_DAILY_CSV = _SCRIPT_DIR / "qz4vtnncykho8sgm.csv"
_DEFAULT_MONTHLY_CSV = _SCRIPT_DIR / "h3ktxzxmldfnwnbs.csv"

_T0 = time.perf_counter()


def log_stage(msg: str) -> None:
    elapsed = time.perf_counter() - _T0
    print(f"[dns_panel +{elapsed:9.1f}s] {msg}", flush=True)


DNS_MONTHS = np.array([3, 6, 9, 12, 15, 18, 21, 24, 30, 36, 48, 60, 72, 84, 96, 108, 120], dtype=float)
DNS_DAYS = DNS_MONTHS * 30.4375
DNS_COLS = [f"y{int(m)}" for m in DNS_MONTHS]
ABS_YIELD_TOL = 0.002

DAILY_USECOLS = [
    "KYTREASNO","KYCRSPID","CRSPID","TCUSIP","TDATDT","TMATDT","IWHY","TCOUPRT","TNIPPY",
    "TVALFC","TFCPDT","IFCPDTF","TFCALDT","TNOTICE","IYMCN","ITYPE","IUNIQ","ITAX","IFLWR",
    "TBANKDT","TSTRIPELIG","TFRGNTGT","CALDT","TDBID","TDASK","TDNOMPRC","TDNOMPRC_FLG",
    "TDSOURCR","TDACCINT","TDRETNUA","TDYLD","TDDURATN","TDPUBOUT","TDTOTOUT","TDPDINT"
]

MONTHLY_USECOLS = [
    "KYTREASNO","KYCRSPID","CRSPID","TCUSIP","TDATDT","TMATDT","IWHY","TCOUPRT","TNIPPY",
    "TVALFC","TFCPDT","IFCPDTF","TFCALDT","TNOTICE","IYMCN","ITYPE","IUNIQ","ITAX","IFLWR",
    "TBANKDT","TSTRIPELIG","TFRGNTGT","MCALDT","TMBID","TMASK","TMNOMPRC","TMNOMPRC_FLG",
    "TMSOURCR","TMACCINT","TMRETNUA","TMYLD","TMDURATN","TMTOTOUT","TMPUBOUT","TMPCYLD",
    "TMRETNXS","TMPDINT"
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild DNS (Diebold-Li 2006) Treasury panel from CRSP")
    parser.add_argument(
        "--source",
        choices=["monthly", "daily"],
        default="daily",
        help="daily: month-start from CALDT (recommended). monthly: WRDS file (often month-end).",
    )
    parser.add_argument("--daily-csv", type=Path, default=_DEFAULT_DAILY_CSV,
                        help="CRSP Treasury daily CSV")
    parser.add_argument("--monthly-csv", type=Path, default=_DEFAULT_MONTHLY_CSV,
                        help="WRDS monthly Treasury CSV")
    parser.add_argument("--output-dir", type=Path, default=Path("./dns_rebuild_output"),
                        help="Directory for outputs and optional checkpoints")
    parser.add_argument("--sample-start", type=str, default="1972-01-01")
    parser.add_argument("--sample-end", type=str, default=pd.Timestamp.today().normalize().strftime("%Y-%m-%d"))
    parser.add_argument("--chunksize", type=int, default=1_000_000,
                        help="Chunk size for reading the full daily CSV")
    parser.add_argument(
        "--accept-month-end-monthly",
        action="store_true",
        help="Allow monthly source when MCALDT is month-end (not month-start timing).",
    )
    parser.add_argument("--checkpoint-every", type=int, default=0, metavar="N",
                        help="If N>0, write dns_panel_checkpoint.parquet every N months in the curve loop")
    parser.add_argument("--progress-log-every", type=int, default=10, metavar="N",
                        help="Log curve progress every N quote months")
    return parser.parse_args()


def require_input_path(path: Path, what: str) -> None:
    if not path.is_file():
        print(f"ERROR: {what} not found or not a file:\n  {path.resolve()}", file=sys.stderr, flush=True)
        raise SystemExit(1)


def _dataframe_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """PyArrow rejects object columns with mixed str/float (e.g. CRSP IDs); stringify them."""
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].astype(str)
    return out


def first_calendar_month_session_snapshot_from_daily(
    daily_csv: Path,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    chunksize: int = 1_000_000,
) -> pd.DataFrame:
    """First priced session on/after calendar 1st within each month (see module docstring)."""
    first_days: dict[pd.Period, pd.Timestamp] = {}
    kept_chunks: list[pd.DataFrame] = []

    log_stage("Snapshot pass 1/2: scan CALDT for each month's first session (on/after 1st).")
    for chunk in tqdm(
        pd.read_csv(daily_csv, usecols=["CALDT"], parse_dates=["CALDT"], chunksize=chunksize),
        desc="Pass 1: month-start sessions",
        file=sys.stdout,
        mininterval=1.0,
    ):
        chunk = chunk[(chunk["CALDT"] >= start_date) & (chunk["CALDT"] <= end_date)]
        if chunk.empty:
            continue
        chunk["month"] = chunk["CALDT"].dt.to_period("M")
        month_start = pd.to_datetime(
            {"year": chunk["CALDT"].dt.year, "month": chunk["CALDT"].dt.month, "day": 1}
        )
        eff_floor = pd.concat([month_start, pd.Series(start_date, index=chunk.index)], axis=1).max(axis=1)
        chunk = chunk[chunk["CALDT"] >= eff_floor]
        if chunk.empty:
            continue
        mins = chunk.groupby("month")["CALDT"].min()
        for month, d in mins.items():
            first_days[month] = min(first_days.get(month, d), d) if month in first_days else d

    if not first_days:
        return pd.DataFrame(columns=DAILY_USECOLS)

    first_day_df = pd.Series(first_days, name="first_day").rename_axis("month").reset_index()
    first_day_map = dict(zip(first_day_df["month"], first_day_df["first_day"]))
    log_stage(f"Snapshot pass 1 complete — {len(first_day_map)} calendar months in range.")

    log_stage("Snapshot pass 2/2: keep full rows on those dates only.")
    for chunk in tqdm(
        pd.read_csv(
            daily_csv,
            usecols=DAILY_USECOLS,
            parse_dates=["CALDT", "TDATDT", "TMATDT", "TFCPDT", "TFCALDT", "TBANKDT"],
            chunksize=chunksize,
            low_memory=False,
        ),
        desc="Pass 2: filter rows",
        file=sys.stdout,
        mininterval=1.0,
    ):
        chunk = chunk[(chunk["CALDT"] >= start_date) & (chunk["CALDT"] <= end_date)]
        if chunk.empty:
            continue
        chunk["month"] = chunk["CALDT"].dt.to_period("M")
        chunk = chunk[chunk["CALDT"] == chunk["month"].map(first_day_map)]
        if not chunk.empty:
            kept_chunks.append(chunk)

    out = pd.concat(kept_chunks, ignore_index=True) if kept_chunks else pd.DataFrame(columns=DAILY_USECOLS)
    # Keep only CALDT here; daily_to_monthly_style() renames CALDT -> MCALDT. Pre-creating MCALDT
    # would duplicate that column after rename and break pd.to_datetime in prepare_crsp_input.
    out = out.drop(columns=["month"], errors="ignore")
    if not out.empty:
        cal0 = pd.to_datetime({"year": out["CALDT"].dt.year, "month": out["CALDT"].dt.month, "day": 1})
        if (out["CALDT"] < cal0).any():
            raise RuntimeError("Internal check failed: CALDT precedes calendar month start.")
    return out


def _monthly_mcaldt_looks_like_month_end(df: pd.DataFrame) -> bool:
    if df.empty:
        return False
    return float(df["MCALDT"].dt.day.median()) >= 26.0


def load_monthly_file(
    path: Path,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    *,
    accept_month_end: bool,
) -> pd.DataFrame:
    log_stage(f"Loading monthly CRSP CSV: {path}")
    df = pd.read_csv(
        path,
        usecols=MONTHLY_USECOLS,
        parse_dates=["MCALDT", "TDATDT", "TMATDT", "TFCPDT", "TFCALDT", "TBANKDT"],
    )
    df = df[(df["MCALDT"] >= start_date) & (df["MCALDT"] <= end_date)].copy()
    log_stage(f"Monthly rows after date filter: {len(df)}")
    if _monthly_mcaldt_looks_like_month_end(df) and not accept_month_end:
        log_stage(
            "ERROR: monthly MCALDT looks like month-end. Use --source daily for month-start quotes, "
            "or pass --accept-month-end-monthly to proceed with month-end timing."
        )
        raise SystemExit(2)
    if _monthly_mcaldt_looks_like_month_end(df):
        log_stage("WARNING: monthly file uses month-end MCALDT (not Diebold-Li month-start convention).")
    return df


def daily_to_monthly_style(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename = {
        "CALDT": "MCALDT",
        "TDBID": "TMBID",
        "TDASK": "TMASK",
        "TDNOMPRC": "TMNOMPRC",
        "TDSOURCR": "TMSOURCR",
        "TDACCINT": "TMACCINT",
        "TDRETNUA": "TMRETNUA",
        "TDYLD": "TMYLD",
        "TDDURATN": "TMDURATN",
        "TDPUBOUT": "TMPUBOUT",
        "TDTOTOUT": "TMTOTOUT",
        "TDPDINT": "TMPDINT",
    }
    return out.rename(columns=rename)


def make_mid_price(df: pd.DataFrame) -> np.ndarray:
    return np.where(
        (df["TMBID"].fillna(-1) > 0) & (df["TMASK"].fillna(-1) > 0),
        (df["TMBID"] + df["TMASK"]) / 2.0,
        np.where(
            df["TMNOMPRC"].fillna(-1) > 0,
            df["TMNOMPRC"],
            np.where(df["TMBID"].fillna(-1) > 0, df["TMBID"], df["TMASK"]),
        ),
    )


def prepare_crsp_input(df: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    log_stage("prepare_crsp_input: date window + CRSP filters + mid prices + YTM")
    x = df.copy()
    x["MCALDT"] = pd.to_datetime(x["MCALDT"])
    x["TDATDT"] = pd.to_datetime(x["TDATDT"])
    x["TMATDT"] = pd.to_datetime(x["TMATDT"])
    for c in ["TFCPDT", "TFCALDT", "TBANKDT"]:
        if c in x.columns:
            x[c] = pd.to_datetime(x[c], errors="coerce")

    x = x[(x["MCALDT"] >= start_date) & (x["MCALDT"] <= end_date)].copy()

    x = x[
        x["ITYPE"].isin([1, 2, 3, 4])
        & (x["ITAX"] == 1)
        & (x["IFLWR"] == 1)
        & (x["TNOTICE"] == 0)
    ].copy()

    x["mid_price"] = make_mid_price(x)
    x["dirty_price"] = x["mid_price"] + x["TMACCINT"].fillna(0.0)
    x["days_to_maturity"] = (x["TMATDT"] - x["MCALDT"]).dt.days.astype(float)
    x["years_to_maturity"] = x["days_to_maturity"] / 365.25

    keep = (((x["ITYPE"] == 4) & (x["days_to_maturity"] >= 30)) |
            ((x["ITYPE"] != 4) & (x["days_to_maturity"] >= 365)))
    x = x[keep].copy()

    x = x[
        x["mid_price"].notna() & (x["mid_price"] > 0) & x["dirty_price"].notna() & (x["dirty_price"] > 0)
    ].copy()

    x["ytm_ann_dec"] = x["TMYLD"] * 365.0
    x["spread"] = (x["TMASK"] - x["TMBID"]).abs().fillna(np.inf)
    x = x.sort_values(["MCALDT", "days_to_maturity", "TCOUPRT", "TCUSIP"], ascending=[True, True, False, True]).reset_index(drop=True)
    log_stage(f"prepare_crsp_input done: {len(x)} bond-day rows, {x['MCALDT'].nunique()} quote dates")
    return x


def remaining_cashflows(row: pd.Series) -> pd.DataFrame:
    quote_date = row["MCALDT"]
    maturity_date = row["TMATDT"]
    coupon_rate = float(row["TCOUPRT"]) / 100.0 if pd.notna(row["TCOUPRT"]) else 0.0
    freq = int(row["TNIPPY"]) if pd.notna(row["TNIPPY"]) else 0

    if (row["ITYPE"] == 4) or (freq == 0) or (coupon_rate == 0.0):
        return pd.DataFrame({"date": [maturity_date], "t": [(maturity_date - quote_date).days / 365.25], "cf": [100.0]})

    step_months = int(round(12 / freq))
    dates = [maturity_date]
    d = maturity_date
    while True:
        d = d - pd.DateOffset(months=step_months)
        if d <= quote_date:
            break
        dates.append(d)

    dates = sorted(dates)
    coupon = 100.0 * coupon_rate / freq
    cashflows = [coupon] * len(dates)
    cashflows[-1] += 100.0
    times = [(dt - quote_date).days / 365.25 for dt in dates]
    return pd.DataFrame({"date": dates, "t": times, "cf": cashflows})


def df_from_segments(t: np.ndarray, breaks: list[float], fwds: list[float]) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    if len(breaks) == 0:
        return np.ones_like(t, dtype=float)

    cum = np.zeros_like(t)
    prev = 0.0
    for b, f in zip(breaks, fwds):
        overlap = np.clip(np.minimum(t, b) - prev, 0.0, None)
        cum += f * overlap
        prev = b
    return np.exp(-cum)


def solve_incremental_forward(row: pd.Series, prev_T: float, prev_breaks: list[float], prev_fwds: list[float]) -> float:
    cf = remaining_cashflows(row)
    known = cf[cf["t"] <= prev_T + 1e-12]
    tail = cf[cf["t"] > prev_T + 1e-12]

    D_prev = float(df_from_segments(np.array([prev_T]), prev_breaks, prev_fwds)[0]) if prev_T > 0 else 1.0
    pv_known = float(np.sum(known["cf"].to_numpy() * df_from_segments(known["t"].to_numpy(), prev_breaks, prev_fwds)))
    target = float(row["dirty_price"]) - pv_known
    if len(tail) == 0:
        return np.nan

    delta = tail["t"].to_numpy() - prev_T
    cfs = tail["cf"].to_numpy()

    def objective(fwd: float) -> float:
        return D_prev * np.sum(cfs * np.exp(-fwd * delta)) - target

    lo, hi = -0.10, 0.30
    flo, fhi = objective(lo), objective(hi)
    expand = 0
    while flo * fhi > 0 and expand < 20:
        lo -= 0.10
        hi += 0.10
        flo, fhi = objective(lo), objective(hi)
        expand += 1

    if flo * fhi > 0:
        return np.nan
    return float(brentq(objective, lo, hi, maxiter=1000))


def _window_mean(vals: np.ndarray) -> float:
    return np.nan if len(vals) == 0 else max(0.0, float(np.mean(vals)))


def first_pass(month_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    g = month_df.sort_values(["days_to_maturity", "TCOUPRT", "TCUSIP"], ascending=[True, False, True]).reset_index(drop=True).copy()
    if len(g) <= 6:
        return g.copy(), g.iloc[0:0].copy()

    included = [g.iloc[0].copy()]
    excluded_rows = []
    for i in range(1, len(g)):
        cand = g.iloc[i]
        shorter = pd.DataFrame(included)[pd.DataFrame(included)["days_to_maturity"] < cand["days_to_maturity"]].tail(3)
        longer = g.iloc[i+1:][g.iloc[i+1:]["days_to_maturity"] > cand["days_to_maturity"]].head(3)

        if len(shorter) < 3 or len(longer) < 3:
            included.append(cand.copy())
            continue

        sbar = _window_mean(shorter["ytm_ann_dec"].to_numpy())
        lbar = _window_mean(longer["ytm_ann_dec"].to_numpy())
        y = max(0.0, float(cand["ytm_ann_dec"]))
        lo, hi = sorted([sbar, lbar])
        if (abs(y - sbar) <= ABS_YIELD_TOL) or (abs(y - lbar) <= ABS_YIELD_TOL) or (lo <= y <= hi):
            included.append(cand.copy())
        else:
            excluded_rows.append(cand.copy())

    kept = pd.DataFrame(included).sort_values(["days_to_maturity", "TCOUPRT", "TCUSIP"], ascending=[True, False, True]).reset_index(drop=True)
    dropped = pd.DataFrame(excluded_rows) if excluded_rows else g.iloc[0:0].copy()
    return kept, dropped


def bootstrap_term_structure(accepted_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    g = accepted_df.sort_values(["TMATDT", "TCOUPRT", "TCUSIP"], ascending=[True, False, True]).reset_index(drop=True).copy()
    if g.empty:
        return pd.DataFrame(), g.iloc[0:0].copy()

    curve_rows = []
    bond_rows = []
    breaks: list[float] = []
    fwds: list[float] = []
    prev_T = 0.0

    for mat_date, group in g.groupby("TMATDT", sort=True):
        T_i = float(group["days_to_maturity"].iloc[0] / 365.25)
        f_candidates = []
        z_candidates = []

        for _, row in group.iterrows():
            fi = solve_incremental_forward(row, prev_T, breaks, fwds)
            if np.isfinite(fi):
                D_prev = float(df_from_segments(np.array([prev_T]), breaks, fwds)[0]) if prev_T > 0 else 1.0
                D_i = D_prev * np.exp(-fi * (T_i - prev_T))
                zi = -np.log(D_i) / T_i
                f_candidates.append(fi)
                z_candidates.append(zi)
                bond_rows.append({
                    "TCUSIP": row["TCUSIP"],
                    "TMATDT": row["TMATDT"],
                    "days_to_maturity": row["days_to_maturity"],
                    "pass_forward": fi,
                    "pass_zero_yield": zi,
                })

        if len(f_candidates) == 0:
            continue

        f_bar = float(np.mean(f_candidates))
        breaks.append(T_i)
        fwds.append(f_bar)
        D_bar = float(df_from_segments(np.array([T_i]), breaks, fwds)[0])
        z_bar = -np.log(D_bar) / T_i
        curve_rows.append({
            "TMATDT": mat_date,
            "days_to_maturity": group["days_to_maturity"].iloc[0],
            "n_issues": len(group),
            "forward_cc": f_bar,
            "zero_yield_cc": z_bar,
        })
        prev_T = T_i

    curve_df = pd.DataFrame(curve_rows)
    bond_curve_df = pd.DataFrame(bond_rows)
    if bond_curve_df.empty:
        out = g.copy()
        out["pass_forward"] = np.nan
        out["pass_zero_yield"] = np.nan
        return curve_df, out

    out = g.merge(bond_curve_df, on=["TCUSIP", "TMATDT", "days_to_maturity"], how="left")
    return curve_df, out


def _find_reversal_sequences(y: np.ndarray) -> list[tuple[int, int]]:
    y = np.asarray(y, dtype=float)
    dy = np.diff(y)
    seqs: list[tuple[int, int]] = []
    i = 0
    while i < len(dy) - 1:
        if abs(dy[i]) <= ABS_YIELD_TOL:
            i += 1
            continue
        sign = np.sign(dy[i])
        j = i + 1
        if j >= len(dy):
            break
        if (abs(dy[j]) > ABS_YIELD_TOL) and (np.sign(dy[j]) == -sign):
            k = j
            while k + 1 < len(dy) and abs(dy[k + 1]) > ABS_YIELD_TOL and np.sign(dy[k + 1]) == sign:
                k += 1
            seqs.append((i, k + 1))
            i = k + 1
        else:
            i += 1
    return seqs


def second_or_fourth_pass(accepted_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if accepted_df.empty:
        return accepted_df.copy(), accepted_df.iloc[0:0].copy(), pd.DataFrame()

    curve_df, bond_curve = bootstrap_term_structure(accepted_df)
    if curve_df.empty:
        return accepted_df.copy(), accepted_df.iloc[0:0].copy(), curve_df

    rep = bond_curve.groupby("TMATDT", as_index=False).agg(
        days_to_maturity=("days_to_maturity", "first"),
        zero_yield_cc=("pass_zero_yield", "mean"),
    ).sort_values("days_to_maturity").reset_index(drop=True)

    seqs = _find_reversal_sequences(rep["zero_yield_cc"].to_numpy())
    delete_mats = set()
    for start_edge, end_edge in seqs:
        positions = list(range(start_edge + 1, end_edge + 1))
        for idx_from_end in range(1, len(positions), 2):
            delete_mats.add(rep.loc[positions[-(idx_from_end + 1)], "TMATDT"])

    kept = accepted_df[~accepted_df["TMATDT"].isin(delete_mats)].copy()
    dropped = accepted_df[accepted_df["TMATDT"].isin(delete_mats)].copy()
    return kept, dropped, curve_df


def implied_zero_for_candidate(row: pd.Series, accepted_df: pd.DataFrame) -> float:
    curve_df, _ = bootstrap_term_structure(accepted_df)
    if curve_df.empty:
        breaks, fwds = [], []
        prev_T = 0.0
    else:
        breaks_all = curve_df["days_to_maturity"].to_numpy() / 365.25
        fwds_all = curve_df["forward_cc"].to_numpy()
        prev_candidates = curve_df[curve_df["TMATDT"] < row["TMATDT"]]
        prev_T = float(prev_candidates["days_to_maturity"].max() / 365.25) if len(prev_candidates) else 0.0
        mask = breaks_all <= prev_T + 1e-12
        breaks = list(breaks_all[mask])
        fwds = list(fwds_all[:np.sum(mask)])

    fi = solve_incremental_forward(row, prev_T, breaks, fwds)
    if not np.isfinite(fi):
        return np.nan

    T = float(row["days_to_maturity"] / 365.25)
    D_prev = float(df_from_segments(np.array([prev_T]), breaks, fwds)[0]) if prev_T > 0 else 1.0
    D = D_prev * np.exp(-fi * (T - prev_T))
    return -np.log(D) / T


def third_pass(pass2_kept: pd.DataFrame, pass12_excluded: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    accepted = pass2_kept.sort_values(["days_to_maturity", "TCOUPRT", "TCUSIP"], ascending=[True, False, True]).copy()
    excluded = pass12_excluded.sort_values(["days_to_maturity", "TCOUPRT", "TCUSIP"], ascending=[True, False, True]).copy()

    put_back = []
    for _, row in excluded.iterrows():
        current_curve, current_bond_curve = bootstrap_term_structure(accepted)
        cand_T = row["days_to_maturity"]

        if current_curve.empty:
            shorter = pd.DataFrame()
            longer = pd.DataFrame()
        else:
            current_bond_curve = current_bond_curve.sort_values(["days_to_maturity", "TCOUPRT", "TCUSIP"], ascending=[True, False, True])
            shorter = current_bond_curve[current_bond_curve["days_to_maturity"] < cand_T].tail(3)
            longer = current_bond_curve[current_bond_curve["days_to_maturity"] > cand_T].head(3)

        if len(shorter) < 3 or len(longer) < 3:
            continue

        z_cand = implied_zero_for_candidate(row, accepted)
        if not np.isfinite(z_cand):
            continue

        sbar = _window_mean(shorter["pass_zero_yield"].to_numpy())
        lbar = _window_mean(longer["pass_zero_yield"].to_numpy())
        lo, hi = sorted([sbar, lbar])
        if (abs(z_cand - sbar) <= ABS_YIELD_TOL) or (abs(z_cand - lbar) <= ABS_YIELD_TOL) or (lo <= z_cand <= hi):
            accepted = pd.concat([accepted, row.to_frame().T], ignore_index=True)
            accepted["TMATDT"] = pd.to_datetime(accepted["TMATDT"])
            accepted["MCALDT"] = pd.to_datetime(accepted["MCALDT"])
            accepted["days_to_maturity"] = accepted["days_to_maturity"].astype(float)
            put_back.append(row)

    put_back_df = pd.DataFrame(put_back) if put_back else excluded.iloc[0:0].copy()
    still_excluded = excluded[~excluded["TCUSIP"].isin(put_back_df["TCUSIP"])].copy() if len(put_back_df) else excluded.copy()
    return accepted, still_excluded, put_back_df


def interpolate_dns_from_curve(curve_df: pd.DataFrame) -> dict[str, float]:
    """Interpolate bootstrapped zero yields to the 17 Diebold-Li maturities (continuously compounded)."""
    curve_df = curve_df.sort_values("days_to_maturity")
    x = curve_df["days_to_maturity"].to_numpy(dtype=float)
    y = curve_df["zero_yield_cc"].to_numpy(dtype=float)

    vals = np.interp(DNS_DAYS, x, y, left=np.nan, right=np.nan)
    if x.min() > DNS_DAYS.min():
        vals[DNS_DAYS < x.min()] = np.nan
    if x.max() < DNS_DAYS.max():
        vals[DNS_DAYS > x.max()] = np.nan
    return dict(zip(DNS_COLS, vals))


def _panel_row_timing(mdate: pd.Timestamp) -> dict[str, object]:
    ts = pd.Timestamp(mdate).normalize()
    cal_start = pd.Timestamp(ts.year, ts.month, 1)
    return {
        "panel_month": ts.strftime("%Y-%m"),
        "calendar_month_start": cal_start,
        "actual_quote_date": ts,
        "exact_calendar_first_of_month": bool(ts.day == 1),
        "calendar_days_after_month_start": int((ts - cal_start).days),
    }


def build_dns_panel(
    crsp_input: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    *,
    checkpoint_every: int = 0,
    progress_every: int = 10,
    output_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    log_stage("Curve stage: bootstrapping + DNS maturity interpolation (month loop).")
    base = prepare_crsp_input(crsp_input, start_date, end_date)
    month_outputs: list[dict[str, object]] = []
    accepted_bonds_all: list[pd.DataFrame] = []
    curve_points_all: list[pd.DataFrame] = []

    n_months = int(base["MCALDT"].nunique())
    log_stage(f"Processing {n_months} distinct quote months (groupby MCALDT).")

    for idx, (mdate, month_df) in enumerate(
        tqdm(
            base.groupby("MCALDT", sort=True),
            total=n_months,
            desc="Monthly DNS curve",
            file=sys.stdout,
            mininterval=1.0,
        ),
        start=1,
    ):
        if progress_every > 0 and (idx == 1 or idx == n_months or idx % progress_every == 0):
            log_stage(f"Curve build progress: {idx}/{n_months} — quote date {pd.Timestamp(mdate).date()}")

        timing = _panel_row_timing(pd.Timestamp(mdate))

        p1_kept, p1_excl = first_pass(month_df)
        p2_kept, p2_drop, _curve_p2 = second_or_fourth_pass(p1_kept)
        pass12_excl = pd.concat([p1_excl, p2_drop], ignore_index=True).drop_duplicates(subset=["TCUSIP", "TMATDT"])
        p3_kept, _p3_excl, _p3_putback = third_pass(p2_kept, pass12_excl)
        p4_kept, _p4_drop, _curve_p4 = second_or_fourth_pass(p3_kept)
        final_curve, _final_bond_curve = bootstrap_term_structure(p4_kept)

        row: dict[str, object] = {
            "date": timing["actual_quote_date"],
            **timing,
            "n_raw": len(month_df),
            "n_pass1": len(p1_kept),
            "n_pass2": len(p2_kept),
            "n_pass3": len(p3_kept),
            "n_pass4": len(p4_kept),
        }
        row.update(interpolate_dns_from_curve(final_curve) if not final_curve.empty else {c: np.nan for c in DNS_COLS})
        month_outputs.append(row)

        kept = p4_kept.copy()
        kept["date"] = timing["actual_quote_date"]
        kept["panel_month"] = timing["panel_month"]
        accepted_bonds_all.append(kept)

        if not final_curve.empty:
            cp = final_curve.copy()
            cp["date"] = timing["actual_quote_date"]
            cp["panel_month"] = timing["panel_month"]
            curve_points_all.append(cp)

        if checkpoint_every > 0 and output_dir is not None and idx % checkpoint_every == 0:
            partial = pd.DataFrame(month_outputs).sort_values("date").reset_index(drop=True)
            output_dir.mkdir(parents=True, exist_ok=True)
            ck = output_dir / "dns_panel_checkpoint.parquet"
            _dataframe_for_parquet(partial).to_parquet(ck, index=False)
            log_stage(f"Checkpoint written ({len(partial)} rows): {ck}")

    dns_panel = pd.DataFrame(month_outputs).sort_values("date").reset_index(drop=True)
    accepted_bonds = pd.concat(accepted_bonds_all, ignore_index=True) if accepted_bonds_all else pd.DataFrame()
    curve_points = pd.concat(curve_points_all, ignore_index=True) if curve_points_all else pd.DataFrame()
    log_stage("Curve stage: all quote months finished.")
    return dns_panel, accepted_bonds, curve_points


def expected_panel_month_labels(sample_start: pd.Timestamp, sample_end: pd.Timestamp) -> list[str]:
    t = pd.Timestamp(sample_start.year, sample_start.month, 1)
    e = pd.Timestamp(sample_end.year, sample_end.month, 1)
    out: list[str] = []
    while t <= e:
        out.append(t.strftime("%Y-%m"))
        t = t + pd.DateOffset(months=1)
    return out


def validate_and_report_panel(
    dns_panel: pd.DataFrame,
    sample_start: pd.Timestamp,
    sample_end: pd.Timestamp,
) -> None:
    log_stage("Validation: DNS columns, panel_month keys, quote dates, month coverage vs CLI range")
    missing = [c for c in DNS_COLS if c not in dns_panel.columns]
    if missing:
        log_stage(f"ERROR: missing DNS yield columns: {missing}")
        raise SystemExit(3)
    if dns_panel.empty:
        log_stage("WARNING: empty panel — skipping validation")
        return

    if dns_panel["panel_month"].duplicated().any():
        log_stage("ERROR: duplicate panel_month")
        raise SystemExit(4)

    chk = pd.to_datetime(dns_panel["actual_quote_date"]).dt.strftime("%Y-%m")
    if not (chk == dns_panel["panel_month"].astype(str)).all():
        log_stage("ERROR: panel_month must match calendar month of actual_quote_date")
        raise SystemExit(5)

    expected = expected_panel_month_labels(sample_start, sample_end)
    have = set(dns_panel["panel_month"].astype(str))
    miss = sorted(set(expected) - have)
    extra = sorted(have - set(expected))
    log_stage(f"Coverage: {len(expected)} months in CLI range; panel has {len(have)} rows.")
    if miss:
        log_stage(f"WARNING: {len(miss)} expected months missing (first 36): {miss[:36]}")
    if extra:
        log_stage(f"INFO: {len(extra)} panel months outside CLI month list (first 12): {extra[:12]}")

    n_exact = int(dns_panel["exact_calendar_first_of_month"].sum())
    log_stage(
        f"Quote dates: {n_exact} on exact calendar 1st; {len(dns_panel) - n_exact} first session after 1st "
        f"(or after --sample-start in the first month)."
    )

    dts = pd.to_datetime(dns_panel["actual_quote_date"])
    if not dts.is_monotonic_increasing:
        log_stage("ERROR: actual_quote_date not strictly increasing")
        raise SystemExit(6)
    if dts.dt.to_period("M").duplicated().any():
        log_stage("ERROR: duplicate calendar month in actual_quote_date")
        raise SystemExit(7)

    show = ["panel_month", "actual_quote_date", "exact_calendar_first_of_month"] + DNS_COLS[:4]
    log_stage("First panel rows:")
    print(dns_panel[show].head(5).to_string(), flush=True)
    log_stage("Last panel rows:")
    print(dns_panel[show].tail(5).to_string(), flush=True)


def write_date_audit(dns_panel: pd.DataFrame, output_dir: Path) -> None:
    audit = pd.DataFrame({
        "month_label": dns_panel["panel_month"],
        "actual_observation_date_used": pd.to_datetime(dns_panel["actual_quote_date"]).dt.strftime("%Y-%m-%d"),
        "exact_first_of_month_flag": dns_panel["exact_calendar_first_of_month"].astype(bool),
        "calendar_month_start": pd.to_datetime(dns_panel["calendar_month_start"]).dt.strftime("%Y-%m-%d"),
        "calendar_days_after_month_start": dns_panel["calendar_days_after_month_start"],
    })
    path = output_dir / "dns_panel_date_audit_1972_present.csv"
    audit.to_csv(path, index=False)
    log_stage(f"Wrote date audit: {path}")


def export_outputs(dns_panel: pd.DataFrame, accepted_bonds: pd.DataFrame, curve_points: pd.DataFrame, output_dir: Path) -> None:
    log_stage("Export: writing panel CSVs, model-ready percent file, and date audit.")
    output_dir.mkdir(parents=True, exist_ok=True)
    dns_panel.to_csv(output_dir / "dns_panel_rebuilt_1972_present.csv", index=False)
    accepted_bonds.to_csv(output_dir / "accepted_bonds_rebuilt_1972_present.csv", index=False)
    curve_points.to_csv(output_dir / "curve_points_rebuilt_1972_present.csv", index=False)

    model_ready = dns_panel.copy()
    model_ready["obs"] = model_ready["panel_month"].str.replace("-", "M", regex=False)
    export = model_ready[["obs", "panel_month", "actual_quote_date", "exact_calendar_first_of_month"] + DNS_COLS].copy()
    for c in DNS_COLS:
        export[c] = 100 * export[c]
    rename = {f"y{int(m)}": f"M{int(m)}" for m in DNS_MONTHS}
    export = export.rename(columns=rename)
    export.to_csv(output_dir / "dns_panel_model_ready_percent_1972_present.csv", index=False)
    write_date_audit(dns_panel, output_dir)


def main() -> None:
    args = parse_args()
    start_date = pd.Timestamp(args.sample_start)
    end_date = pd.Timestamp(args.sample_end)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    log_stage(f"Run start | source={args.source} | {args.sample_start} .. {args.sample_end}")

    if args.source == "daily":
        require_input_path(args.daily_csv, "Daily Treasury CSV (--daily-csv)")
        log_stage(f"Daily input: {args.daily_csv.resolve()}")
        daily_snap = first_calendar_month_session_snapshot_from_daily(
            args.daily_csv, start_date, end_date, args.chunksize
        )
        crsp_input = daily_to_monthly_style(daily_snap)
        snap_path = args.output_dir / "daily_month_start_snapshot_1972_present.parquet"
        _dataframe_for_parquet(daily_snap).to_parquet(snap_path, index=False)
        log_stage(f"Saved month-start daily snapshot: {snap_path}")
    else:
        require_input_path(args.monthly_csv, "Monthly Treasury CSV (--monthly-csv)")
        crsp_input = load_monthly_file(
            args.monthly_csv,
            start_date,
            end_date,
            accept_month_end=args.accept_month_end_monthly,
        )

    log_stage(f"CRSP input shape after load / snapshot: {crsp_input.shape}")
    dns_panel, accepted_bonds, curve_points = build_dns_panel(
        crsp_input,
        start_date,
        end_date,
        checkpoint_every=args.checkpoint_every,
        progress_every=args.progress_log_every,
        output_dir=args.output_dir,
    )
    export_outputs(dns_panel, accepted_bonds, curve_points, args.output_dir)
    validate_and_report_panel(dns_panel, start_date, end_date)

    log_stage("Run complete.")
    log_stage(f"Outputs directory: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
