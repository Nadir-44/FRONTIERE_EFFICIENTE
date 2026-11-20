"""
=============================================================
PORTFOLIO CONSTRUCTION PIPELINE — Streamlit App
Author  : Nadir Stiti
Purpose : Weekly PM meeting — Efficient Frontier builder
          Tab 1 : Equity
          Tab 2 : Multi-Asset (class + per-instrument constraints)
=============================================================
"""

import warnings
warnings.filterwarnings("ignore")

import io
import os
import json
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import minimize

# ─────────────────────────────────────────────────────────────
# 0. PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Portfolio Construction Pipeline",
    page_icon="📈",
    layout="wide",
)
st.title("📈 Portfolio Construction Pipeline")
st.caption("Weekly PM meeting tool · Efficient Frontier builder")

# ─────────────────────────────────────────────────────────────
# 0bis. SAVED CONFIGURATIONS (named fund parameter presets — Multi-Asset only)
# ─────────────────────────────────────────────────────────────
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fund_configs.json")


def load_all_configs() -> dict:
    """Read the on-disk store of named multi-asset fund configurations."""
    if not os.path.exists(CONFIG_FILE):
        return {"multi_asset": {}}
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {"multi_asset": {}}
    data.setdefault("multi_asset", {})
    return data


def save_all_configs(data: dict) -> None:
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


WEEKS_YEAR = 52
COL_LABEL = 0
COL_DATE = 1
COL_DATA_START = 2
ID_FIELD_HINTS = ("isin", "name", "produit", "product", "ticker")

ASSET_CLASSES = ["Equity", "Fixed Income", "Commodities", "FX", "Real Estate", "Cash"]
UNCLASSIFIED = "Unclassified"

ASSET_CLASS_SYNONYMS = {
    "equity": "Equity", "equities": "Equity", "stock": "Equity", "stocks": "Equity",
    "action": "Equity", "actions": "Equity", "share": "Equity", "shares": "Equity",
    "fixed income": "Fixed Income", "bond": "Fixed Income", "bonds": "Fixed Income",
    "fi": "Fixed Income", "obligation": "Fixed Income", "obligations": "Fixed Income",
    "credit": "Fixed Income", "govt": "Fixed Income", "government bond": "Fixed Income",
    "commodity": "Commodities", "commodities": "Commodities",
    "matieres premieres": "Commodities", "matières premières": "Commodities",
    "fx": "FX", "currency": "FX", "currencies": "FX", "devise": "FX", "devises": "FX",
    "real estate": "Real Estate", "reit": "Real Estate", "reits": "Real Estate",
    "immobilier": "Real Estate", "property": "Real Estate",
    "cash": "Cash", "money market": "Cash", "liquidites": "Cash", "liquidités": "Cash",
}


def normalize_asset_class(raw) -> str:
    if pd.isna(raw):
        return UNCLASSIFIED
    s = str(raw).strip().lower()
    if s in ("", "nan", "#n/a", "n/a", "na", "none"):
        return UNCLASSIFIED
    return ASSET_CLASS_SYNONYMS.get(s, UNCLASSIFIED)


# ─────────────────────────────────────────────────────────────
# 1. SHARED OPTIMISATION HELPERS (used by both tabs)
# ─────────────────────────────────────────────────────────────
def portfolio_stats(w, mu_arr, Sigma_arr):
    ret = float(w @ mu_arr)
    vol = float(np.sqrt(w @ Sigma_arr @ w))
    return ret, vol


def build_class_constraints(tickers, class_of: dict, class_bounds: dict):
    """One (>=min) and one (<=max) inequality constraint per class present.
    Analytical jacobians (constant indicator vectors) — cheap and exact,
    avoids SLSQP falling back to finite-difference approximation."""
    N = len(tickers)
    constraints = []
    classes = sorted(set(class_of.values()))
    for c in classes:
        if c not in class_bounds:
            continue
        idx = [i for i, t in enumerate(tickers) if class_of[t] == c]
        if not idx:
            continue
        cmin, cmax = class_bounds[c]

        def _max_jac(w, idx=idx, N=N):
            g = np.zeros(N)
            g[idx] = -1.0
            return g

        def _min_jac(w, idx=idx, N=N):
            g = np.zeros(N)
            g[idx] = 1.0
            return g

        constraints.append({"type": "ineq", "fun": lambda w, idx=idx, cmax=cmax: cmax - np.sum(w[idx]), "jac": _max_jac})
        constraints.append({"type": "ineq", "fun": lambda w, idx=idx, cmin=cmin: np.sum(w[idx]) - cmin, "jac": _min_jac})
    return constraints


def _vol_ineq_jac(w, Sigma_arr):
    """Gradient of (vol_target - sqrt(w'Σw)) w.r.t. w. Guards against div-by-zero at w=0."""
    Sw = Sigma_arr @ w
    vol = float(np.sqrt(w @ Sw))
    if vol < 1e-12:
        return np.zeros_like(w)
    return -Sw / vol


def get_min_vol_portfolio(mu_arr, Sigma_arr, N, bounds, extra_constraints=None):
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1, "jac": lambda w: np.ones(N)}]
    if extra_constraints:
        constraints += extra_constraints
    res = minimize(
        lambda w: float(w @ Sigma_arr @ w),
        np.ones(N) / N, method="SLSQP", bounds=bounds, constraints=constraints,
        jac=lambda w: 2 * (Sigma_arr @ w),
        options={"ftol": 1e-12, "maxiter": 3000},
    )
    if not res.success:
        return None, None
    lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
    w = np.clip(res.x, lo, hi)
    s = w.sum()
    if s <= 0:
        return None, None
    w = w / s
    _, vol = portfolio_stats(w, mu_arr, Sigma_arr)
    return w, vol


def get_max_return_portfolio(mu_arr, Sigma_arr, N, bounds, extra_constraints=None):
    """Unconstrained-by-vol max-return portfolio — its volatility is the ceiling
    attainable given the weight/class bounds alone. Linear objective + linear
    constraints (sum=1, box bounds, class sums) => convex problem, any KKT
    point found is the global optimum, so a single start is enough."""
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1, "jac": lambda w: np.ones(N)}]
    if extra_constraints:
        constraints += extra_constraints
    res = minimize(
        lambda w: float(-(w @ mu_arr)),
        np.ones(N) / N, method="SLSQP", bounds=bounds, constraints=constraints,
        jac=lambda w: -mu_arr,
        options={"ftol": 1e-10, "maxiter": 2000},
    )
    if not res.success:
        return None, None
    lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
    w = np.clip(res.x, lo, hi)
    s = w.sum()
    if s <= 0:
        return None, None
    w = w / s
    ret, vol = portfolio_stats(w, mu_arr, Sigma_arr)
    return w, vol


def max_return_given_vol(
    vol_target, mu_arr, Sigma_arr, N, n_starts, bounds,
    extra_constraints=None, w_minvol=None, min_vol=None,
    w_maxret=None, max_vol=None, seed=42,
):
    if min_vol is not None and vol_target < min_vol - 0.005:
        w = w_minvol.copy() if w_minvol is not None else np.array([(b[0] + b[1]) / 2 for b in bounds])
        ret, vol = portfolio_stats(w, mu_arr, Sigma_arr)
        return {"w": w, "ret": ret, "vol": vol, "feasible": "below_min_vol"}

    if max_vol is not None and vol_target > max_vol + 0.005:
        w = w_maxret.copy() if w_maxret is not None else np.array([(b[0] + b[1]) / 2 for b in bounds])
        ret, vol = portfolio_stats(w, mu_arr, Sigma_arr)
        return {"w": w, "ret": ret, "vol": vol, "feasible": "above_max_vol"}

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1, "jac": lambda w: np.ones(N)},
        {
            "type": "ineq",
            "fun": lambda w: vol_target - float(np.sqrt(w @ Sigma_arr @ w)),
            "jac": lambda w: _vol_ineq_jac(w, Sigma_arr),
        },
    ]
    if extra_constraints:
        constraints += extra_constraints

    rng = np.random.default_rng(seed)
    lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
    candidates = []
    if w_minvol is not None:
        candidates.append(w_minvol.copy())
    for _ in range(max(int(n_starts) - len(candidates), 0)):
        c = rng.dirichlet(np.ones(N))
        c = np.clip(c, lo, hi)
        if c.sum() > 0:
            c = c / c.sum()
        candidates.append(c)

    best = None
    for w0 in candidates:
        res = minimize(
            lambda w: float(-(w @ mu_arr)),
            w0, method="SLSQP", bounds=bounds, constraints=constraints,
            jac=lambda w: -mu_arr,
            options={"ftol": 1e-9, "maxiter": 300},
        )
        if res.success:
            w = np.clip(res.x, lo, hi)
            s = w.sum()
            if s <= 0:
                continue
            w = w / s
            vol = float(np.sqrt(w @ Sigma_arr @ w))
            ret = float(w @ mu_arr)
            if vol <= vol_target + 1e-6:
                if best is None or ret > best["ret"]:
                    best = {"w": w, "ret": ret, "vol": vol, "feasible": "optimized"}

    if best is None:
        w = w_minvol.copy() if w_minvol is not None else np.array([(b[0] + b[1]) / 2 for b in bounds])
        ret, vol = portfolio_stats(w, mu_arr, Sigma_arr)
        best = {"w": w, "ret": ret, "vol": vol, "feasible": "min_vol_fallback"}

    return best


def build_chart(df_results, tickers, mu_arr, Sigma_arr, risk_free, n_sim, N, bounds=None):
    vols = df_results["realized_vol"].values.astype(float)
    rets = df_results["expected_return"].values.astype(float)
    sharpes = df_results["sharpe_ratio"].values.astype(float)
    vtargets = df_results["vol_target"].values.astype(float)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax_main, ax_bar) = plt.subplots(
        1, 2, figsize=(20, 9), gridspec_kw={"width_ratios": [3, 1]}, facecolor="white"
    )

    ax = ax_main
    ax.set_facecolor("#f7f9fc")

    rng = np.random.default_rng(0)
    if bounds is not None:
        lo_arr = np.array([b[0] for b in bounds])
        hi_arr = np.array([b[1] for b in bounds])
    else:
        lo_arr = np.zeros(N)
        hi_arr = np.ones(N)

    # A full-N Dirichlet(1,...,1) draw collapses to ~1/N per weight when N is
    # large (e.g. 500+ tickers), producing near-identical, over-diversified
    # portfolios — a tight blob instead of the usual Markowitz "bullet" cloud,
    # and it ignores the per-instrument bounds entirely. Instead, draw over a
    # random SPARSE subset of tickers sized to plausibly hit the bounds, so
    # the simulated cloud actually varies in composition like the real
    # (concentrated, bound-constrained) frontier portfolios do.
    avg_max = float(np.mean(hi_arr[hi_arr > 0])) if np.any(hi_arr > 0) else 1.0
    k_typical = max(2, min(N, int(np.ceil(1.0 / avg_max)))) if avg_max > 0 else N
    k_lo = max(2, k_typical // 2)
    k_hi = max(k_lo + 1, min(N, k_typical * 2))

    sim_v, sim_r, sim_s = [], [], []
    for _ in range(int(n_sim)):
        k = int(rng.integers(k_lo, k_hi + 1))
        idx = rng.choice(N, size=k, replace=False)
        c = np.zeros(N)
        sub = rng.dirichlet(np.ones(k))
        sub = np.clip(sub, lo_arr[idx], hi_arr[idx])
        sub_sum = sub.sum()
        if sub_sum <= 0:
            continue
        c[idx] = sub / sub_sum
        r = float(c @ mu_arr)
        v = float(np.sqrt(c @ Sigma_arr @ c))
        s = (r - risk_free) / v if v > 1e-8 else 0
        sim_v.append(v); sim_r.append(r); sim_s.append(s)
    sim_v, sim_r, sim_s = np.array(sim_v), np.array(sim_r), np.array(sim_s)
    sim_s_cl = np.clip(sim_s, np.percentile(sim_s, 5), np.percentile(sim_s, 95))
    ax.scatter(sim_v, sim_r, c=sim_s_cl, cmap="RdYlGn", s=12, alpha=0.35, linewidths=0, zorder=1)

    order = np.argsort(vols)
    ax.plot(vols[order], rets[order], color="#1565C0", lw=2.5, alpha=0.9, label="Efficient Frontier", zorder=3)

    valid_sharpes = sharpes[~np.isnan(sharpes)]
    norm = plt.Normalize(vmin=valid_sharpes.min(), vmax=valid_sharpes.max())
    cmap_pts = cm.RdYlGn
    sc_pts = ax.scatter(vols, rets, c=sharpes, cmap=cmap_pts, norm=norm, s=260, zorder=6,
                         edgecolors="#1565C0", linewidths=1.5)

    cbar = plt.colorbar(sc_pts, ax=ax, pad=0.02, fraction=0.025, aspect=30)
    cbar.set_label("Sharpe Ratio", fontsize=11, labelpad=10)
    cbar.ax.tick_params(labelsize=9)

    for i, (v, r, s, vt) in enumerate(zip(vols, rets, sharpes, vtargets)):
        ax.annotate(
            f"{vt*100:.2f}%",
            xy=(v, r),
            xytext=(0, 10),
            textcoords="offset points",
            fontsize=7.5,
            fontweight="bold",
            color="#1a1a2e",
            ha="center",
            va="bottom",
            zorder=10,
        )

    # ── Clamp X-axis to the optimized frontier range (not the simulated cloud) ──
    vol_pad = (vols.max() - vols.min()) * 0.08   # 8% padding left/right
    ax.set_xlim(
        max(0.0, vols.min() - vol_pad),
        vols.max() + vol_pad,
    )

    # Clamp Y-axis to the optimized frontier range
    ret_pad = (rets.max() - rets.min()) * 0.15
    ax.set_ylim(
        rets.min() - ret_pad,
        rets.max() + ret_pad,
    )

    ax.set_xlabel("Annualized Volatility", fontsize=13, labelpad=10, color="#222")
    ax.set_ylabel("Annualized Return", fontsize=13, labelpad=10, color="#222")
    ax.set_title("Efficient Frontier — Best Performing Baskets per Volatility Target",
                 fontsize=14, fontweight="bold", pad=16, color="#1a1a2e")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*100:.1f}%"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y*100:.1f}%"))
    ax.tick_params(labelsize=10, colors="#333")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6, color="#ccc")
    ax.legend(fontsize=10, loc="upper left", framealpha=0.9, edgecolor="#aaa")

    ax2 = ax_bar
    ax2.set_facecolor("#f7f9fc")
    labels = [f"{vt*100:.0f}%" for vt in vtargets]
    bar_colors = [cmap_pts(norm(s)) if not np.isnan(s) else (0.7, 0.7, 0.7) for s in sharpes]
    bars = ax2.barh(labels, sharpes, color=bar_colors, edgecolor="#555", linewidth=0.7, height=0.6)
    for bar, s in zip(bars, sharpes):
        x_pos = bar.get_width()
        label_x = x_pos + 0.01 if x_pos >= 0 else x_pos - 0.01
        ha_val = "left" if x_pos >= 0 else "right"
        ax2.text(label_x, bar.get_y() + bar.get_height() / 2, f"{s:.3f}",
                  va="center", ha=ha_val, fontsize=9, fontweight="bold", color="#1a1a2e")
    ax2.axvline(0, color="#333", lw=0.8)
    ax2.set_xlabel("Sharpe Ratio", fontsize=12, labelpad=8)
    ax2.set_title("Sharpe per\nVol Target", fontsize=12, fontweight="bold", pad=12, color="#1a1a2e")
    ax2.tick_params(labelsize=10)
    ax2.grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.5)
    ax2.invert_yaxis()

    fig.text(
        0.5, 0.01,
        f"Generated on {pd.Timestamp.today().strftime('%Y-%m-%d')} | "
        f"Risk-free rate: {risk_free*100:.0f}% | Universe: {N} assets",
        ha="center", fontsize=8, color="#666",
    )
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    return fig


# ═════════════════════════════════════════════════════════════
# TAB 1 — EQUITY  (fixed row layout, sector + per-instrument bounds)
# ═════════════════════════════════════════════════════════════
# Row 3 (1-indexed) -> GICS 1 (secteur), starting col B
# Row 4 (1-indexed) -> GICS 2 (sous-secteur), starting col B
# Row 5 (1-indexed) -> Ticker, starting col B
# Rows 6-7          -> irrelevant data, ignored
# Row 8+            -> Prices ; dates in col A, values from col B
EQ_ROW_GICS1 = 2       # 0-indexed row 3
EQ_ROW_GICS2 = 3       # 0-indexed row 4
EQ_ROW_TICKER = 4      # 0-indexed row 5
EQ_ROW_PRICES = 7      # 0-indexed row 8
EQ_COL_DATE = 0         # column A
EQ_COL_DATA_START = 1   # column B

# 👉 Chemin local par défaut, chargé automatiquement si aucun fichier n'est uploadé (Ctrl+F pour remplacer).
EQ_DEFAULT_FILE_PATH = r"P:\GESTION_RPSE\GES - Gestion Epargne Salariale\16_PERSONNEL\Nadir\FRONTIERE_EFFICIENTE\UNIVERS_BLOOM.xlsx"

# ── Benchmark sheet layout ────────────────────────────────
# Row 4 -> nom de chaque indice, col B+ (ex: "SX5E Index"), plusieurs indices possibles
# Row 7+ -> prix ; dates en col A, valeurs à partir de col B
BENCH_SHEET_NAME = "BENCH"
BENCH_ROW_NAME = 3      # 0-indexed row 4
BENCH_ROW_PRICES = 6    # 0-indexed row 7
BENCH_COL_DATE = 0       # column A
BENCH_COL_DATA_START = 1  # column B


def find_sheet_name(sheet_names, target):
    """Case/whitespace-insensitive lookup — returns the actual sheet name or None."""
    norm_target = target.strip().lower()
    for s in sheet_names:
        if s.strip().lower() == norm_target:
            return s
    return None


@st.cache_data(show_spinner=False)
def load_benchmark(file_bytes: bytes, sheet_name=BENCH_SHEET_NAME) -> pd.DataFrame:
    """Returns a DataFrame indexed by date, one column per benchmark index found."""
    raw = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name, header=None, index_col=None)

    name_row = raw.iloc[BENCH_ROW_NAME, BENCH_COL_DATA_START:]
    names_raw = []
    for nm in name_row.values:
        nm_str = str(nm).strip()
        if nm_str in ("", "nan", "NaN"):
            continue
        names_raw.append(nm_str)
    n_bench_raw = len(names_raw)

    price_block = raw.iloc[BENCH_ROW_PRICES:, :].copy()
    dates = pd.to_datetime(price_block.iloc[:, BENCH_COL_DATE], errors="coerce")
    prices = price_block.iloc[:, BENCH_COL_DATA_START:BENCH_COL_DATA_START + n_bench_raw].copy()
    prices.columns = names_raw
    prices.index = dates
    prices = prices.loc[:, ~prices.columns.duplicated(keep="first")]

    prices = prices[prices.index.notna()]
    prices = prices.apply(pd.to_numeric, errors="coerce")
    prices = prices.dropna(axis=1, how="all")
    return prices.sort_index()


# ── Benchmark GICS 1 weights sheet layout ─────────────────
# Row 1 -> nom de chaque bench, col B+ (mêmes noms que la feuille "bench")
# Row 2+ -> col A = nom du secteur GICS 1, col B+ = poids en décimal (0.15 = 15%)
BENCH_POIDS_SHEET_NAME = "BENCH_POIDS"
BENCH_POIDS_ROW_NAME = 0        # 0-indexed row 1
BENCH_POIDS_ROW_DATA_START = 1  # 0-indexed row 2
BENCH_POIDS_COL_SECTOR = 0       # column A
BENCH_POIDS_COL_DATA_START = 1   # column B

# ── Benchmark GICS 2 weights sheet layout ─────────────────
# Ligne d'en-tête détectée automatiquement (actuellement ligne 7) : col A = "GICS 2",
# col B+ = nom de chaque bench ("SX5E Index", "SXXP Index"... mêmes noms que BENCH).
# Lignes suivantes : col A = sous-secteur GICS 2, col B+ = poids (format % Excel = décimal).
# Plusieurs orthographes acceptées pour le nom d'onglet (insensible à la casse).
BENCH_POIDS_GICS2_SHEET_NAMES = ["GICS_2", "GICS 2", "GICS2"]

# 👉 Feuilles à exclure du sélecteur "Univers" (Equity) — ajoute ici tout nom de
# feuille qui n'est pas un univers investissable. Comparaison insensible à la
# casse/espaces (via find_sheet_name).
EQ_EXCLUDED_SHEET_NAMES = [BENCH_SHEET_NAME, BENCH_POIDS_SHEET_NAME, "GICS", "GICS 1", "GICS_1", "GICS1"] + BENCH_POIDS_GICS2_SHEET_NAMES


@st.cache_data(show_spinner=False)
def load_bench_weights(file_bytes: bytes, sheet_name=BENCH_POIDS_SHEET_NAME, auto_header=False) -> pd.DataFrame:
    """Returns a DataFrame indexed by sector name, one column per benchmark, decimal weights.

    auto_header=False : ligne des noms de bench fixée à BENCH_POIDS_ROW_NAME (format BENCH_POIDS).
    auto_header=True  : ligne des noms de bench = première ligne non vide à partir de la col B
                        (ex. onglet "GICS 2" où les en-têtes sont en ligne 7).
    """
    raw = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name, header=None, index_col=None)

    header_row = BENCH_POIDS_ROW_NAME
    if auto_header:
        block = raw.iloc[:, BENCH_POIDS_COL_DATA_START:]
        non_empty = block.notna().any(axis=1)
        if non_empty.any():
            header_row = int(non_empty.idxmax())
    data_start_row = header_row + 1

    name_row = raw.iloc[header_row, BENCH_POIDS_COL_DATA_START:]
    names_raw = []
    for nm in name_row.values:
        nm_str = str(nm).strip()
        if nm_str in ("", "nan", "NaN"):
            continue
        names_raw.append(nm_str)
    n_bench_raw = len(names_raw)

    data_block = raw.iloc[data_start_row:, :]
    sectors = data_block.iloc[:, BENCH_POIDS_COL_SECTOR].astype(str).str.strip()
    weights = data_block.iloc[:, BENCH_POIDS_COL_DATA_START:BENCH_POIDS_COL_DATA_START + n_bench_raw].copy()
    weights.columns = names_raw
    weights = weights.apply(pd.to_numeric, errors="coerce")
    # Sécurité : si les poids sont saisis en points (4.22) au lieu de décimal (0.0422)
    if auto_header and weights.max().max() > 1.5:
        weights = weights / 100
    weights.index = sectors
    weights = weights[weights.index.notna() & (weights.index != "") & (weights.index.str.lower() != "nan")]
    weights = weights.loc[~weights.index.duplicated(keep="first")]
    return weights


@st.cache_data(show_spinner=False)
def get_sheet_names(file_bytes: bytes) -> list:
    return pd.ExcelFile(io.BytesIO(file_bytes), engine="openpyxl").sheet_names



@st.cache_data(show_spinner=False)
def eq_parse_sheet(file_bytes: bytes, sheet_name):
    """Raw per-sheet parse — no resample/lookback/cleaning yet, so multiple
    sheets can be concatenated first and cleaned once as a combined universe."""
    raw = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name, header=None, index_col=None)

    gics1_row = raw.iloc[EQ_ROW_GICS1, EQ_COL_DATA_START:]
    gics2_row = raw.iloc[EQ_ROW_GICS2, EQ_COL_DATA_START:]
    ticker_row = raw.iloc[EQ_ROW_TICKER, EQ_COL_DATA_START:]

    tickers_raw, gics1_map, gics2_map = [], {}, {}
    for tkr, g1, g2 in zip(ticker_row.values, gics1_row.values, gics2_row.values):
        tkr_str = str(tkr).strip()
        if tkr_str in ("", "nan", "NaN"):
            continue
        g1_str = str(g1).strip()
        g2_str = str(g2).strip()
        tickers_raw.append(tkr_str)
        gics1_map[tkr_str] = g1_str if g1_str not in ("", "nan", "NaN") else UNCLASSIFIED
        gics2_map[tkr_str] = g2_str if g2_str not in ("", "nan", "NaN") else UNCLASSIFIED
    n_assets_raw = len(tickers_raw)

    price_block = raw.iloc[EQ_ROW_PRICES:, :].copy()
    dates = pd.to_datetime(price_block.iloc[:, EQ_COL_DATE], errors="coerce")
    prices = price_block.iloc[:, EQ_COL_DATA_START:EQ_COL_DATA_START + n_assets_raw].copy()
    prices.columns = tickers_raw
    prices.index = dates
    prices = prices.loc[:, ~prices.columns.duplicated(keep="first")]

    prices = prices[prices.index.notna()]
    prices = prices.apply(pd.to_numeric, errors="coerce")
    prices = prices.dropna(axis=1, how="all")

    return prices, gics1_map, gics2_map


def eq_load_and_clean_multi(file_bytes: bytes, lookback_years: int, sheet_names_list):
    """Loads and concatenates one or several universe sheets, then applies a
    single combined cleaning pass (resample/lookback/dropna) so tickers common
    to several sheets aren't double-filtered inconsistently."""
    all_prices = []
    gics1_map, gics2_map = {}, {}
    duplicate_tickers = set()
    for sn in sheet_names_list:
        prices_raw, g1, g2 = eq_parse_sheet(file_bytes, sn)
        overlap = set(prices_raw.columns) & set(gics1_map.keys())
        duplicate_tickers |= overlap
        all_prices.append(prices_raw)
        for t, v in g1.items():
            gics1_map.setdefault(t, v)
        for t, v in g2.items():
            gics2_map.setdefault(t, v)

    combined_raw = pd.concat(all_prices, axis=1)
    combined_raw = combined_raw.loc[:, ~combined_raw.columns.duplicated(keep="first")]

    prices = clean_prices(combined_raw, lookback_years)
    tickers = list(prices.columns)
    returns = prices.pct_change().dropna()
    return prices, returns, tickers, gics1_map, gics2_map, sorted(duplicate_tickers)


def render_equity_tab():
    st.subheader("Univers Equity")
    with st.expander("⚙️ Paramètres", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            uploaded_file = st.file_uploader("DATASTREAM.xlsx / .xlsm (Equity) — laisse vide pour utiliser le fichier par défaut", type=["xlsx", "xlsm"], key="eq_upload")
            selected_sheets = []
            file_bytes, file_source = None, None
            if uploaded_file is not None:
                file_bytes = uploaded_file.getvalue()
                file_source = uploaded_file.name
            else:
                try:
                    with open(EQ_DEFAULT_FILE_PATH, "rb") as f:
                        file_bytes = f.read()
                    file_source = EQ_DEFAULT_FILE_PATH
                except (FileNotFoundError, OSError):
                    file_bytes = None
            if file_bytes is not None:
                st.caption(f"📂 Fichier chargé : {file_source}")
                if uploaded_file is None:
                    try:
                        mtime = os.path.getmtime(EQ_DEFAULT_FILE_PATH)
                        last_update = datetime.datetime.fromtimestamp(mtime).strftime("%d/%m/%Y à %H:%M")
                        st.caption(f"🕒 Dernière mise à jour : {last_update}")
                    except OSError:
                        pass
                sheet_names = get_sheet_names(file_bytes)
                bench_sheet_actual = find_sheet_name(sheet_names, BENCH_SHEET_NAME)
                excluded_actual = {find_sheet_name(sheet_names, name) for name in EQ_EXCLUDED_SHEET_NAMES}
                excluded_actual.discard(None)
                universe_sheet_choices = [s for s in sheet_names if s not in excluded_actual]
                selected_sheets = st.multiselect(
                    "Univers (une ou plusieurs feuilles à concaténer)",
                    universe_sheet_choices,
                    default=universe_sheet_choices[:1],
                    key="eq_sheets",
                )
                selected_bench_name = None
                return_mode = "Absolu"
                if bench_sheet_actual is not None:
                    bench_prices_all = load_benchmark(file_bytes, bench_sheet_actual)
                    bench_names = list(bench_prices_all.columns)
                    if bench_names:
                        selected_bench_name = st.selectbox("Benchmark", bench_names, index=0, key="eq_bench_select")
                        return_mode = st.radio(
                            "Rendements utilisés pour l'optimisation",
                            ["Absolu (R(ticker))", "Excédentaire (R(ticker) − R(bench))"],
                            index=0, key="eq_return_mode", horizontal=True,
                        )
                    else:
                        st.caption(f"⚠️ Feuille '{bench_sheet_actual}' trouvée mais aucun indice détecté en ligne {BENCH_ROW_NAME + 1}.")
            risk_free = st.number_input("Risk-free rate (annual)", 0.0, 0.20, 0.02, 0.00001, format="%.5f", key="eq_rf")
            lookback_years = st.number_input("Lookback window (years)", 1, 15, 5, 1, key="eq_lookback")
        with c2:
            vt_min_pct, vt_max_pct = st.slider("Volatility target range (souhaitée) %", 1, 50, (1, 25), 1, key="eq_vt_range_pct")
            vt_min, vt_max = vt_min_pct / 100, vt_max_pct / 100
            n_targets = st.slider("Number of vol targets", 3, 15, 9, key="eq_n_targets")
            default_max_weight = st.slider(
                "Max weight per ticker (%) — valeur par défaut", 1, 100, 20, 1, key="eq_default_max_w",
                help="Pré-remplit la colonne Max % du tableau ci-dessous pour tous les tickers. "
                     "Tu peux ensuite ajuster ticker par ticker dans le tableau.",
            )
        c3, c4 = st.columns(2)
        with c3:
            n_sim = st.number_input("Random portfolios (chart cloud)", 500, 20000, 5000, 500, key="eq_nsim")
        with c4:
            n_starts = st.number_input("Optimizer random restarts", 5, 200, 20, 5, key="eq_nstarts")

    if file_bytes is None:
        st.info(
            f"👈 Upload un DATASTREAM.xlsx, ou dépose-le au chemin par défaut configuré : "
            f"`{EQ_DEFAULT_FILE_PATH}`"
        )
        return

    if not selected_sheets:
        st.info("👈 Sélectionne au moins une feuille d'univers dans les paramètres.")
        return

    with st.spinner("Parsing DATASTREAM.xlsx..."):
        prices, returns, tickers, gics1_map, gics2_map, duplicate_tickers = eq_load_and_clean_multi(
            file_bytes, lookback_years, selected_sheets
        )

    if len(selected_sheets) > 1:
        st.caption(f"Univers concaténé : {', '.join(selected_sheets)}")
        if duplicate_tickers:
            st.caption(
                f"ℹ️ {len(duplicate_tickers)} ticker(s) présent(s) dans plusieurs feuilles sélectionnées "
                f"(classification GICS conservée de la première feuille où trouvés) : "
                + ", ".join(duplicate_tickers[:20]) + (" ..." if len(duplicate_tickers) > 20 else "")
            )

    # ── Excess returns vs benchmark (R(t) - R(b)) ────────────
    use_excess = False
    bench_available = False
    bench_name = selected_bench_name
    raw_returns = returns
    bench_returns_aligned = None
    excess_returns = None
    if bench_name is not None:
        bench_prices = bench_prices_all[bench_name]
        bench_weekly = bench_prices.resample("W").last()
        bench_returns = bench_weekly.pct_change().dropna()

        common_idx = returns.index.intersection(bench_returns.index)
        if len(common_idx) < max(10, int(0.5 * len(returns))):
            st.warning(
                f"⚠️ Recouvrement de dates insuffisant avec '{bench_name}' "
                f"({len(common_idx)} dates communes sur {len(returns)}) — rendements excédentaires "
                f"indisponibles, calcul effectué sur rendements absolus."
            )
        else:
            bench_returns_aligned = bench_returns.loc[common_idx]
            excess_returns = raw_returns.loc[common_idx].sub(bench_returns_aligned, axis=0)
            bench_available = True
            if return_mode.startswith("Excédentaire"):
                returns = excess_returns
                use_excess = True
    elif find_sheet_name(sheet_names, BENCH_SHEET_NAME) is None:
        st.caption(
            f"ℹ️ Ajoute une feuille '{BENCH_SHEET_NAME}' au classeur pour pouvoir basculer sur les "
            f"rendements excédentaires vs un indice."
        )

    if use_excess:
        st.info(
            f"ℹ️ Frontière calculée sur les **rendements excédentaires** vs **{bench_name}** "
            f"(R(ticker) − R(bench)). Les métriques 'Return' et 'Volatility' ci-dessous représentent donc "
            f"le **rendement actif** et la **tracking error**, pas des valeurs absolues."
        )
    elif bench_available:
        st.caption(
            f"Frontière calculée sur les **rendements absolus** (le mode « Excédentaire » vs {bench_name} "
            f"est disponible — bascule via le sélecteur dans les paramètres si besoin)."
        )

    n_assets, n_weeks = len(tickers), len(prices)
    c1, c2, c3 = st.columns(3)
    c1.metric("Assets loaded", n_assets)
    c2.metric("Weekly observations", n_weeks)
    c3.metric("Date range", f"{prices.index[0].date()} → {prices.index[-1].date()}")

    with st.expander("📋 Universe detail", expanded=False):
        detail_df = pd.DataFrame({
            "Ticker": tickers,
            "GICS 1": [gics1_map.get(t, UNCLASSIFIED) for t in tickers],
            "GICS 2": [gics2_map.get(t, UNCLASSIFIED) for t in tickers],
        })
        st.dataframe(detail_df, use_container_width=True, hide_index=True)

    if bench_available:
        tab_labels = ["Prices", "Weekly returns", "Bench return", "Ticker − Bench"]
        tabs = st.tabs(tab_labels)
        tab_prices, tab_returns, tab_bench, tab_excess = tabs
    else:
        tab_prices, tab_returns = st.tabs(["Prices", "Weekly returns"])

    with tab_prices:
        st.dataframe(prices.tail(20), use_container_width=True, height=350)
        st.download_button("Download prices.csv", prices.to_csv().encode(), "prices.csv", "text/csv", key="eq_dl_prices")
    with tab_returns:
        st.dataframe(raw_returns.tail(20), use_container_width=True, height=350)
        st.download_button("Download returns.csv", raw_returns.to_csv().encode(), "returns.csv", "text/csv", key="eq_dl_returns")
    if bench_available:
        with tab_bench:
            bench_df = bench_returns_aligned.rename(bench_name).to_frame()
            st.dataframe(bench_df.tail(20), use_container_width=True, height=350)
            st.download_button(
                "Download bench_returns.csv", bench_df.to_csv().encode(), "bench_returns.csv", "text/csv",
                key="eq_dl_bench",
            )
        with tab_excess:
            st.dataframe(excess_returns.tail(20), use_container_width=True, height=350)
            st.download_button(
                "Download excess_returns.csv", excess_returns.to_csv().encode(), "excess_returns.csv", "text/csv",
                key="eq_dl_excess",
            )

    # ── Editable per-instrument constraints table ────────────
    st.subheader("📝 Contraintes par instrument")
    editor_df = pd.DataFrame({
        "Ticker": tickers,
        "GICS 1": [gics1_map.get(t, UNCLASSIFIED) for t in tickers],
        "GICS 2": [gics2_map.get(t, UNCLASSIFIED) for t in tickers],
        "Min %": 0.0,
        "Max %": float(default_max_weight),
    })
    st.caption(f"Max % pré-rempli à {default_max_weight}% pour tous les tickers — modifie une cellule pour l'ajuster individuellement.")
    edited_df = st.data_editor(
        editor_df,
        column_config={
            "Ticker": st.column_config.TextColumn(disabled=True),
            "GICS 1": st.column_config.TextColumn(disabled=True),
            "GICS 2": st.column_config.TextColumn(disabled=True),
            "Min %": st.column_config.NumberColumn(min_value=0.0, max_value=100.0, step=0.01, format="%.2f"),
            "Max %": st.column_config.NumberColumn(min_value=0.0, max_value=100.0, step=0.01, format="%.2f"),
        },
        hide_index=True, use_container_width=True, key="eq_editor",
        height=min(500, 35 * len(editor_df) + 40),
    )

    invalid_bounds = edited_df[edited_df["Min %"] > edited_df["Max %"]]
    blocked = False
    if len(invalid_bounds) > 0:
        blocked = True
        st.error("🚫 Min % > Max % pour : " + ", ".join(invalid_bounds["Ticker"].tolist()))

    # ── Sector-level constraints (GICS 1) — editable table ──
    st.subheader("📊 Contraintes par secteur (GICS 1)")
    gics1_values = sorted(set(edited_df["GICS 1"]))

    bench_weights_for_sel = None
    bench_poids_sheet_actual = find_sheet_name(sheet_names, BENCH_POIDS_SHEET_NAME)
    if bench_name is not None and bench_poids_sheet_actual is not None:
        bench_weights_all = load_bench_weights(file_bytes, bench_poids_sheet_actual)
        if bench_name in bench_weights_all.columns:
            bench_weights_for_sel = bench_weights_all[bench_name]
        else:
            st.caption(
                f"⚠️ Aucune colonne '{bench_name}' trouvée dans '{bench_poids_sheet_actual}' — bornes secteur "
                f"non préremplies depuis le bench."
            )

    if bench_weights_for_sel is not None:
        gics1_tolerance = st.slider(
            "Écart autorisé autour du poids du bench (± points de %)", 0, 50, 5, 1, key="eq_gics1_tolerance",
            help="Préremplit Min % = poids du bench − x, Max % = poids du bench + x, pour chaque secteur. "
                 "Modifiable ensuite ligne par ligne dans le tableau.",
        )
        st.caption(
            f"Bornes préremplies autour du poids de {bench_name} dans '{BENCH_POIDS_SHEET_NAME}' "
            f"(± {gics1_tolerance} pts). Secteurs absents du bench restent non bornés (0–100%)."
        )

    def _gics1_default_bounds(sec):
        if bench_weights_for_sel is not None and sec in bench_weights_for_sel.index and pd.notna(bench_weights_for_sel[sec]):
            w_pct = float(bench_weights_for_sel[sec]) * 100
            lo = max(0.0, w_pct - gics1_tolerance)
            hi = min(100.0, w_pct + gics1_tolerance)
            return round(lo, 1), round(hi, 1)
        return 0.0, 100.0

    gics1_default_bounds = [_gics1_default_bounds(sec) for sec in gics1_values]
    gics1_editor_df = pd.DataFrame({
        "GICS 1": gics1_values,
        "Min %": [b[0] for b in gics1_default_bounds],
        "Max %": [b[1] for b in gics1_default_bounds],
    })
    edited_gics1 = st.data_editor(
        gics1_editor_df,
        column_config={
            "GICS 1": st.column_config.TextColumn(disabled=True),
            "Min %": st.column_config.NumberColumn(min_value=0.0, max_value=100.0, step=0.01, format="%.2f"),
            "Max %": st.column_config.NumberColumn(min_value=0.0, max_value=100.0, step=0.01, format="%.2f"),
        },
        hide_index=True, use_container_width=True, key="eq_gics1_editor",
        height=min(400, 35 * len(gics1_editor_df) + 40),
    )
    invalid_gics1 = edited_gics1[edited_gics1["Min %"] > edited_gics1["Max %"]]
    if len(invalid_gics1) > 0:
        blocked = True
        st.error("🚫 Min % > Max % pour le(s) secteur(s) : " + ", ".join(invalid_gics1["GICS 1"].tolist()))
    gics1_bounds = {row["GICS 1"]: (row["Min %"] / 100, row["Max %"] / 100) for _, row in edited_gics1.iterrows()}

    # ── Sub-sector constraints (GICS 2) — prefilled from the GICS_2 bench weights sheet ──
    bench_gics2_for_sel = None
    bench_gics2_sheet_actual = None
    for _cand in BENCH_POIDS_GICS2_SHEET_NAMES:
        bench_gics2_sheet_actual = find_sheet_name(sheet_names, _cand)
        if bench_gics2_sheet_actual is not None:
            break
    if bench_name is not None and bench_gics2_sheet_actual is not None:
        bench_gics2_all = load_bench_weights(file_bytes, bench_gics2_sheet_actual, auto_header=True)
        if bench_name in bench_gics2_all.columns:
            bench_gics2_for_sel = bench_gics2_all[bench_name]

    with st.expander(
        "📊 Contraintes par sous-secteur (GICS 2)" + ("" if bench_gics2_for_sel is not None else " — optionnel"),
        expanded=bench_gics2_for_sel is not None,
    ):
        gics2_values = sorted(set(edited_df["GICS 2"]))

        if bench_gics2_for_sel is not None:
            gics2_tolerance = st.slider(
                "Écart autorisé autour du poids du bench GICS 2 (± points de %)", 0, 50, 2, 1,
                key="eq_gics2_tolerance",
                help="Préremplit Min % = poids du bench − x, Max % = poids du bench + x, pour chaque "
                     "sous-secteur GICS 2. Modifiable ensuite ligne par ligne dans le tableau.",
            )
            st.caption(
                f"Bornes préremplies autour du poids de {bench_name} dans '{bench_gics2_sheet_actual}' "
                f"(± {gics2_tolerance} pts). Sous-secteurs absents du bench restent non bornés (0–100%)."
            )
        elif bench_name is not None and bench_gics2_sheet_actual is not None:
            st.caption(
                f"⚠️ Aucune colonne '{bench_name}' trouvée dans '{bench_gics2_sheet_actual}' — bornes "
                f"sous-secteur non préremplies depuis le bench."
            )

        def _gics2_default_bounds(sec):
            if bench_gics2_for_sel is not None and sec in bench_gics2_for_sel.index and pd.notna(bench_gics2_for_sel[sec]):
                w_pct = float(bench_gics2_for_sel[sec]) * 100
                lo = max(0.0, w_pct - gics2_tolerance)
                hi = min(100.0, w_pct + gics2_tolerance)
                return round(lo, 1), round(hi, 1)
            return 0.0, 100.0

        gics2_default_bounds = [_gics2_default_bounds(sec) for sec in gics2_values]
        gics2_editor_df = pd.DataFrame({
            "GICS 2": gics2_values,
            "Min %": [b[0] for b in gics2_default_bounds],
            "Max %": [b[1] for b in gics2_default_bounds],
        })
        edited_gics2 = st.data_editor(
            gics2_editor_df,
            column_config={
                "GICS 2": st.column_config.TextColumn(disabled=True),
                "Min %": st.column_config.NumberColumn(min_value=0.0, max_value=100.0, step=0.01, format="%.2f"),
                "Max %": st.column_config.NumberColumn(min_value=0.0, max_value=100.0, step=0.01, format="%.2f"),
            },
            hide_index=True, use_container_width=True, key="eq_gics2_editor",
            height=min(400, 35 * len(gics2_editor_df) + 40),
        )
        invalid_gics2 = edited_gics2[edited_gics2["Min %"] > edited_gics2["Max %"]]
        if len(invalid_gics2) > 0:
            blocked = True
            st.error("🚫 Min % > Max % pour le(s) sous-secteur(s) : " + ", ".join(invalid_gics2["GICS 2"].tolist()))
        gics2_bounds = {row["GICS 2"]: (row["Min %"] / 100, row["Max %"] / 100) for _, row in edited_gics2.iterrows()}

    run_btn = st.button("🚀 Run optimisation (Equity)", type="primary", key="eq_run", disabled=blocked)
    if blocked:
        return
    if not run_btn:
        st.info("Ajuste les paramètres puis clique sur **Run optimisation**.")
        return

    gics1_of = dict(zip(edited_df["Ticker"], edited_df["GICS 1"]))
    gics2_of = dict(zip(edited_df["Ticker"], edited_df["GICS 2"]))
    bounds = [
        (float(row["Min %"]) / 100, float(row["Max %"]) / 100)
        for _, row in edited_df.set_index("Ticker").loc[tickers].reset_index().iterrows()
    ]

    mu_arr = (returns.mean() * WEEKS_YEAR).values
    Sigma_arr = (returns.cov() * WEEKS_YEAR).values
    N = n_assets

    extra_constraints = (
        build_class_constraints(tickers, gics1_of, gics1_bounds)
        + build_class_constraints(tickers, gics2_of, gics2_bounds)
    )

    df_results = run_frontier(vt_min, vt_max, n_targets, mu_arr, Sigma_arr, N, n_starts, bounds, tickers, risk_free, extra_constraints)
    if df_results is None:
        st.caption(
            "Astuce : vérifie que la somme des Max % d'un secteur reste ≥ à son Min % de secteur, "
            "et inversement, et que les bornes par instrument laissent assez de marge."
        )
        return
    render_results(df_results, tickers, mu_arr, Sigma_arr, risk_free, n_sim, N, key_prefix="eq", bounds=bounds, class_of=gics1_of)


# ═════════════════════════════════════════════════════════════
# TAB 2 — MULTI-ASSET  (fixed DATASTREAM row layout + Asset Type row)
# ═════════════════════════════════════════════════════════════
# Row 0 -> EMPTY
# Row 1 -> Asset Type (bond, equity, ...)
# Row 2 -> Name
# Row 3 -> ISIN
# Row 4+ -> Prices
# Col A -> EMPTY, Col B -> dates, Col C+ -> instruments
MA_ROW_TYPE = 1
MA_ROW_NAME = 2
MA_ROW_ISIN = 3
MA_ROW_PRICES = 4


@st.cache_data(show_spinner=False)
def ma_load_and_clean(file_bytes: bytes, sheet_name=0):
    raw = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name, header=None, index_col=None)

    type_row = raw.iloc[MA_ROW_TYPE, COL_DATA_START:]
    name_row = raw.iloc[MA_ROW_NAME, COL_DATA_START:]
    isin_row = raw.iloc[MA_ROW_ISIN, COL_DATA_START:]

    tickers_raw, name_map, type_map = [], {}, {}
    for typ, name, isin in zip(type_row.values, name_row.values, isin_row.values):
        type_str = str(typ).strip()
        name_str = str(name).strip()
        isin_str = str(isin).strip()
        label = isin_str if isin_str not in ("", "nan", "NaN") else name_str
        if not label or label in ("nan", "NaN"):
            continue
        tickers_raw.append(label)
        name_map[label] = name_str if name_str not in ("", "nan", "NaN") else label
        type_map[label] = type_str if type_str not in ("", "nan", "NaN") else ""

    price_block = raw.iloc[MA_ROW_PRICES:, :].copy()
    dates = pd.to_datetime(price_block.iloc[:, COL_DATE], errors="coerce")
    prices = price_block.iloc[:, COL_DATA_START:COL_DATA_START + len(tickers_raw)].copy()
    prices.columns = tickers_raw
    prices.index = dates
    prices = prices.loc[:, ~prices.columns.duplicated(keep="first")]

    prices = prices[prices.index.notna()]
    prices = prices.apply(pd.to_numeric, errors="coerce")
    prices = prices.dropna(axis=1, how="all")

    return prices, name_map, type_map


def clean_prices(prices: pd.DataFrame, lookback_years: int) -> pd.DataFrame:
    prices = prices.dropna(axis=1, how="all")
    prices = prices.resample("W").last()
    cutoff = prices.index.max() - pd.DateOffset(years=lookback_years)
    prices = prices[prices.index >= cutoff]
    prices = prices[prices.isna().mean(axis=1) < 0.5]
    prices = prices.ffill().bfill()
    prices = prices.dropna(axis=1, thresh=int(len(prices) * 0.8))
    return prices


def run_frontier(vt_min, vt_max, n_targets, mu_arr, Sigma_arr, N, n_starts, bounds, tickers, risk_free, extra_constraints=None):
    with st.spinner("Computing minimum variance portfolio..."):
        w_minvol, min_vol = get_min_vol_portfolio(mu_arr, Sigma_arr, N, bounds, extra_constraints)
    if w_minvol is None:
        st.error("❌ Contraintes infaisables (bornes par actif / par classe incompatibles). Assouplis les bornes.")
        return None

    with st.spinner("Computing maximum return portfolio (vol ceiling)..."):
        w_maxret, max_vol = get_max_return_portfolio(mu_arr, Sigma_arr, N, bounds, extra_constraints)
    if w_maxret is None:
        w_maxret, max_vol = w_minvol, min_vol

    c1, c2 = st.columns(2)
    c1.info(f"ℹ️ Volatilité minimale atteignable : **{min_vol*100:.2f}%**")
    c2.info(f"ℹ️ Volatilité maximale atteignable (compte tenu des bornes de poids) : **{max_vol*100:.2f}%**")

    # ── Intersect the desired [vt_min, vt_max] range with what's physically
    # attainable [min_vol, max_vol] — never widen beyond either side.
    if vt_max < min_vol - 1e-9 or vt_min > max_vol + 1e-9:
        clamp = min_vol if vt_max < min_vol else max_vol
        st.warning(
            f"⚠️ La plage de vol demandée ({vt_min*100:.0f}%–{vt_max*100:.0f}%) ne recoupe pas la plage "
            f"atteignable ({min_vol*100:.2f}%–{max_vol*100:.2f}%). Tous les portefeuilles sont ramenés à "
            f"{clamp*100:.2f}%."
        )
        effective_min = effective_max = clamp
    else:
        effective_min = max(vt_min, min_vol)
        effective_max = min(vt_max, max_vol)
        if vt_max < max_vol - 1e-9:
            st.caption(f"Plafond demandé ({vt_max*100:.0f}%) déjà en dessous du plafond atteignable — utilisé tel quel.")
        elif max_vol < vt_max - 1e-9:
            st.caption(f"Plafond demandé ({vt_max*100:.0f}%) au-dessus du plafond atteignable — ramené à {max_vol*100:.2f}%.")
        if min_vol < vt_min - 1e-9:
            st.caption(f"Plancher demandé ({vt_min*100:.0f}%) au-dessus du plancher atteignable — utilisé tel quel.")
        elif vt_min < min_vol - 1e-9:
            st.caption(f"Plancher demandé ({vt_min*100:.0f}%) en dessous du plancher atteignable — remonté à {min_vol*100:.2f}%.")

    if effective_max <= effective_min + 1e-6:
        sorted_targets = [effective_min] * n_targets
    else:
        sorted_targets = list(np.linspace(effective_min, effective_max, n_targets))

    st.caption(
        f"Les {n_targets} cibles de volatilité sont réparties uniformément entre "
        f"{effective_min*100:.2f}% et {effective_max*100:.2f}% — l'intersection entre ta plage souhaitée et "
        f"la plage physiquement atteignable."
    )

    first_is_minvol = effective_min <= min_vol + 1e-9
    last_is_maxret = effective_max >= max_vol - 1e-9

    progress = st.progress(0.0, text="Optimising efficient frontier...")
    records = []
    for i, vt in enumerate(sorted_targets):
        progress.progress(i / len(sorted_targets), text=f"Calcul en cours — vol target {vt*100:.1f}%...")
        if i == 0 and first_is_minvol:
            w, flag = w_minvol, "min_vol"
            ret, vol = portfolio_stats(w, mu_arr, Sigma_arr)
        elif i == len(sorted_targets) - 1 and last_is_maxret:
            w, flag = w_maxret, "max_return"
            ret, vol = portfolio_stats(w, mu_arr, Sigma_arr)
        else:
            result = max_return_given_vol(
                vt, mu_arr, Sigma_arr, N, int(n_starts), bounds,
                extra_constraints=extra_constraints,
                w_minvol=w_minvol, min_vol=min_vol,
                w_maxret=w_maxret, max_vol=max_vol,
            )
            w, ret, vol, flag = result["w"], result["ret"], result["vol"], result["feasible"]
        sharpe = (ret - risk_free) / vol if vol > 1e-8 else np.nan
        row = {
            "vol_target": round(vt, 3), "realized_vol": round(vol, 6),
            "expected_return": round(ret, 6),
            "sharpe_ratio": round(float(sharpe), 4) if not np.isnan(sharpe) else np.nan,
            "feasible": flag,
        }
        for ticker, weight in zip(tickers, w):
            row[ticker] = round(float(weight), 6)
        records.append(row)
        progress.progress((i + 1) / len(sorted_targets), text=f"Vol target {vt*100:.0f}% — {flag}")
    progress.empty()
    df_results = pd.DataFrame(records)
    df_results["feasible"] = df_results["feasible"].astype(str)
    return df_results


def render_results(
    df_results,
    tickers,
    mu_arr,
    Sigma_arr,
    risk_free,
    n_sim,
    N,
    key_prefix,
    bounds=None,
    class_of=None,
    name_map=None,
):
    """
    Affiche les résultats de l'optimisation.

    Parameters
    ----------
    name_map : dict, optional
        Mapping {ticker/ISIN: nom du produit}.
        Utilisé principalement dans la partie Multi-Asset pour afficher
        le nom du produit dans le tableau et dans le bar chart.
    """
    best_sharpe_idx = df_results["sharpe_ratio"].idxmax()
    
    st.success(f"✅ Optimisation complete — {len(df_results)} portfolios built.")

    st.subheader("Results table")

    display_cols = [
        "vol_target",
        "realized_vol",
        "expected_return",
        "sharpe_ratio",
        "feasible",
    ]

    display_df = df_results[display_cols].copy()
    display_df[""] = display_df.index.map(
        lambda i: "⭐ Best Sharpe" if i == best_sharpe_idx else ""
    )

    st.dataframe(
        display_df.style.format({
            "vol_target": "{:.1%}",
            "realized_vol": "{:.2%}",
            "expected_return": "{:.2%}",
            "sharpe_ratio": "{:.3f}",
        }),
        use_container_width=True,
        hide_index=True,
        height=380,
    )

    st.download_button(
        "Download efficient_frontier_baskets.csv",
        df_results.to_csv(index=False).encode(),
        "efficient_frontier_baskets.csv",
        "text/csv",
        key=f"{key_prefix}_dl_results",
    )

    st.subheader("Basket weights by volatility target")

    def get_instrument_name(ticker):
        """
        Retourne le nom lisible de l'instrument si disponible.
        Sinon, utilise le ticker/ISIN comme fallback.
        """
        if not name_map:
            return str(ticker)

        instrument_name = name_map.get(ticker, ticker)

        if pd.isna(instrument_name):
            return str(ticker)

        instrument_name = str(instrument_name).strip()

        if instrument_name in ("", "nan", "NaN", "None"):
            return str(ticker)

        return instrument_name

    def format_chart_label(ticker, max_length=30):
        """
        Construit un libellé lisible pour le bar chart.
        Le nom est légèrement tronqué uniquement s'il est très long.
        """
        label = get_instrument_name(ticker)

        if len(label) > max_length:
            return label[:max_length - 3] + "..."

        return label

    for i, vt in enumerate(df_results["vol_target"]):
        row_w = df_results.iloc[i][tickers].astype(float)

        # Affichage uniquement des positions supérieures à 0,10 %
        top = row_w[row_w > 0.001].sort_values(ascending=False)

        sharpe_val = df_results.iloc[i]["sharpe_ratio"]

        with st.expander(
            f"Vol target {vt * 100:.2f}% — Sharpe {sharpe_val:.3f}"
        ):
            col_chart, col_table = st.columns([2, 1])

            # ─────────────────────────────────────────────────
            # Allocation bar chart
            # Multi-Asset : nom du produit à la place de l'ISIN
            # Equity      : ticker conservé
            # ─────────────────────────────────────────────────
            with col_chart:
                chart_labels = [
                    format_chart_label(ticker)
                    for ticker in top.index
                ]

                # Largeur adaptée au nombre de positions affichées
                chart_width = max(10, min(24, len(top) * 1.15))

                fig_w, ax_w = plt.subplots(
                    figsize=(chart_width, 5)
                )

                bars = ax_w.bar(
                    range(len(top)),
                    top.values * 100,
                    color="#1565C0",
                    alpha=0.85,
                    edgecolor="white",
                )

                ax_w.set_xticks(range(len(top)))
                ax_w.set_xticklabels(
                    chart_labels,
                    rotation=45,
                    ha="right",
                    fontsize=8,
                )

                ax_w.set_ylabel("Weight (%)")

                ax_w.set_title(
                    f"Allocation — Vol target {vt * 100:.2f}%",
                    fontsize=11,
                    fontweight="bold",
                )

                ax_w.yaxis.set_major_formatter(
                    plt.FuncFormatter(lambda y, _: f"{y:.1f}%")
                )

                ax_w.grid(
                    True,
                    axis="y",
                    linestyle="--",
                    alpha=0.5,
                )

                # Affichage du poids au-dessus de chaque barre
                for bar, weight in zip(bars, top.values):
                    ax_w.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        f"{weight:.1%}",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        color="#1a1a2e",
                    )

                plt.tight_layout()

                st.pyplot(
                    fig_w,
                    use_container_width=True,
                )

                plt.close(fig_w)

            # ─────────────────────────────────────────────────
            # Allocation table
            # Multi-Asset : Ticker/ISIN + Name + Asset Class
            # Equity      : Ticker + GICS 1 + Weight
            # ─────────────────────────────────────────────────
            with col_table:
                tbl = top.rename_axis("Ticker").reset_index(name="Weight")

                if name_map:
                    tbl.insert(
                        1,
                        "Name",
                        tbl["Ticker"].map(get_instrument_name),
                    )

                if class_of:
                    class_position = 2 if name_map else 1

                    tbl.insert(
                        class_position,
                        "Asset Class",
                        tbl["Ticker"].map(
                            lambda ticker: class_of.get(
                                ticker,
                                UNCLASSIFIED,
                            )
                        ),
                    )

                tbl["Weight"] = tbl["Weight"].map("{:.2%}".format)

                st.dataframe(
                    tbl,
                    use_container_width=True,
                    hide_index=True,
                    height=min(
                        400,
                        35 * len(top) + 40,
                    ),
                )

            # ─────────────────────────────────────────────────
            # Allocation agrégée par classe
            # ─────────────────────────────────────────────────
            if class_of:
                class_labels = [
                    class_of.get(ticker, UNCLASSIFIED)
                    for ticker in row_w.index
                ]

                class_alloc = row_w.groupby(class_labels).sum()
                class_alloc = class_alloc.sort_values(ascending=False)

                st.caption("Allocation par classe d'actifs")

                fig_cls, ax_cls = plt.subplots(figsize=(6, 3))
                ax_cls.bar(
                    class_alloc.index,
                    class_alloc.values * 100,
                    color="#1565C0",
                    alpha=0.85,
                    edgecolor="white",
                )
                ax_cls.set_ylabel("Weight (%)")
                ax_cls.yaxis.set_major_formatter(
                    plt.FuncFormatter(lambda y, _: f"{y:.1f}%")
                )
                for i, (cls_name, val) in enumerate(class_alloc.items()):
                    ax_cls.text(
                        i, val * 100,
                        f"{val * 100:.1f}%",
                        ha="center", va="bottom",
                        fontsize=8, fontweight="bold", color="#1a1a2e",
                    )
                ax_cls.set_xticks(range(len(class_alloc)))
                ax_cls.set_xticklabels(class_alloc.index, rotation=30, ha="right", fontsize=9)
                ax_cls.grid(True, axis="y", linestyle="--", alpha=0.5)
                plt.tight_layout()
                st.pyplot(fig_cls, use_container_width=True)
                plt.close(fig_cls)


    # ─────────────────────────────────────────────────────────
    # Efficient frontier chart
    # ─────────────────────────────────────────────────────────
    st.subheader("Efficient frontier chart")

    with st.spinner("Rendering chart..."):
        fig = build_chart(
            df_results,
            tickers,
            mu_arr,
            Sigma_arr,
            risk_free,
            n_sim,
            N,
            bounds=bounds,
        )

        st.pyplot(
            fig,
            use_container_width=True,
        )

        buf = io.BytesIO()

        fig.savefig(
            buf,
            format="png",
            dpi=150,
            bbox_inches="tight",
            facecolor="white",
        )

        buf.seek(0)
        plt.close(fig)

        st.download_button(
            "Download efficient_frontier.png",
            buf.getvalue(),
            "efficient_frontier.png",
            "image/png",
            key=f"{key_prefix}_dl_png",
        )



def _apply_ma_config_load():
    """Callback for the 'Charger' button — runs BEFORE the script body re-renders,
    so it's safe to write into session_state keys already bound to widgets."""
    selected = st.session_state.get("ma_config_select")
    if not selected or selected == "— Sélectionner —":
        return
    all_configs = load_all_configs()
    cfg = all_configs.get("multi_asset", {}).get(selected)
    if not cfg:
        return
    st.session_state["ma_rf"] = cfg.get("risk_free", 0.02)
    st.session_state["ma_lookback"] = cfg.get("lookback_years", 5)
    st.session_state["ma_vt_range"] = tuple(cfg.get("vt_range", (0.01, 0.25)))
    st.session_state["ma_n_targets"] = cfg.get("n_targets", 9)
    st.session_state["ma_nsim"] = cfg.get("n_sim", 5000)
    st.session_state["ma_nstarts"] = cfg.get("n_starts", 20)
    for cls, cls_bounds in cfg.get("class_bounds", {}).items():
        st.session_state[f"ma_class_{cls}"] = tuple(int(x) for x in cls_bounds)
    st.session_state["ma_loaded_instrument_bounds"] = cfg.get("instrument_bounds", {})
    st.session_state["ma_editor_version"] = st.session_state.get("ma_editor_version", 0) + 1
    st.session_state["_ma_load_message"] = f"Configuration « {selected} » chargée."


def render_multi_asset_tab():
    st.subheader("Univers Multi-Asset")
    with st.expander("⚙️ Paramètres généraux", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            uploaded_file = st.file_uploader("Fichier maître (Bloomberg Excel .xlsx / .xlsm)", type=["xlsx", "xlsm"], key="ma_upload")
            sheet_name = 0
            if uploaded_file is not None:
                sheet_names = get_sheet_names(uploaded_file.getvalue())
                if len(sheet_names) > 1:
                    sheet_name = st.selectbox("Feuille Excel", sheet_names, key="ma_sheet")
                else:
                    sheet_name = sheet_names[0]
            risk_free = st.number_input("Risk-free rate (annual)", 0.0, 0.20, 0.02, 0.00001, format="%.5f", key="ma_rf")
            lookback_years = st.number_input("Lookback window (years)", 1, 15, 5, 1, key="ma_lookback")
        with c2:
            vt_min_pct, vt_max_pct = st.slider("Volatility target range (souhaitée) %", 1, 50, (1, 25), 1, key="ma_vt_range_pct")
            vt_min, vt_max = vt_min_pct / 100, vt_max_pct / 100
            n_targets = st.slider("Number of vol targets", 3, 15, 9, key="ma_n_targets")
            n_sim = st.number_input("Random portfolios (chart cloud)", 500, 20000, 5000, 500, key="ma_nsim")
            n_starts = st.number_input("Optimizer random restarts", 5, 200, 20, 5, key="ma_nstarts")

    if st.session_state.get("_ma_load_message"):
        st.success(st.session_state.pop("_ma_load_message"))

    with st.expander("💾 Configurations du fonds (Multi-Asset)", expanded=False):
        all_configs = load_all_configs()
        ma_configs = all_configs["multi_asset"]
        cc1, cc2 = st.columns(2)
        with cc1:
            config_names = ["— Sélectionner —"] + sorted(ma_configs.keys())
            selected_config = st.selectbox("Configuration enregistrée", config_names, key="ma_config_select")
            lc, dc = st.columns(2)
            lc.button("📂 Charger", key="ma_config_load", disabled=(selected_config == "— Sélectionner —"),
                      use_container_width=True, on_click=_apply_ma_config_load)
            delete_clicked = dc.button("🗑️ Supprimer", key="ma_config_delete", disabled=(selected_config == "— Sélectionner —"), use_container_width=True)
        with cc2:
            st.text_input("Nom pour l'enregistrement", key="ma_config_name")
            st.caption("Le bouton d'enregistrement apparaît plus bas, une fois l'univers et les contraintes définis.")

        if delete_clicked and selected_config in ma_configs:
            del ma_configs[selected_config]
            all_configs["multi_asset"] = ma_configs
            save_all_configs(all_configs)
            st.success(f"Configuration « {selected_config} » supprimée.")
            st.rerun()

    if uploaded_file is None:
        st.info("👈 Upload le fichier maître Bloomberg (Excel) pour commencer.")
        return

    file_bytes = uploaded_file.getvalue()
    with st.spinner("Lecture du fichier maître..."):
        try:
            prices_full, name_map, type_map = ma_load_and_clean(file_bytes, sheet_name)
        except (ValueError, IndexError) as e:
            st.error(f"Erreur de lecture du fichier : {e}")
            return

    prices = clean_prices(prices_full, lookback_years)
    tickers = list(prices.columns)
    if len(tickers) < 2:
        st.warning("Moins de 2 actifs après nettoyage (historique insuffisant). Réduis la fenêtre de lookback.")
        return
    returns = prices.pct_change().dropna()

    # ── Asset class comes straight from the "Asset Type" row ────
    detected_classes = pd.Series({t: normalize_asset_class(type_map.get(t)) for t in tickers})
    name_col_vals = pd.Series({t: name_map.get(t, t) for t in tickers})

    # ── Editable constraints table ──────────────────────────
    st.subheader("📝 Classification & contraintes par instrument")
    loaded_instr = st.session_state.get("ma_loaded_instrument_bounds", {})
    editor_df = pd.DataFrame({
        "Ticker": tickers,
        "Name": [name_col_vals.get(t, t) for t in tickers],
        "Asset Class": [loaded_instr.get(t, {}).get("asset_class", detected_classes.get(t, UNCLASSIFIED)) for t in tickers],
        "Min %": [loaded_instr.get(t, {}).get("min_pct", 0.0) for t in tickers],
        "Max %": [loaded_instr.get(t, {}).get("max_pct", 20.0) for t in tickers],
    })
    if loaded_instr:
        st.caption("↳ Classification et bornes par instrument reprises de la configuration chargée (modifiable ci-dessous).")

    edited_df = st.data_editor(
        editor_df,
        column_config={
            "Ticker": st.column_config.TextColumn(disabled=True),
            "Name": st.column_config.TextColumn(disabled=True),
            "Asset Class": st.column_config.SelectboxColumn(options=ASSET_CLASSES + [UNCLASSIFIED], required=True),
            "Min %": st.column_config.NumberColumn(min_value=0.0, max_value=100.0, step=0.01, format="%.2f"),
            "Max %": st.column_config.NumberColumn(min_value=0.0, max_value=100.0, step=0.01, format="%.2f"),
        },
        hide_index=True, use_container_width=True,
        key=f"ma_editor_{st.session_state.get('ma_editor_version', 0)}",
        height=min(500, 35 * len(editor_df) + 40),
    )

    unclassified_rows = edited_df[edited_df["Asset Class"] == UNCLASSIFIED]
    invalid_bounds = edited_df[edited_df["Min %"] > edited_df["Max %"]]

    blocked = False
    if len(unclassified_rows) > 0:
        blocked = True
        st.error(
            f"🚫 {len(unclassified_rows)} instrument(s) non classifié(s) — reclassifie-les dans le tableau "
            f"avant de lancer l'optimisation : " + ", ".join(unclassified_rows["Ticker"].tolist())
        )
    if len(invalid_bounds) > 0:
        blocked = True
        st.error(
            "🚫 Min % > Max % pour : " + ", ".join(invalid_bounds["Ticker"].tolist())
        )

    # ── Class-level constraints ─────────────────────────────
    st.subheader("📊 Contraintes par classe d'actifs")
    present_classes = sorted(set(edited_df["Asset Class"]) & set(ASSET_CLASSES))
    if not present_classes:
        present_classes = ASSET_CLASSES

    class_bounds = {}
    cols = st.columns(min(3, len(present_classes)))
    for i, cls in enumerate(present_classes):
        with cols[i % len(cols)]:
            lo, hi = st.slider(cls, 0, 100, (0, 100), 1, key=f"ma_class_{cls}")
            class_bounds[cls] = (lo / 100, hi / 100)

    ma_config_name = st.session_state.get("ma_config_name", "")
    save_config_clicked = st.button(
        "💾 Enregistrer cette configuration", key="ma_config_save",
        disabled=not ma_config_name.strip(),
    )
    if save_config_clicked:
        name_to_save = ma_config_name.strip()
        instrument_bounds = {
            row["Ticker"]: {
                "asset_class": row["Asset Class"],
                "min_pct": float(row["Min %"]),
                "max_pct": float(row["Max %"]),
            }
            for _, row in edited_df.iterrows()
        }
        all_configs = load_all_configs()
        all_configs["multi_asset"][name_to_save] = {
            "risk_free": risk_free,
            "lookback_years": lookback_years,
            "n_targets": n_targets,
            "vt_range": [vt_min, vt_max],
            "n_sim": n_sim,
            "n_starts": n_starts,
            "class_bounds": {cls: [int(round(lo * 100)), int(round(hi * 100))] for cls, (lo, hi) in class_bounds.items()},
            "instrument_bounds": instrument_bounds,
        }
        save_all_configs(all_configs)
        st.success(f"Configuration « {name_to_save} » enregistrée.")
        st.rerun()

    run_btn = st.button("🚀 Lancer l'optimisation multi-asset", type="primary", key="ma_run", disabled=blocked)
    if blocked:
        return
    if not run_btn:
        st.info("Ajuste la classification, les bornes et les paramètres, puis clique sur **Lancer l'optimisation**.")
        return

    class_of = dict(zip(edited_df["Ticker"], edited_df["Asset Class"]))
    bounds = [
        (float(row["Min %"]) / 100, float(row["Max %"]) / 100)
        for _, row in edited_df.set_index("Ticker").loc[tickers].reset_index().iterrows()
    ]

    mu_arr = (returns.mean() * WEEKS_YEAR).values
    Sigma_arr = (returns.cov() * WEEKS_YEAR).values
    N = len(tickers)

    extra_constraints = build_class_constraints(tickers, class_of, class_bounds)

    df_results = run_frontier(vt_min, vt_max, n_targets, mu_arr, Sigma_arr, N, n_starts, bounds, tickers, risk_free, extra_constraints)
    if df_results is None:
        st.caption(
            "Astuce : vérifie que la somme des Max % d'une classe reste ≥ à son Min % de classe, "
            "et inversement, et que les bornes par instrument laissent assez de marge."
        )
        return
    render_results(
    df_results,
    tickers,
    mu_arr,
    Sigma_arr,
    risk_free,
    n_sim,
    N,
    key_prefix="ma",
    bounds=bounds,
    class_of=class_of,
    name_map=name_map,
)



# ─────────────────────────────────────────────────────────────
# 2. TABS
# ─────────────────────────────────────────────────────────────
tab_equity, tab_multi_asset = st.tabs(["📊 Equity", "🌍 Multi-Asset"])
with tab_equity:
    render_equity_tab()
with tab_multi_asset:
    render_multi_asset_tab()
