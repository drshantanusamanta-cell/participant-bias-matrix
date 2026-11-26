"""
core.py - data collection, validation and analysis engine for the
Participant F&O Bias Matrix (Dr Shantanu Samanta).

Sources (NSE archives, published after market close):
  * fao_participant_oi_DDMMYYYY.csv   - participant-wise open interest
  * fao_participant_vol_DDMMYYYY.csv  - participant-wise trading volume
  * fii_stats_DD-Mon-YYYY.xls         - FII derivatives statistics (Rs crore, index split)
  * NIFTY 50 close (^NSEI via yfinance) - for charts and backtest only
"""
from __future__ import annotations

import io
import os
import datetime as dt
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

BASE = "https://nsearchives.nseindia.com/content"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Accept": "*/*",
    "Referer": "https://www.nseindia.com/",
}
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CACHE_FILE = os.path.join(DATA_DIR, "history.csv")
MISSING_FILE = os.path.join(DATA_DIR, "missing_dates.txt")

CATS = ["Client", "DII", "FII", "Pro", "TOTAL"]
COLMAP = {
    "Future Index Long": "FIL", "Future Index Short": "FIS",
    "Future Stock Long": "FSL", "Future Stock Short": "FSS",
    "Option Index Call Long": "OICL", "Option Index Put Long": "OIPL",
    "Option Index Call Short": "OICS", "Option Index Put Short": "OIPS",
    "Option Stock Call Long": "OSCL", "Option Stock Put Long": "OSPL",
    "Option Stock Call Short": "OSCS", "Option Stock Put Short": "OSPS",
    "Total Long Contracts": "TL", "Total Short Contracts": "TS",
}
FII_ROWS = {"INDEX FUTURES": "IDXF", "NIFTY FUTURES": "NF",
            "INDEX OPTIONS": "IDXO", "NIFTY OPTIONS": "NO",
            "STOCK FUTURES": "STKF", "STOCK OPTIONS": "STKO"}
FII_COLS = ["buy_ct", "buy_cr", "sell_ct", "sell_cr", "oi_ct", "oi_cr"]


# ----------------------------------------------------------------- parsing
def parse_participant_csv(raw: bytes, prefix: str) -> dict:
    """Parse an OI or volume participant CSV -> flat dict {prefix_cat_col: value}."""
    df = pd.read_csv(io.BytesIO(raw), skiprows=1)
    df.columns = [str(c).strip() for c in df.columns]
    df["Client Type"] = df["Client Type"].astype(str).str.strip()
    out = {}
    for _, row in df.iterrows():
        cat = row["Client Type"]
        if cat not in CATS:
            continue
        for long_name, short in COLMAP.items():
            if long_name in df.columns:
                out[f"{prefix}{cat}_{short}"] = float(str(row[long_name]).replace(",", ""))
    return out


def parse_fii_stats(raw: bytes) -> dict:
    df = pd.read_excel(io.BytesIO(raw), header=None)
    out = {}
    for _, row in df.iterrows():
        label = str(row.iloc[0]).strip().upper()
        if label in FII_ROWS:
            key = FII_ROWS[label]
            for i, c in enumerate(FII_COLS, start=1):
                try:
                    out[f"FS_{key}_{c}"] = float(row.iloc[i])
                except (TypeError, ValueError):
                    out[f"FS_{key}_{c}"] = np.nan
    return out


def validate_oi(rec: dict) -> list[str]:
    """Identity checks: TOTAL long == short per segment; categories sum to TOTAL."""
    issues = []
    pairs = [("FIL", "FIS"), ("FSL", "FSS"), ("OICL", "OICS"), ("OIPL", "OIPS"),
             ("OSCL", "OSCS"), ("OSPL", "OSPS"), ("TL", "TS")]
    for a, b in pairs:
        ta, tb = rec.get(f"TOTAL_{a}"), rec.get(f"TOTAL_{b}")
        if ta is None or tb is None or abs(ta - tb) > max(2, 0.0005 * ta):
            issues.append(f"TOTAL {a}≠{b}")
    for col in COLMAP.values():
        s = sum(rec.get(f"{c}_{col}", 0) for c in CATS[:4])
        t = rec.get(f"TOTAL_{col}")
        if t is not None and abs(s - t) > max(2, 0.0005 * t):  # NSE rounding noise
            issues.append(f"sum≠TOTAL {col}")
    return issues


# ----------------------------------------------------------------- fetching
def _get(url: str, session: requests.Session):
    try:
        r = session.get(url, headers=HEADERS, timeout=20)
        if r.status_code == 200 and len(r.content) > 200:
            return r.content
        return None
    except requests.RequestException:
        return None


def fetch_day(d: dt.date, session: requests.Session) -> dict | None:
    """Fetch and parse all three files for one date. None if not a trading day."""
    oi = _get(f"{BASE}/nsccl/fao_participant_oi_{d:%d%m%Y}.csv", session)
    if oi is None:
        return None
    rec = {"date": pd.Timestamp(d)}
    try:
        rec.update(parse_participant_csv(oi, ""))
    except Exception:
        return None
    vol = _get(f"{BASE}/nsccl/fao_participant_vol_{d:%d%m%Y}.csv", session)
    if vol is not None:
        try:
            rec.update(parse_participant_csv(vol, "V_"))
        except Exception:
            pass
    fs = _get(f"{BASE}/fo/fii_stats_{d.day:02d}-{d:%b}-{d.year}.xls", session)
    if fs is not None:
        try:
            rec.update(parse_fii_stats(fs))
        except Exception:
            pass
    rec["issues"] = "; ".join(validate_oi(rec))
    return rec


def load_cache() -> tuple[pd.DataFrame, set]:
    hist = pd.DataFrame()
    if os.path.exists(CACHE_FILE):
        hist = pd.read_csv(CACHE_FILE, parse_dates=["date"])
    missing = set()
    if os.path.exists(MISSING_FILE):
        with open(MISSING_FILE) as f:
            missing = {line.strip() for line in f if line.strip()}
    return hist, missing


def save_cache(hist: pd.DataFrame, missing: set):
    os.makedirs(DATA_DIR, exist_ok=True)
    hist.sort_values("date").to_csv(CACHE_FILE, index=False)
    with open(MISSING_FILE, "w") as f:
        f.write("\n".join(sorted(missing)))


def update_history(start: dt.date, end: dt.date, progress=None, workers: int = 8) -> pd.DataFrame:
    """Incrementally fetch every weekday between start and end that is not cached."""
    hist, missing = load_cache()
    have = set(hist["date"].dt.date) if not hist.empty else set()
    days = [start + dt.timedelta(n) for n in range((end - start).days + 1)]
    todo = [d for d in days if d.weekday() < 5 and d not in have and d.isoformat() not in missing]
    new = []
    if todo:
        session = requests.Session()
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for i, (d, rec) in enumerate(zip(todo, ex.map(lambda x: fetch_day(x, session), todo))):
                if rec is None:
                    # remember holidays, but keep re-trying the last 3 days (file may be late)
                    if (dt.date.today() - d).days > 3:
                        missing.add(d.isoformat())
                else:
                    new.append(rec)
                if progress:
                    progress((i + 1) / len(todo), f"Fetched {d:%d %b %Y}")
    if new:
        hist = pd.concat([hist, pd.DataFrame(new)], ignore_index=True)
    if not hist.empty:
        hist = hist.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
    save_cache(hist, missing)
    return hist[(hist["date"].dt.date >= start) & (hist["date"].dt.date <= end)].reset_index(drop=True)


def history_info() -> dict:
    hist, _ = load_cache()
    if hist.empty:
        return {"n": 0, "first": None, "last": None, "mtime": 0}
    return {"n": len(hist), "first": hist["date"].min(), "last": hist["date"].max(),
            "mtime": os.path.getmtime(CACHE_FILE)}


def add_nifty(hist: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Fill NIFTY open/high/low/close for every stored session (Yahoo Finance)."""
    if hist.empty:
        return hist, False
    o = fetch_nifty_ohlc(hist["date"].min().date(), hist["date"].max().date())
    if o.empty:
        return hist, False
    hist = hist.copy()
    for c in NIFTY_COLS.values():
        new = hist["date"].map(o[c])
        hist[c] = new.fillna(hist[c]) if c in hist.columns else new
    return hist, True


def fetch_and_save(start: dt.date | None = None, end: dt.date | None = None, progress=None) -> dict:
    """MANUAL fetch: download NSE files for missing trading days (default: from the day after
    the last stored session up to today), add NIFTY closes, and save everything to history.csv."""
    before = history_info()
    end = end or dt.date.today()
    if start is None:
        start = (before["last"].date() + dt.timedelta(days=1)) if before["n"] else end - dt.timedelta(days=730)
    if start <= end:
        update_history(start, end, progress=progress)
    hist, missing = load_cache()
    hist, nifty_ok = add_nifty(hist)
    save_cache(hist, missing)
    after = history_info()
    return {"added": after["n"] - before["n"], "last": after["last"], "nifty_ok": nifty_ok}


def merge_into_history(new: pd.DataFrame) -> int:
    """Merge records (from uploaded NSE files or a saved history.csv) into the stored CSV."""
    if new is None or new.empty:
        return 0
    new = new.copy()
    new["date"] = pd.to_datetime(new["date"])
    if "issues" not in new.columns:
        new["issues"] = ["; ".join(validate_oi(r)) for r in new.to_dict("records")]
    new = new.drop_duplicates("date", keep="last")
    hist, missing = load_cache()
    n0 = len(hist)
    if hist.empty:
        hist = new
    else:  # new non-blank values win; blanks never wipe stored values
        hist = new.set_index("date").combine_first(hist.set_index("date")).reset_index().copy()
    hist = hist.sort_values("date").reset_index(drop=True)
    hist["issues"] = ["; ".join(validate_oi(r)) for r in hist.to_dict("records")]
    save_cache(hist, missing)
    return len(hist) - n0


def records_from_uploads(files) -> pd.DataFrame:
    """Manual fallback: build history from uploaded NSE files (names must be NSE's own)."""
    by_date: dict = {}
    extra = []
    for f in files:
        name = f.name
        raw = f.read()
        try:
            if name.startswith("fao_participant_oi_"):
                d = dt.datetime.strptime(name[19:27], "%d%m%Y").date()
                by_date.setdefault(d, {"date": pd.Timestamp(d)}).update(parse_participant_csv(raw, ""))
            elif name.startswith("fao_participant_vol_"):
                d = dt.datetime.strptime(name[20:28], "%d%m%Y").date()
                by_date.setdefault(d, {"date": pd.Timestamp(d)}).update(parse_participant_csv(raw, "V_"))
            elif name.lower().endswith(".csv") and "participant" not in name:
                df = pd.read_csv(io.BytesIO(raw), parse_dates=["date"])
                if "FII_FIL" in df.columns:
                    extra.append(df)
            elif name.startswith("fii_stats_"):
                d = dt.datetime.strptime(name[10:21], "%d-%b-%Y").date()
                by_date.setdefault(d, {"date": pd.Timestamp(d)}).update(parse_fii_stats(raw))
        except Exception:
            continue
    recs = [r for r in by_date.values() if "FII_FIL" in r]
    saved = [x for x in extra if not x.empty]
    if saved:
        out = pd.concat(saved, ignore_index=True)
        out["date"] = pd.to_datetime(out["date"])
        out = out.drop_duplicates("date", keep="last").set_index("date")
        if recs:  # NSE files fill in / update the saved CSV, never blank it
            out = pd.DataFrame(recs).set_index("date").combine_first(out)
        return out.reset_index().sort_values("date").reset_index(drop=True)
    for r in recs:
        r["issues"] = "; ".join(validate_oi(r))
    return pd.DataFrame(recs).sort_values("date").reset_index(drop=True) if recs else pd.DataFrame()


NIFTY_COLS = {"Open": "NIFTY_O", "High": "NIFTY_H", "Low": "NIFTY_L", "Close": "NIFTY"}


def fetch_nifty_ohlc(start: dt.date, end: dt.date) -> pd.DataFrame:
    """NIFTY 50 daily open/high/low/close from Yahoo Finance, columns NIFTY_O/H/L and NIFTY (close)."""
    try:
        import yfinance as yf
        d = yf.download("^NSEI", start=start, end=end + dt.timedelta(days=1),
                        progress=False, auto_adjust=False)
        if d.empty:
            return pd.DataFrame()
        out = pd.DataFrame(index=pd.to_datetime(d.index).tz_localize(None).normalize())
        for src, dst in NIFTY_COLS.items():
            col = d[src]
            out[dst] = (col.iloc[:, 0] if isinstance(col, pd.DataFrame) else col).values
        return out
    except Exception:
        return pd.DataFrame()


def fetch_nifty(start: dt.date, end: dt.date) -> pd.Series:
    o = fetch_nifty_ohlc(start, end)
    return o["NIFTY"] if not o.empty else pd.Series(dtype=float)


# ----------------------------------------------------------------- metrics
def _col(h, name):
    return h[name] if name in h.columns else pd.Series(np.nan, index=h.index)


def compute_metrics(h: pd.DataFrame) -> pd.DataFrame:
    h = h.sort_values("date").set_index("date")
    m = pd.DataFrame(index=h.index)

    L, S = h["FII_FIL"], h["FII_FIS"]
    gross = L + S
    net = L - S
    tot_gross = h["TOTAL_FIL"] + h["TOTAL_FIS"]

    # Flags: expiry (index-futures OI falls sharply) and structural jumps (lot size etc.)
    chg = tot_gross.pct_change()
    expiry = chg < -0.20
    m["flag_expiry"] = expiry | expiry.shift(1, fill_value=False)
    m["flag_break"] = chg > 0.40

    # A - index futures
    m["A1"] = L / gross * 100
    m["A2"] = net
    scale = gross.rolling(20, min_periods=5).mean()
    m["A3_5"] = (net - net.shift(5)) / scale * 100
    m["A3_20"] = (net - net.shift(20)) / scale * 100
    dL, dS = L.diff(), S.diff()
    a4 = pd.Series(0.0, index=h.index)
    a4[(dL >= 0) & (dS <= 0) & ~((dL == 0) & (dS == 0))] = 2
    a4[(dL > 0) & (dS > 0) & (dL > dS)] = 1
    a4[(dL > 0) & (dS > 0) & (dS > dL)] = -1
    a4[(dL <= 0) & (dS >= 0) & ~((dL == 0) & (dS == 0))] = -2
    a4[(dL < 0) & (dS < 0) & (dS.abs() > dL.abs())] = 1
    a4[(dL < 0) & (dS < 0) & (dL.abs() > dS.abs())] = -1
    a4[m["flag_expiry"] | m["flag_break"]] = np.nan
    a4.iloc[0] = np.nan
    m["A4_day"] = a4
    m["A4"] = a4.rolling(3, min_periods=1).mean()
    m["A5"] = h["Client_FIL"] / (h["Client_FIL"] + h["Client_FIS"]) * 100
    m["A8"] = gross / tot_gross * 100

    # B - index options (contract counts, NOT delta-weighted)
    def opt_net(cat):
        cl, cs, pl, ps = (h[f"{cat}_OICL"], h[f"{cat}_OICS"], h[f"{cat}_OIPL"], h[f"{cat}_OIPS"])
        return ((cl - cs) + (ps - pl)) / (cl + cs + pl + ps) * 100
    m["B1"] = opt_net("FII")
    m["B3"] = h["FII_OIPS"] / (h["FII_OIPS"] + h["FII_OICS"]) * 100
    m["B4"] = opt_net("Pro")
    m["B6"] = m["B1"] - m["B1"].shift(5)

    # C - stock futures
    m["C1"] = h["FII_FSL"] / (h["FII_FSL"] + h["FII_FSS"]) * 100
    m["C2"] = m["A1"] - m["C1"]

    # D - churn (confidence only)
    vL, vS = _col(h, "V_FII_FIL"), _col(h, "V_FII_FIS")
    moved = (dL.abs() + dS.abs()).replace(0, np.nan)
    m["D3"] = ((vL + vS) / moved).clip(upper=200)

    # E - NIFTY-specific FII net Rs flow (directional), from FII stats
    nf_net = _col(h, "FS_NF_buy_cr") - _col(h, "FS_NF_sell_cr")
    m["E_day"] = nf_net
    m["E2_5"] = nf_net.rolling(5, min_periods=4).sum()
    m["E2_20"] = nf_net.rolling(20, min_periods=15).sum()
    m["E_oi_cr"] = _col(h, "FS_NF_oi_cr")  # gross (long+short) - context only

    m["issues"] = h.get("issues", pd.Series("", index=h.index)).fillna("")
    return m


def rolling_pct(s: pd.Series, window: int = 252, min_periods: int = 60) -> pd.Series:
    def f(x):
        x = x[~np.isnan(x)]
        if len(x) < 2:
            return np.nan
        return (x <= x[-1]).mean() * 100
    return s.rolling(window, min_periods=min_periods).apply(f, raw=True)


def pct_to_score(p: pd.Series) -> pd.Series:
    bins = [-0.1, 20, 40, 60, 80, 100.1]
    return pd.cut(p, bins=bins, labels=[-2, -1, 0, 1, 2]).astype(float)


# metric key -> (label, horizon component?, plain-English reading)
METRIC_INFO = {
    "A1": ("FII index-futures long share", "FIIs' long share of their index-futures book"),
    "A3_5": ("FII net flow, 5 sessions", "Are FIIs adding longs or shorts this week?"),
    "A3_20": ("FII net flow, 20 sessions", "Direction of FII positioning over a month"),
    "A4": ("FII build-up type (3-day)", "Longs added / shorts covered vs shorts added / longs cut"),
    "B1": ("FII options net tilt", "Long calls + short puts vs long puts + short calls"),
    "B3": ("FII put-writing share", "Share of FII option writing that is put writing (hypothesis)"),
    "B4": ("Pro options net tilt", "Proprietary desks' option tilt"),
    "B6": ("FII options flow, 5 sessions", "Change in FII option tilt this week"),
    "C1": ("FII stock-futures long share", "FII risk appetite in single stocks"),
    "E2_5": ("FII NIFTY futures net ₹, 5 sessions", "Rupee buying minus selling of NIFTY futures"),
    "E2_20": ("FII NIFTY futures net ₹, 20 sessions", "Monthly rupee flow into NIFTY futures"),
}

DEFAULT_WEIGHTS = {
    "short": {"A3_5": 25, "B6": 25, "A4": 20, "E2_5": 15, "B4": 15},
    "medium": {"A1": 30, "A3_20": 20, "E2_20": 20, "C1": 15, "B3": 15},
}


def score_metrics(m: pd.DataFrame, window: int = 252) -> tuple[pd.DataFrame, pd.DataFrame]:
    pct = pd.DataFrame(index=m.index)
    sc = pd.DataFrame(index=m.index)
    for k in METRIC_INFO:
        if k == "A4":  # already on -2..+2
            pct[k] = np.nan
            sc[k] = m["A4"].round()
            continue
        pct[k] = rolling_pct(m[k], window)
        sc[k] = pct_to_score(pct[k])
    for k in ["A5", "A8", "C2", "D3"]:
        pct[k] = rolling_pct(m[k], window)
    return pct, sc


def composite(sc: pd.DataFrame, weights: dict) -> pd.Series:
    w = pd.Series(weights, dtype=float)
    w = w[w > 0]
    S = sc[w.index]
    avail = S.notna().mul(w, axis=1)
    num = S.fillna(0).mul(w, axis=1).sum(axis=1)
    den = avail.sum(axis=1)
    out = num / den
    out[den < 0.5 * w.sum()] = np.nan  # need at least half the weight present
    return out


def bias_label(x: float) -> str:
    if pd.isna(x):
        return "Insufficient data"
    if x >= 1.0:
        return "Strong bullish"
    if x >= 0.4:
        return "Bullish"
    if x > -0.4:
        return "Neutral"
    if x > -1.0:
        return "Bearish"
    return "Strong bearish"


def bucket3(x: float) -> str:
    if pd.isna(x):
        return "Neutral"
    return "Bullish" if x >= 0.4 else ("Bearish" if x <= -0.4 else "Neutral")


CROSS = {
    ("Bullish", "Bullish"): "Aligned uptrend", ("Bullish", "Neutral"): "Possible turn up",
    ("Bullish", "Bearish"): "Counter-trend bounce", ("Neutral", "Bullish"): "Pause in an uptrend",
    ("Neutral", "Neutral"): "Range", ("Neutral", "Bearish"): "Pause in a downtrend",
    ("Bearish", "Bullish"): "Pullback in an uptrend", ("Bearish", "Neutral"): "Possible turn down",
    ("Bearish", "Bearish"): "Aligned downtrend",
}


def alerts(m: pd.DataFrame, pct: pd.DataFrame) -> list[tuple[str, str]]:
    """Return list of (title, explanation) for the latest row. All are hypotheses."""
    out = []
    i = m.index[-1]
    p = pct.loc[i]
    if p["A1"] <= 10 and m.loc[i, "A3_5"] > 0:
        out.append(("Short-covering risk", f"FII long share ({m.loc[i, 'A1']:.1f}%) is in the bottom 10% of "
                    f"the look-back (percentile {p['A1']:.0f}), and FIIs have started reducing net shorts."))
    if p["A1"] >= 90 and m.loc[i, "A3_5"] < 0:
        out.append(("Long-unwinding risk", f"FII long share ({m.loc[i, 'A1']:.1f}%) is in the top 10% of the "
                    f"look-back (percentile {p['A1']:.0f}) - high for them, not necessarily net long - and their "
                    "net position fell over the last 5 sessions."))
    if p["A5"] >= 90:
        out.append(("Clients crowded long", f"Client long share ({m.loc[i, 'A5']:.1f}%) is in the top 10% "
                    "of the look-back."))
    if p["A5"] <= 10:
        out.append(("Clients crowded short", f"Client long share ({m.loc[i, 'A5']:.1f}%) is in the bottom 10% "
                    "of the look-back - low for them, even if still above 50%."))
    if p["C2"] <= 10 and p["C1"] >= 50:
        out.append(("Index hedge", "FIIs are short the index but not short stocks - may be a hedge "
                    "rather than an outright bearish bet."))
    return out


def confidence_flags(m: pd.DataFrame, pct: pd.DataFrame) -> list[str]:
    i = m.index[-1]
    flags = []
    if pct.loc[i, "A8"] < 20:
        flags.append("Low conviction: FII share of index-futures OI is in the bottom 20% of the look-back.")
    if pct.loc[i, "D3"] > 80:
        flags.append("Churn-driven: today's FII trading was mostly intraday, little was carried overnight.")
    if m.loc[i, "flag_expiry"]:
        flags.append("Expiry window: open-interest changes reflect roll-over / expiry, not fresh views.")
    if m.loc[i, "flag_break"]:
        flags.append("Structural jump in OI (possible lot-size change) - flows may be distorted.")
    if m.loc[i, "issues"]:
        flags.append(f"Data check failed: {m.loc[i, 'issues']}")
    return flags


# ----------------------------------------------------------------- backtest
def backtest(sig: pd.DataFrame, nifty: pd.Series) -> dict:
    """Signal at T is usable at close of T+1. Short = T+1->T+6, medium = T+1->T+21."""
    px = nifty.reindex(sig.index).ffill()
    entry = px.shift(-1)
    fwd = {"short": px.shift(-6) / entry - 1, "medium": px.shift(-21) / entry - 1}
    res = {}
    for hz, cols in [("short", list(DEFAULT_WEIGHTS["short"]) + ["SHORT"]),
                     ("medium", list(DEFAULT_WEIGHTS["medium"]) + ["MEDIUM"])]:
        rows = []
        r = fwd[hz]
        for c in cols:
            if c not in sig:
                continue
            s = sig[c]
            if c in ("SHORT", "MEDIUM"):
                s = s.apply(lambda v: np.nan if pd.isna(v) else
                            (2 if v >= 1 else 1 if v >= 0.4 else 0 if v > -0.4 else -1 if v > -1 else -2))
            df = pd.DataFrame({"s": s, "r": r}).dropna()
            if len(df) < 30:
                continue
            g = df.groupby("s")["r"]
            row = {"metric": c, "n": len(df)}
            for b in [-2, -1, 0, 1, 2]:
                row[f"avg {b:+d}"] = g.mean().get(b, np.nan) * 100
            bull = df[df.s > 0]
            bear = df[df.s < 0]
            row["hit bull %"] = (bull.r > 0).mean() * 100 if len(bull) else np.nan
            row["hit bear %"] = (bear.r < 0).mean() * 100 if len(bear) else np.nan
            row["spread (bull−bear) %"] = (bull.r.mean() - bear.r.mean()) * 100 if len(bull) and len(bear) else np.nan
            # Spearman = Pearson on ranks (avoids needing scipy)
            row["rank corr"] = df["s"].rank().corr(df["r"].rank())
            rows.append(row)
        res[hz] = pd.DataFrame(rows)
        res[hz + "_base"] = r.dropna().mean() * 100 if r.notna().any() else np.nan
    return res


# ----------------------------------------------------------------- date-wise lag check
def lag_check(short: pd.Series, med: pd.Series, nifty: pd.Series, s_lag: int = 5, m_lag: int = 20,
              window: int = 60, band: float = 0.4) -> pd.DataFrame:
    """For each session T: NIFTY close, the composite from `lag` sessions earlier, NIFTY's % move since
    then, whether the earlier bias called the direction, and the rolling correlation (%) between the
    lagged composite and the subsequent NIFTY move over the last `window` sessions."""
    px = nifty.reindex(short.index).ffill()
    out = pd.DataFrame({"NIFTY close": px})
    for tag, comp, lag in [("Short", short, s_lag), ("Medium", med, m_lag)]:
        past = comp.shift(lag)
        move = (px / px.shift(lag) - 1) * 100
        bias = past.apply(lambda v: np.nan if pd.isna(v) else ("Bullish" if v >= band else
                                                               "Bearish" if v <= -band else "Neutral"))
        hit = pd.Series(np.nan, index=comp.index, dtype=object)
        called = bias.isin(["Bullish", "Bearish"]) & move.notna()
        hit[called] = np.where((bias[called] == "Bullish") == (move[called] > 0), "✓", "✗")
        hit[bias.eq("Neutral") & move.notna()] = "–"
        corr = past.rolling(window, min_periods=max(10, window // 2)).corr(move) * 100
        out[f"{tag} composite (T−{lag})"] = past
        out[f"{tag} bias (T−{lag})"] = bias
        out[f"NIFTY % move T−{lag}→T"] = move
        out[f"{tag} call"] = hit
        out[f"{tag} rolling corr % ({window})"] = corr
    return out


def hit_rate(col: pd.Series) -> tuple[float, int]:
    called = col.isin(["✓", "✗"])
    n = int(called.sum())
    return ((col[called] == "✓").mean() * 100 if n else np.nan), n


# ----------------------------------------------------------------- saved default settings
SETTINGS_FILE = os.path.join(DATA_DIR, "settings.json")


def load_settings() -> dict:
    """Defaults every visitor sees; the owner can overwrite them from the app."""
    import json
    base = {"years": 2, "window": 252,
            "short": dict(DEFAULT_WEIGHTS["short"]), "medium": dict(DEFAULT_WEIGHTS["medium"])}
    try:
        with open(SETTINGS_FILE) as f:
            saved = json.load(f)
        base["years"] = saved.get("years", base["years"])
        base["window"] = saved.get("window", base["window"])
        for hz in ("short", "medium"):
            base[hz].update({k: v for k, v in saved.get(hz, {}).items() if k in base[hz]})
    except (OSError, ValueError):
        pass
    return base


def save_settings(cfg: dict) -> str:
    import json
    os.makedirs(DATA_DIR, exist_ok=True)
    txt = json.dumps(cfg, indent=2)
    with open(SETTINGS_FILE, "w") as f:
        f.write(txt)
    return txt


# ----------------------------------------------------------------- FII vs Client divergence
def fii_client_divergence(m: pd.DataFrame, lookback: int = 3, thr: float = 3.0) -> pd.DataFrame:
    """Days when FII long share and Client long share move in opposite directions by more than `thr`
    percentage points over `lookback` sessions. 'start' marks only the first day of each run."""
    d1, d5 = m["A1"].diff(lookback), m["A5"].diff(lookback)
    bull = (d1 > thr) & (d5 < -thr)   # FII adding longs, clients cutting longs
    bear = (d1 < -thr) & (d5 > thr)   # FII cutting longs, clients adding longs
    out = pd.DataFrame({"bull": bull, "bear": bear, "dFII": d1, "dClient": d5})
    out["bull_start"] = bull & ~bull.shift(1, fill_value=False)
    out["bear_start"] = bear & ~bear.shift(1, fill_value=False)
    return out
