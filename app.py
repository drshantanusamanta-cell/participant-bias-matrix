"""
Participant F&O Bias Matrix - Streamlit app
Reads NSE participant-wise OI / volume and FII derivatives statistics,
scores them on the Participant F&O Bias Matrix framework and shows a
short-term (1-5 sessions) and medium-term (2-6 weeks) NIFTY bias.

© Dr Shantanu Samanta. For education and research only - not investment advice.
"""
import datetime as dt
import hmac
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import core

st.set_page_config(page_title="Participant F&O Bias Matrix", page_icon="🧭", layout="wide")

SCORE_COL = {-2: "#c0392b", -1: "#e67e22", 0: "#95a5a6", 1: "#27ae60", 2: "#1e8449"}
BIAS_COL = {"Strong bullish": "#1e8449", "Bullish": "#27ae60", "Neutral": "#7f8c8d",
            "Bearish": "#e67e22", "Strong bearish": "#c0392b", "Insufficient data": "#bdc3c7"}
CAT_COL = {"FII": "#2e86c1", "DII": "#8e44ad", "Pro": "#d68910", "Client": "#17a589"}
UNITS = {"A1": "%", "A3_5": "% of FII OI", "A3_20": "% of FII OI", "A4": "score", "B1": "%",
         "B3": "%", "B4": "%", "B6": "pts", "C1": "%", "E2_5": "₹ cr", "E2_20": "₹ cr"}

st.markdown("""
<style>
.card{border-radius:12px;padding:14px 16px;margin-bottom:10px;color:white}
.card h4{margin:0 0 4px 0;font-size:1.05rem}
.card p{margin:0;font-size:.9rem;opacity:.95}
.flag{border-left:5px solid #f39c12;background:#fef5e7;padding:8px 12px;margin-bottom:6px;border-radius:6px;font-size:.9rem}
.big{font-size:1.6rem;font-weight:700}
.ctitle{font-weight:600;font-size:1rem;margin:.6rem 0 .1rem 0;line-height:1.3}
@media (max-width:640px){.big{font-size:1.3rem}.card h4{font-size:.95rem}.ctitle{font-size:.92rem}}
</style>""", unsafe_allow_html=True)


# ----------------------------------------------------------------- data
@st.cache_data(show_spinner=False)
def load_history(mtime: float) -> pd.DataFrame:
    """Read the saved CSV only - no network. Re-read whenever the file changes."""
    return core.load_cache()[0]


def run_fetch(start=None, end=None):
    bar = st.sidebar.progress(0.0, text="Starting…")
    try:
        r = core.fetch_and_save(start, end, progress=lambda f, t: bar.progress(min(f, 1.0), text=t))
        bar.empty()
        msg = (f"Added {r['added']} new session(s). Latest stored: {r['last']:%d %b %Y}."
               if r["added"] else f"No new sessions found (NSE publishes after ~7 PM IST). "
                                  f"Latest stored: {r['last']:%d %b %Y}.")
        if not r["nifty_ok"]:
            msg += " NIFTY closes could not be updated this time."
        st.session_state["fetch_msg"] = ("success", msg)
    except Exception as e:
        bar.empty()
        st.session_state["fetch_msg"] = ("error", f"Fetch failed: {e}. If NSE is blocking this server, "
                                                  "use 'Add data from files' instead.")
    st.rerun()


def show(target, fig, zoom=False):
    """Render a Plotly chart so it works on phone, tablet and desktop: title as page text above the chart
    (wraps instead of being cut off), legend below the plot, toolbar only on zoomable charts."""
    title = fig.layout.title.text
    if title:
        target.markdown(f"<div class='ctitle'>{title}</div>", unsafe_allow_html=True)
        fig.update_layout(title=None)
    has_sel = bool(fig.layout.xaxis.rangeselector and fig.layout.xaxis.rangeselector.buttons)
    if fig.layout.showlegend is not False:
        fig.update_layout(legend=dict(orientation="h", x=0, xanchor="left", y=-0.12, yanchor="top",
                                      font=dict(size=11)))
    fig.update_layout(margin=dict(l=8, r=8, t=40 if has_sel else 12, b=10), autosize=True)
    # No floating toolbar: it covered titles and the 1M-All buttons on small screens.
    # Drag-to-zoom and double-click-to-reset still work everywhere; mouse-wheel zoom on zoomable charts.
    cfg = {"displaylogo": False, "responsive": True, "displayModeBar": False, "scrollZoom": zoom}
    target.plotly_chart(fig, width="stretch", config=cfg)


WINDOWS = ["1M", "3M", "6M", "1Y", "YTD", "ALL"]


def chart_header(title: str, key: str, last: pd.Timestamp):
    """Title on the left, 1M/3M/6M/1Y/YTD/ALL picker on the right (stacks on phones).
    Returns the window start date (None = all). Data is sliced, so every axis rescales to the window."""
    c1, c2 = st.columns([3, 2], vertical_alignment="bottom")
    c1.markdown(f"<div class='ctitle'>{title}</div>", unsafe_allow_html=True)
    choice = c2.segmented_control("Time window", WINDOWS, default="ALL", key=key,
                                  label_visibility="collapsed") or "ALL"
    months = {"1M": 1, "3M": 3, "6M": 6, "1Y": 12}
    if choice in months:
        return last - pd.DateOffset(months=months[choice])
    if choice == "YTD":
        return pd.Timestamp(year=last.year, month=1, day=1)
    return None


def in_window(x, start):
    """Slice a Series/DataFrame with a DatetimeIndex to the chosen window."""
    return x if start is None or len(x) == 0 else x[x.index >= start]


def fmt(k, v):
    if pd.isna(v):
        return "–"
    u = UNITS.get(k, "")
    if u == "₹ cr":
        return f"₹{v:,.0f} cr"
    if u == "score":
        return f"{v:+.1f}"
    if u == "pts":
        return f"{v:+.1f} pts"
    if u.startswith("% of"):
        return f"{v:+.1f}%"
    return f"{v:.1f}%"


def gauge(value, title):
    label = core.bias_label(value)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=0 if pd.isna(value) else value,
        number={"valueformat": "+.2f", "font": {"size": 34}},
        gauge={"axis": {"range": [-2, 2], "tickvals": [-2, -1, -0.4, 0.4, 1, 2]},
               "bar": {"color": "#2c3e50", "thickness": 0.25},
               "steps": [{"range": [-2, -1], "color": "#f5b7b1"}, {"range": [-1, -0.4], "color": "#fad7a0"},
                         {"range": [-0.4, 0.4], "color": "#e5e8e8"}, {"range": [0.4, 1], "color": "#abebc6"},
                         {"range": [1, 2], "color": "#82e0aa"}]}))
    fig.update_layout(height=240, margin=dict(l=25, r=25, t=20, b=10))
    return fig


def cross_matrix(short_b, med_b):
    rows = ["Bullish", "Neutral", "Bearish"]
    z, text = [], []
    for s in rows:
        zr, tr = [], []
        for m in rows:
            zr.append(1 if (s == short_b and m == med_b) else 0)
            tr.append(core.CROSS[(s, m)].replace(" in an ", "<br>in an ").replace(" in a ", "<br>in a ")
                      .replace("Possible turn", "Possible<br>turn").replace("Counter-trend ", "Counter-trend<br>")
                      .replace("Aligned ", "Aligned<br>"))
        z.append(zr)
        text.append(tr)
    fig = go.Figure(go.Heatmap(z=z, x=[f"Medium<br>{m}" for m in rows], y=[f"Short<br>{s}" for s in rows],
                               text=text, texttemplate="%{text}", textfont={"size": 12},
                               colorscale=[[0, "#f4f6f7"], [1, "#f9e79f"]], showscale=False,
                               xgap=4, ygap=4, hoverinfo="skip"))
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=11))
    fig.update_xaxes(tickangle=0, tickfont=dict(size=11), side="bottom")
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10),
                      title={"text": "Where today sits (highlighted)", "font": {"size": 14}})
    return fig


# ----------------------------------------------------------------- sidebar
def owner_password() -> str:
    try:
        return str(st.secrets.get("OWNER_PASSWORD", ""))
    except Exception:  # no secrets file at all
        return ""


def owner_login() -> bool:
    """Owner login in the sidebar. The password lives in Streamlit secrets (OWNER_PASSWORD), never in the code."""
    if st.session_state.get("is_owner"):
        if st.sidebar.button("🔓 Log out of owner mode", width="stretch"):
            st.session_state["is_owner"] = False
            st.rerun()
        return True
    with st.sidebar.expander("🔒 Owner login"):
        pw = owner_password()
        if not pw:
            st.caption("Owner tools are disabled: no OWNER_PASSWORD is set in the app's Secrets.")
            return False
        tries = st.session_state.get("login_tries", 0)
        if tries >= 5:
            st.error("Too many wrong attempts. Reload the page to try again.")
            return False
        with st.form("owner_login", clear_on_submit=True, border=False):
            entered = st.text_input("Password", type="password")
            ok = st.form_submit_button("Log in", width="stretch")
        if ok:
            if hmac.compare_digest(entered.encode(), pw.encode()):
                st.session_state["is_owner"] = True
                st.session_state["login_tries"] = 0
                st.rerun()
            else:
                st.session_state["login_tries"] = tries + 1
                st.error("Wrong password.")
    return False


def sidebar():
    info = core.history_info()
    cfg = core.load_settings()
    st.sidebar.header("📁 Data")
    if info["n"]:
        st.sidebar.markdown(f"**{info['n']:,} sessions** stored  \n{info['first']:%d %b %Y} → "
                            f"**{info['last']:%d %b %Y}**")
    else:
        st.sidebar.warning("No data saved yet.")

    owner = owner_login()
    if not owner:
        st.sidebar.caption("View-only. Data updates and analysis settings are managed by the owner.")
        return info, cfg["years"], cfg["window"], cfg["short"], cfg["medium"], False

    # ---------------- owner tools
    st.sidebar.success("Owner mode")
    if st.sidebar.button("📥 Fetch new days from NSE", type="primary", width="stretch",
                         help="Downloads only the trading days after the last stored session, "
                              "adds NIFTY prices, and saves to data/history.csv."):
        run_fetch()
    with st.sidebar.expander("Backfill / re-fetch a date range"):
        c1, c2 = st.columns(2)
        d1 = c1.date_input("From", dt.date.today() - dt.timedelta(days=30), format="DD/MM/YYYY")
        d2 = c2.date_input("To", dt.date.today(), format="DD/MM/YYYY")
        if st.button("Fetch this range", width="stretch"):
            run_fetch(d1, d2)
    if info["n"]:
        with open(core.CACHE_FILE, "rb") as f:
            st.sidebar.download_button("⬇️ Download history.csv", f.read(), "history.csv", "text/csv",
                                       width="stretch")
    with st.sidebar.expander("Add data from files"):
        st.caption("Upload a saved history.csv and/or NSE files with their original names "
                   "(fao_participant_oi_*.csv, fao_participant_vol_*.csv, fii_stats_*.xls). "
                   "They are merged into the saved CSV.")
        files = st.file_uploader("Files", accept_multiple_files=True, label_visibility="collapsed")
        if files and st.button("Merge into saved CSV", width="stretch"):
            n = core.merge_into_history(core.records_from_uploads(files))
            st.session_state["fetch_msg"] = ("success", f"Merged files: {n} new session(s) added.")
            st.rerun()
    st.sidebar.caption("On Streamlit Cloud the saved files reset when the app restarts. After fetching, "
                       "download history.csv and replace data/history.csv in GitHub to keep it.")

    st.sidebar.header("⚙️ Analysis settings")
    years_opts = [1.5, 2, 3, 5, 8, 11]
    years = st.sidebar.select_slider("History to analyse", options=years_opts,
                                     value=cfg["years"] if cfg["years"] in years_opts else 2,
                                     format_func=lambda y: f"{y} years",
                                     help="Counted back from the latest stored session.")
    win_opts = [126, 189, 252, 504]
    window = st.sidebar.select_slider("Percentile look-back (sessions)", options=win_opts,
                                      value=cfg["window"] if cfg["window"] in win_opts else 252)
    with st.sidebar.expander("Short-term weights"):
        ws = {k: st.slider(core.METRIC_INFO[k][0], 0, 100, int(v), 5, key=f"s_{k}")
              for k, v in cfg["short"].items()}
    with st.sidebar.expander("Medium-term weights"):
        wm = {k: st.slider(core.METRIC_INFO[k][0], 0, 100, int(v), 5, key=f"m_{k}")
              for k, v in cfg["medium"].items()}
    st.sidebar.caption("Changes apply to your session. Save them to make them what visitors see.")
    if st.sidebar.button("💾 Save as default for visitors", width="stretch"):
        txt = core.save_settings({"years": years, "window": window, "short": ws, "medium": wm})
        st.session_state["settings_json"] = txt
        st.session_state["fetch_msg"] = ("success", "Saved. Visitors now see these settings.")
        st.rerun()
    if os.path.exists(core.SETTINGS_FILE):
        with open(core.SETTINGS_FILE, "rb") as f:
            st.sidebar.download_button("⬇️ Download settings.json", f.read(), "settings.json",
                                       "application/json", width="stretch",
                                       help="Upload to GitHub as data/settings.json so the defaults survive restarts.")
    return info, years, window, ws, wm, True


# ----------------------------------------------------------------- main
def main():
    st.title("🧭 Participant F&O Bias Matrix")
    st.caption("NIFTY short-term (1–5 sessions) and medium-term (2–6 weeks) bias from NSE participant-wise "
               "F&O positions and FII derivatives statistics. Education and research only — not investment advice.")
    info, years, window, ws, wm, owner = sidebar()

    msg = st.session_state.pop("fetch_msg", None)
    if msg:
        (st.success if msg[0] == "success" else st.error)(msg[1])

    if not info["n"]:
        st.info("No saved data yet. Log in as owner and press **📥 Fetch new days from NSE**." if not owner
                else "No saved data yet. Press **📥 Fetch new days from NSE** in the sidebar "
                     "(or add files under 'Add data from files').")
        return
    full = load_history(info["mtime"])
    start = full["date"].max() - pd.Timedelta(days=int(years * 365.25))
    hist = full[full["date"] >= start].reset_index(drop=True)
    if len(hist) < 30:
        st.error("Fewer than 30 sessions saved - fetch or add more data.")
        return
    stale = (pd.Timestamp(dt.date.today()) - full["date"].max()).days
    if stale > 3 and owner:
        st.warning(f"Saved data ends {full['date'].max():%d %b %Y} ({stale} days ago). "
                   "Press **📥 Fetch new days from NSE** to update.")

    m = core.compute_metrics(hist)
    pct, sc = core.score_metrics(m, window)
    short = core.composite(sc, ws)
    med = core.composite(sc, wm)
    nifty = (hist.set_index("date")["NIFTY"].dropna() if "NIFTY" in hist.columns
             else pd.Series(dtype=float))
    end = dt.date.today()
    last = m.index[-1]
    s_val, m_val = short.iloc[-1], med.iloc[-1]
    s_lab, m_lab = core.bias_label(s_val), core.bias_label(m_val)

    st.success(f"Analysed {len(m)} sessions · latest NSE data: **{last:%a %d %b %Y}**")

    tabs = st.tabs(["📊 Today's bias", "👥 Who holds what", "📈 Trends", "🧪 Backtest", "📚 Data & method"])

    # ---------------- TAB 1
    with tabs[0]:
        c1, c2, c3 = st.columns([1, 1, 1.3])
        for col, val, lab, ttl in [(c1, s_val, s_lab, "Short term · 1–5 sessions"),
                                   (c2, m_val, m_lab, "Medium term · 2–6 weeks")]:
            col.markdown(f"<div style='text-align:center'><span style='color:#566573'>{ttl}</span><br>"
                         f"<span style='font-size:1.5rem;font-weight:700;color:{BIAS_COL[lab]}'>{lab}</span></div>",
                         unsafe_allow_html=True)
            col.plotly_chart(gauge(val, ttl), width="stretch", config={"displayModeBar": False})
        show(c3, cross_matrix(core.bucket3(s_val), core.bucket3(m_val)))

        verdict = core.CROSS[(core.bucket3(s_val), core.bucket3(m_val))]
        nclose = nifty.get(last, np.nan) if not nifty.empty else np.nan
        st.markdown(f"<div class='card' style='background:#34495e'><h4>Overall read: {verdict}</h4>"
                    f"<p>Short term {s_lab} ({s_val:+.2f}) · Medium term {m_lab} ({m_val:+.2f})"
                    f"{'' if pd.isna(nclose) else f' · NIFTY close {nclose:,.0f}'}</p></div>",
                    unsafe_allow_html=True)

        a, f = core.alerts(m, pct), core.confidence_flags(m, pct)
        ca, cf = st.columns(2)
        with ca:
            st.subheader("🚨 Alerts (hypotheses)")
            if not a:
                st.caption("No alert conditions today.")
            for t, txt in a:
                st.markdown(f"<div class='card' style='background:#7d3c98'><h4>{t}</h4><p>{txt}</p></div>",
                            unsafe_allow_html=True)
        with cf:
            st.subheader("⚠️ Confidence flags")
            if not f:
                st.caption("No confidence issues today.")
            for t in f:
                st.markdown(f"<div class='flag'>{t}</div>", unsafe_allow_html=True)

        st.subheader("Scorecard")
        for hz, w in [("Short term", ws), ("Medium term", wm)]:
            st.markdown(f"**{hz}**")
            cols = st.columns(len(w))
            for col, (k, wt) in zip(cols, w.items()):
                scv = sc.loc[last, k]
                colr = SCORE_COL.get(int(scv), "#bdc3c7") if not pd.isna(scv) else "#bdc3c7"
                p = pct.loc[last, k]
                ptxt = "" if pd.isna(p) else f"{p:.0f}th pct of past {window}"
                col.markdown(
                    f"<div class='card' style='background:{colr}'><h4>{core.METRIC_INFO[k][0]}</h4>"
                    f"<div class='big'>{fmt(k, m.loc[last, k])}</div>"
                    f"<p>Score {'' if pd.isna(scv) else f'{int(scv):+d}'} · weight {wt}%<br>{ptxt}</p>"
                    f"<p style='font-size:.8rem;margin-top:4px'>{core.METRIC_INFO[k][1]}</p></div>",
                    unsafe_allow_html=True)
        st.caption("Colours: dark green +2 · green +1 · grey 0 · orange −1 · red −2. "
                   "Scores are percentile ranks versus the look-back, so they are relative to recent history.")

    # ---------------- TAB 2
    with tabs[1]:
        h = hist.set_index("date").loc[last]
        cats = ["FII", "DII", "Pro", "Client"]
        c1, c2 = st.columns(2)
        fig = go.Figure()
        fig.add_bar(name="Long", x=cats, y=[h[f"{c}_FIL"] for c in cats], marker_color="#27ae60")
        fig.add_bar(name="Short", x=cats, y=[-h[f"{c}_FIS"] for c in cats], marker_color="#c0392b")
        fig.add_scatter(name="Net", x=cats, y=[h[f"{c}_FIL"] - h[f"{c}_FIS"] for c in cats],
                        mode="markers+text", marker=dict(size=14, color="#2c3e50", symbol="diamond"),
                        text=[f"<b>{h[f'{c}_FIL'] - h[f'{c}_FIS']:+,.0f}</b>" for c in cats],
                        textposition="middle right", textfont=dict(color="#1c2833", size=13))
        fig.update_layout(barmode="relative", title="Index futures: long vs short (contracts)", height=380,
                          xaxis_range=[-0.5, 3.95])
        show(c1, fig)

        share = [h[f"{c}_FIL"] / (h[f"{c}_FIL"] + h[f"{c}_FIS"]) * 100 for c in cats]
        fig = go.Figure(go.Bar(x=share, y=cats, orientation="h", marker_color=[CAT_COL[c] for c in cats],
                               text=[f"{s:.1f}%" for s in share], textposition="outside"))
        fig.add_vline(x=50, line_dash="dash", line_color="grey")
        fig.update_layout(title="Long share of each group's index-futures book", xaxis_range=[0, 100], height=380)
        show(c2, fig)

        c1, c2 = st.columns(2)
        for col, cat in [(c1, "FII"), (c2, "Pro")]:
            vals = [h[f"{cat}_OICL"], h[f"{cat}_OIPS"], -h[f"{cat}_OICS"], -h[f"{cat}_OIPL"]]
            fig = go.Figure(go.Bar(x=["Call long", "Put short", "Call short", "Put long"], y=vals,
                                   marker_color=["#27ae60", "#82e0aa", "#e67e22", "#c0392b"],
                                   text=[f"{abs(v):,.0f}" for v in vals], textposition="outside"))
            fig.update_layout(title=f"{cat} index options (bullish legs up, bearish legs down)", height=360)
            show(col, fig)
        st.caption("Option counts are contracts, not delta-weighted: a far out-of-the-money option counts the same "
                   "as an at-the-money one. Every long has a short somewhere, so the four groups' nets add to zero.")

    # ---------------- TAB 3
    with tabs[2]:
        # One chart: NIFTY candles (left axis) with both composites overlaid (right axis, −2…+2 bands)
        w0 = chart_header("NIFTY 50 (candles) with composite bias", "win_comp", last)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        hh = in_window(hist.set_index("date"), w0)
        has_ohlc = all(c in hh.columns for c in ["NIFTY_O", "NIFTY_H", "NIFTY_L", "NIFTY"])
        if has_ohlc and hh["NIFTY_O"].notna().any():
            o = hh[["NIFTY_O", "NIFTY_H", "NIFTY_L", "NIFTY"]].dropna()
            fig.add_trace(go.Candlestick(
                x=o.index, open=o["NIFTY_O"], high=o["NIFTY_H"], low=o["NIFTY_L"], close=o["NIFTY"],
                name="NIFTY 50", increasing=dict(line=dict(color="#1e8449", width=1), fillcolor="#27ae60"),
                decreasing=dict(line=dict(color="#c0392b", width=1), fillcolor="#e74c3c"),
                whiskerwidth=0.3), secondary_y=False)
        elif not nifty.empty:
            fig.add_scatter(x=in_window(nifty, w0).index, y=in_window(nifty, w0), name="NIFTY 50",
                            line=dict(color="#1b4f72", width=2.2), secondary_y=False)
        for y0, y1, c in [(1, 2, "#82e0aa"), (0.4, 1, "#d5f5e3"), (-1, -0.4, "#fdebd0"), (-2, -1, "#f5b7b1")]:
            fig.add_hrect(y0=y0, y1=y1, fillcolor=c, opacity=0.22, line_width=0, layer="below",
                          secondary_y=True)
        fig.add_hline(y=0, line=dict(color="#95a5a6", width=1, dash="dot"), secondary_y=True)
        sw, mw = in_window(short, w0), in_window(med, w0)
        fig.add_scatter(x=sw.index, y=sw, name="Short-term composite",
                        line=dict(color="#e67e22", width=1.2), opacity=0.75, secondary_y=True,
                        hovertemplate="%{y:+.2f}<extra>Short</extra>")
        fig.add_scatter(x=mw.index, y=mw, name="Medium-term composite",
                        line=dict(color="#2e86c1", width=2.6), secondary_y=True,
                        hovertemplate="%{y:+.2f}<extra>Medium</extra>")
        fig.update_layout(
            height=620, hovermode="x unified", dragmode="zoom", plot_bgcolor="white")
        fig.update_xaxes(
            rangebreaks=[dict(bounds=["sat", "mon"])], rangeslider=dict(visible=False),
            showspikes=True, spikemode="across", spikethickness=1, spikecolor="#7f8c8d")
        fig.update_yaxes(title="NIFTY 50", tickformat=",.0f", gridcolor="#eaeded", secondary_y=False)
        fig.update_yaxes(title="Composite (−2 … +2)", range=[-2.1, 2.1], fixedrange=True, showgrid=False,
                         tickvals=[-2, -1, -0.4, 0, 0.4, 1, 2], secondary_y=True)
        show(st, fig, zoom=True)
        st.caption("Green/red candles = NIFTY up/down day (left scale). Blue = medium-term composite, orange = "
                   "short-term (right scale); background bands = strong bearish → strong bullish zones. "
                   "Pick 1M–ALL to change the period (all scales refit), drag a box to magnify further, "
                   "and double-click to undo the zoom.")

        with st.expander("📅 Date-wise check: did the earlier bias call NIFTY's move?", expanded=False):
            if nifty.empty:
                st.warning("NIFTY closes are missing from the saved data - press Fetch to add them.")
            else:
                k1, k2, k3, k4, k5 = st.columns(5)
                s_lag = k1.number_input("Short-term lag (sessions)", 1, 30, 5, key="lag_s")
                m_lag = k2.number_input("Medium-term lag (sessions)", 5, 90, 20, key="lag_m")
                win = k3.number_input("Rolling correlation window", 20, 504, 60, step=10, key="lag_w")
                band = k4.number_input("Neutral band (±)", 0.0, 1.5, 0.4, step=0.1, key="lag_b")
                rows = k5.number_input("Sessions to show", 10, len(m), min(60, len(m)), step=10, key="lag_r")
                lc = core.lag_check(short, med, nifty, int(s_lag), int(m_lag), int(win), float(band))
                view = lc.iloc[::-1].head(int(rows)).copy()
                view.index = view.index.strftime("%a %d %b %Y")
                view.index.name = "Date"

                hs, ns = core.hit_rate(view["Short call"])
                hm, nm = core.hit_rate(view["Medium call"])
                cs = view[f"Short rolling corr % ({int(win)})"].iloc[0]
                cm = view[f"Medium rolling corr % ({int(win)})"].iloc[0]
                q1, q2, q3, q4 = st.columns(4)
                q1.metric(f"Short-term hit rate ({ns} calls)", "–" if pd.isna(hs) else f"{hs:.0f}%")
                q2.metric(f"Short-term correlation, last {int(win)}", "–" if pd.isna(cs) else f"{cs:+.0f}%")
                q3.metric(f"Medium-term hit rate ({nm} calls)", "–" if pd.isna(hm) else f"{hm:.0f}%")
                q4.metric(f"Medium-term correlation, last {int(win)}", "–" if pd.isna(cm) else f"{cm:+.0f}%")

                corr_cols = [c for c in view.columns if "rolling corr" in c]
                num_cols = [c for c in view.columns if c not in corr_cols and view[c].dtype.kind == "f"]

                def colour_call(v):
                    return {"✓": "background-color:#abebc6", "✗": "background-color:#f5b7b1"}.get(v, "")

                def colour_bias(v):
                    return {"Bullish": "color:#1e8449;font-weight:600",
                            "Bearish": "color:#c0392b;font-weight:600"}.get(v, "color:#7f8c8d")

                def colour_move(v):
                    return "" if pd.isna(v) else ("color:#1e8449" if v > 0 else "color:#c0392b")

                sty = (view.style
                       .format("{:,.2f}", subset=num_cols, na_rep="–")
                       .format("{:,.0f}", subset=["NIFTY close"], na_rep="–")
                       .format("{:+.0f}%", subset=corr_cols, na_rep="–")
                       .format("{:+.2f}%", subset=[c for c in view.columns if c.startswith("NIFTY % move")],
                               na_rep="–")
                       .map(colour_call, subset=["Short call", "Medium call"])
                       .map(colour_bias, subset=[c for c in view.columns if " bias " in c])
                       .map(colour_move, subset=[c for c in view.columns if c.startswith("NIFTY % move")])
                       .background_gradient(subset=corr_cols, cmap="RdYlGn", vmin=-60, vmax=60))
                st.dataframe(sty, width="stretch", height=460)
                st.download_button("⬇️ Download this table (CSV)", lc.iloc[::-1].to_csv().encode(),
                                   "nifty_vs_composite_check.csv", "text/csv")
                st.caption(
                    "Each row is a trading day T. **Composite (T−lag)** is the reading that many sessions earlier; "
                    "**NIFTY % move** is the change in NIFTY close from T−lag to T. **Call** = ✓ when a bullish/bearish "
                    "reading was followed by a NIFTY move in the same direction, ✗ when opposite, – when the reading "
                    "was inside the neutral band. **Rolling corr %** = correlation (×100) between the lagged composite "
                    "and the following NIFTY move over the last N sessions: +100% perfect, 0% none, negative = "
                    "worked the wrong way. NSE files come out after the close, so in practice you could act one "
                    "session later than T−lag.")

        nifty_line = dict(color="#7d3c98", width=2.2)

        w1 = chart_header("FII NIFTY futures net ₹ crore (buy − sell) vs NIFTY spot", "win_e", last)
        mm = in_window(m, w1)
        nn = in_window(nifty[nifty.index >= m.index.min()], w1) if not nifty.empty else nifty
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        e = mm["E_day"].dropna()
        fig.add_bar(x=e.index, y=e, name="FII NIFTY futures net ₹ cr (daily)",
                    marker_color=np.where(e >= 0, "#27ae60", "#c0392b"), opacity=0.8,
                    hovertemplate="₹%{y:,.0f} cr<extra>Daily</extra>")
        fig.add_scatter(x=mm.index, y=mm["E2_20"], name="20-session sum", line=dict(color="#2c3e50", width=2),
                        hovertemplate="₹%{y:,.0f} cr<extra>20-session</extra>")
        if not nn.empty:
            fig.add_scatter(x=nn.index, y=nn, name="NIFTY spot", line=nifty_line, secondary_y=True,
                            hovertemplate="%{y:,.0f}<extra>NIFTY</extra>")
        fig.add_hline(y=0, line=dict(color="#95a5a6", width=1))
        fig.update_layout(height=420,
                          hovermode="x unified", bargap=0, dragmode="zoom")
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
        fig.update_yaxes(title="₹ crore", tickformat=",.0f", secondary_y=False)
        fig.update_yaxes(title="NIFTY", tickformat=",.0f", showgrid=False, secondary_y=True)
        show(st, fig, zoom=True)
        if mm["E_day"].notna().sum() < len(mm):
            st.caption("NSE's FII statistics split out NIFTY futures only from 31 Jan 2023; earlier sessions are blank.")

        w2 = chart_header("Who is long the index? (vs NIFTY spot)", "win_long", last)
        k1, k2, k3 = st.columns([1, 1, 2])
        dv_n = k1.number_input("Change over (sessions)", 1, 10, 3, key="dv_n",
                               help="How many sessions back to measure each group's change in long share.")
        dv_t = k2.number_input("Minimum change (% points)", 0.5, 10.0, 3.0, step=0.5, key="dv_t",
                               help="Both groups must move at least this much, in opposite directions.")
        dv_all = k3.toggle("Mark every day (not just the first day of each run)", False, key="dv_all")
        dv = core.fii_client_divergence(m, int(dv_n), float(dv_t))
        mm = in_window(m, w2)
        dvw = in_window(dv, w2)
        nn = in_window(nifty[nifty.index >= m.index.min()], w2) if not nifty.empty else nifty
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_scatter(x=mm.index, y=mm["A1"], name="FII long share %", line=dict(color="#2e86c1", width=2))
        fig.add_scatter(x=mm.index, y=mm["A5"], name="Client long share %", line=dict(color="#17a589", dash="dot"))
        if not nn.empty:
            fig.add_scatter(x=nn.index, y=nn, name="NIFTY spot", line=nifty_line, secondary_y=True,
                            hovertemplate="%{y:,.0f}<extra>NIFTY</extra>")
        for col, lab, sym, colr in [("bull", "FII adding longs · Clients cutting", "triangle-up", "#1e8449"),
                                    ("bear", "FII cutting longs · Clients adding", "triangle-down", "#c0392b")]:
            idx = dvw.index[dvw[col] if dv_all else dvw[col + "_start"]]
            if len(idx):
                fig.add_scatter(
                    x=idx, y=mm.loc[idx, "A1"], mode="markers", name=lab,
                    marker=dict(symbol=sym, size=12, color=colr, line=dict(color="white", width=1)),
                    customdata=np.stack([dvw.loc[idx, "dFII"], dvw.loc[idx, "dClient"]], axis=1),
                    hovertemplate="FII %{customdata[0]:+.1f} pts · Client %{customdata[1]:+.1f} pts"
                                  f"<extra>{'▲' if col == 'bull' else '▼'}</extra>")
        fig.update_layout(height=430, hovermode="x unified")
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
        fig.update_yaxes(title="Long share %", secondary_y=False)
        fig.update_yaxes(title="NIFTY", tickformat=",.0f", showgrid=False, secondary_y=True)
        show(st, fig)
        nb, nr = int(dvw["bull_start"].sum()), int(dvw["bear_start"].sum())
        st.caption(f"▲ green = FII long share up and Client long share down by ≥{dv_t:g} pts over {dv_n} "
                   f"sessions ({nb} episodes in view); ▼ red = the reverse ({nr} episodes). Tested on 12 years "
                   "of data (default settings), these marks describe what just happened: NIFTY had risen over the "
                   "same 3 sessions in about 4 of 5 ▲ cases and fallen in about 7 of 10 ▼ cases. They did not "
                   "predict the next 5–20 sessions, where NIFTY rose about as often as it normally does.")

        w4 = chart_header("Net index futures: FII, Pro and FII + Pro combined (vs NIFTY spot)", "win_fp", last)
        hh2 = in_window(hist.set_index("date"), w4)
        fii_net = hh2["FII_FIL"] - hh2["FII_FIS"]
        pro_net = hh2["Pro_FIL"] - hh2["Pro_FIS"]
        comb = fii_net + pro_net
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_bar(x=fii_net.index, y=fii_net, name="FII net", marker_color="#2e86c1", opacity=0.55,
                    hovertemplate="%{y:,.0f}<extra>FII net</extra>")
        fig.add_bar(x=pro_net.index, y=pro_net, name="Pro net", marker_color="#d68910", opacity=0.75,
                    hovertemplate="%{y:,.0f}<extra>Pro net</extra>")
        fig.add_scatter(x=comb.index, y=comb, name="FII + Pro net", line=dict(color="#1c2833", width=2.2),
                        hovertemplate="%{y:,.0f}<extra>FII + Pro</extra>")
        if not nifty.empty:
            nn = in_window(nifty[nifty.index >= m.index.min()], w4)
            fig.add_scatter(x=nn.index, y=nn, name="NIFTY spot", line=nifty_line, secondary_y=True,
                            hovertemplate="%{y:,.0f}<extra>NIFTY</extra>")
        fig.add_hline(y=0, line=dict(color="#95a5a6", width=1))
        fig.update_layout(height=430, hovermode="x unified", barmode="overlay", bargap=0)
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
        fig.update_yaxes(title="Net contracts", tickformat=",.0f", secondary_y=False)
        fig.update_yaxes(title="NIFTY", tickformat=",.0f", showgrid=False, secondary_y=True)
        show(st, fig)
        st.caption("Net = long − short index-futures contracts. FII + Pro is the mirror image of Client + DII "
                   "(all four groups add up to zero). Over 12 years Pro desks often took the other side of FIIs "
                   "day to day, so combining them partly cancels out and did not forecast NIFTY better than FII "
                   "alone. One modest pattern held in both 2015–20 and 2021–26: when Pro desks built net longs over "
                   "20 sessions, NIFTY tended to do slightly worse over the next 20 (and vice versa). Treat it as "
                   "context, not a signal. Contract counts jump when lot sizes change.")

        w3 = chart_header("FII net index futures (contracts) vs NIFTY spot - contracts affected by "
                          "lot-size changes", "win_a2", last)
        mm = in_window(m, w3)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_bar(x=mm.index, y=mm["A2"], name="FII net index futures",
                    marker_color=np.where(mm["A2"] >= 0, "#27ae60", "#c0392b"), opacity=0.75,
                    hovertemplate="%{y:,.0f} contracts<extra>FII net</extra>")
        if not nifty.empty:
            nn = in_window(nifty[nifty.index >= m.index.min()], w3)
            fig.add_scatter(x=nn.index, y=nn, name="NIFTY spot", line=dict(color="#1b4f72", width=2),
                            hovertemplate="%{y:,.0f}<extra>NIFTY</extra>", secondary_y=True)
        fig.update_layout(height=420, hovermode="x unified", bargap=0)
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])  # hide weekend gaps
        fig.update_yaxes(title="Net contracts", tickformat=",.0f", secondary_y=False)
        fig.update_yaxes(title="NIFTY", tickformat=",.0f", showgrid=False, secondary_y=True)
        show(st, fig)

    # ---------------- TAB 4
    with tabs[3]:
        st.markdown("Signal on day **T** is acted on at the close of **T+1** (NSE files come out after the close). "
                    "Short term = return T+1→T+6; medium term = T+1→T+21. Composite rows use your current "
                    "sidebar weights, so you can test changes here.")
        if nifty.empty:
            st.warning("NIFTY prices unavailable right now, so the backtest cannot run.")
        else:
            sig = sc.copy()
            sig["SHORT"], sig["MEDIUM"] = short, med
            try:
                bt = core.backtest(sig, nifty)
            except Exception as e:
                st.error(f"Backtest could not run: {e}")
                bt = None
            for hz in (["short", "medium"] if bt else []):
                st.subheader(f"{hz.title()} term · average forward return of all sessions: "
                             f"{bt[hz + '_base']:.2f}%")
                df = bt[hz]
                if df.empty:
                    st.caption("Not enough data.")
                    continue
                df = df.copy()
                df["metric"] = df["metric"].map(lambda k: core.METRIC_INFO.get(k, (k,))[0] if k in core.METRIC_INFO
                                                else f"▶ {k.title()} composite")
                st.dataframe(df.set_index("metric").style.format("{:.2f}", na_rep="–").format("{:,.0f}", subset=["n"])
                             .background_gradient(subset=["rank corr"], cmap="RdYlGn", vmin=-0.3, vmax=0.3),
                             width="stretch")
                fig = go.Figure()
                for _, r in df.iterrows():
                    fig.add_bar(name=r["metric"], x=["Score −2", "Score −1", "Score 0", "Score +1", "Score +2"],
                                y=[r[f"avg {b:+d}"] for b in [-2, -1, 0, 1, 2]])
                fig.update_layout(barmode="group", height=340, yaxis_title="avg forward return %", xaxis_type="category",
                                  title="A useful signal rises from −2 to +2")
                show(st, fig)
            st.info("How to read: **rank corr** above about +0.1 means higher scores were followed by better NIFTY "
                    "returns; near 0 means no edge; negative means the signal worked the opposite way. "
                    "Overlapping windows inflate apparent significance — treat results as indicative.")

    # ---------------- TAB 5
    with tabs[4]:
        bad = m[m["issues"] != ""]
        c1, c2, c3 = st.columns(3)
        c1.metric("Sessions", len(m))
        c2.metric("Failed data checks", len(bad))
        c3.metric("Expiry-window sessions", int(m["flag_expiry"].sum()))
        if len(bad):
            st.dataframe(bad[["issues"]])
        st.download_button("⬇️ Download metrics (CSV)", m.join(sc, rsuffix="_score").to_csv().encode(),
                           "participant_bias_metrics.csv", "text/csv")
        if owner:
            st.download_button("⬇️ Download raw NSE history (CSV)", hist.to_csv(index=False).encode(),
                               "participant_raw_history.csv", "text/csv")
        st.markdown("""
**Sources** — NSE archives: `fao_participant_oi_DDMMYYYY.csv`, `fao_participant_vol_DDMMYYYY.csv`,
`fii_stats_DD-Mon-YYYY.xls`; NIFTY close from Yahoo Finance (^NSEI).

**Checks** — for every day, TOTAL long must equal TOTAL short in each segment and the four groups must add up to TOTAL.

**Scoring** — every metric is turned into a percentile rank over the look-back: ≥80 → +2, 60–80 → +1,
40–60 → 0, 20–40 → −1, ≤20 → −2. Build-up type (A4) is scored directly. Composite = weighted average (−2…+2).
Bias bands: ≥+1 strong bullish · +0.4…+1 bullish · −0.4…+0.4 neutral · −1…−0.4 bearish · ≤−1 strong bearish.

**Flags** — expiry = total index-futures OI falls >20% in a day (that day and the next are flagged);
structural break = OI jumps >40% in a day (possible lot-size change).

**Limits** — contract counts, not delta; the participant file mixes all indices (only the FII ₹ flow is
NIFTY-specific, and it is missing in older FII files); Client = everyone not FII/DII/Pro;
all weights, thresholds and alerts are hypotheses to be tested.
""")
        st.caption("© Dr Shantanu Samanta · For education and research only. Not investment advice; "
                   "not a SEBI-registered research recommendation.")


if __name__ == "__main__":
    main()
