"""
Participant F&O Bias Matrix - Streamlit app
Reads NSE participant-wise OI / volume and FII derivatives statistics,
scores them on the Participant F&O Bias Matrix framework and shows a
short-term (1-5 sessions) and medium-term (2-6 weeks) NIFTY bias.

© Dr Shantanu Samanta. For education and research only - not investment advice.
"""
import datetime as dt

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
    fig = go.Figure(go.Heatmap(z=z, x=[f"Medium: {m}" for m in rows], y=[f"Short: {s}" for s in rows],
                               text=text, texttemplate="%{text}", textfont={"size": 12},
                               colorscale=[[0, "#f4f6f7"], [1, "#f9e79f"]], showscale=False,
                               xgap=4, ygap=4, hoverinfo="skip"))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10),
                      title={"text": "Where today sits (highlighted)", "font": {"size": 14}})
    return fig


# ----------------------------------------------------------------- sidebar
def sidebar():
    info = core.history_info()
    st.sidebar.header("📁 Data (saved CSV)")
    if info["n"]:
        st.sidebar.markdown(f"**{info['n']:,} sessions** stored  \n{info['first']:%d %b %Y} → "
                            f"**{info['last']:%d %b %Y}**")
    else:
        st.sidebar.warning("No data saved yet - press Fetch.")
    if st.sidebar.button("📥 Fetch new days from NSE", type="primary", width="stretch",
                         help="Downloads only the trading days after the last stored session, "
                              "adds NIFTY closes, and saves to data/history.csv."):
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
    st.sidebar.caption("On Streamlit Cloud the saved CSV resets when the app restarts. After fetching, "
                       "download history.csv and replace data/history.csv in GitHub to keep it.")

    st.sidebar.header("⚙️ Analysis settings")
    years = st.sidebar.select_slider("History to analyse", options=[1.5, 2, 3, 5, 8, 11], value=2,
                                     format_func=lambda y: f"{y} years",
                                     help="Counted back from the latest stored session.")
    window = st.sidebar.select_slider("Percentile look-back (sessions)", options=[126, 189, 252, 504], value=252)
    with st.sidebar.expander("Short-term weights"):
        ws = {k: st.slider(core.METRIC_INFO[k][0], 0, 50, v, 5, key=f"s_{k}")
              for k, v in core.DEFAULT_WEIGHTS["short"].items()}
    with st.sidebar.expander("Medium-term weights"):
        wm = {k: st.slider(core.METRIC_INFO[k][0], 0, 50, v, 5, key=f"m_{k}")
              for k, v in core.DEFAULT_WEIGHTS["medium"].items()}
    st.sidebar.caption("Weights are starting guesses - check the Backtest tab before trusting them.")
    return info, years, window, ws, wm


# ----------------------------------------------------------------- main
def main():
    st.title("🧭 Participant F&O Bias Matrix")
    st.caption("NIFTY short-term (1–5 sessions) and medium-term (2–6 weeks) bias from NSE participant-wise "
               "F&O positions and FII derivatives statistics. Education and research only — not investment advice.")
    info, years, window, ws, wm = sidebar()

    msg = st.session_state.pop("fetch_msg", None)
    if msg:
        (st.success if msg[0] == "success" else st.error)(msg[1])

    if not info["n"]:
        st.info("No saved data yet. Press **📥 Fetch new days from NSE** in the sidebar "
                "(or add files under 'Add data from files').")
        return
    full = load_history(info["mtime"])
    start = full["date"].max() - pd.Timedelta(days=int(years * 365.25))
    hist = full[full["date"] >= start].reset_index(drop=True)
    if len(hist) < 30:
        st.error("Fewer than 30 sessions saved - fetch or add more data.")
        return
    stale = (pd.Timestamp(dt.date.today()) - full["date"].max()).days
    if stale > 3:
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
            col.plotly_chart(gauge(val, ttl), width="stretch")
        c3.plotly_chart(cross_matrix(core.bucket3(s_val), core.bucket3(m_val)), width="stretch")

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
        c1.plotly_chart(fig, width="stretch")

        share = [h[f"{c}_FIL"] / (h[f"{c}_FIL"] + h[f"{c}_FIS"]) * 100 for c in cats]
        fig = go.Figure(go.Bar(x=share, y=cats, orientation="h", marker_color=[CAT_COL[c] for c in cats],
                               text=[f"{s:.1f}%" for s in share], textposition="outside"))
        fig.add_vline(x=50, line_dash="dash", line_color="grey")
        fig.update_layout(title="Long share of each group's index-futures book", xaxis_range=[0, 100], height=380)
        c2.plotly_chart(fig, width="stretch")

        c1, c2 = st.columns(2)
        for col, cat in [(c1, "FII"), (c2, "Pro")]:
            vals = [h[f"{cat}_OICL"], h[f"{cat}_OIPS"], -h[f"{cat}_OICS"], -h[f"{cat}_OIPL"]]
            fig = go.Figure(go.Bar(x=["Call long", "Put short", "Call short", "Put long"], y=vals,
                                   marker_color=["#27ae60", "#82e0aa", "#e67e22", "#c0392b"],
                                   text=[f"{abs(v):,.0f}" for v in vals], textposition="outside"))
            fig.update_layout(title=f"{cat} index options (bullish legs up, bearish legs down)", height=360)
            col.plotly_chart(fig, width="stretch")
        st.caption("Option counts are contracts, not delta-weighted: a far out-of-the-money option counts the same "
                   "as an at-the-money one. Every long has a short somewhere, so the four groups' nets add to zero.")

    # ---------------- TAB 3
    with tabs[2]:
        # NIFTY on its own panel (top) so it is clearly visible; composites below on a shared date axis
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.45, 0.55],
                            subplot_titles=("NIFTY 50 spot (close)", "Composite bias (−2 … +2)"))
        if not nifty.empty:
            nn = nifty[nifty.index >= m.index.min()]
            fig.add_scatter(x=nn.index, y=nn, name="NIFTY spot", line=dict(color="#1b4f72", width=2.4),
                            fill="tozeroy", fillcolor="rgba(46,134,193,0.10)",
                            hovertemplate="%{y:,.0f}<extra>NIFTY</extra>", row=1, col=1)
            pad = (nn.max() - nn.min()) * 0.05
            fig.update_yaxes(range=[nn.min() - pad, nn.max() + pad], tickformat=",.0f", row=1, col=1)
        fig.add_scatter(x=short.index, y=short, name="Short-term composite",
                        line=dict(color="#e67e22", width=1.3), opacity=0.85, row=2, col=1)
        fig.add_scatter(x=med.index, y=med, name="Medium-term composite",
                        line=dict(color="#2e86c1", width=2.4), row=2, col=1)
        for y0, y1, c in [(1, 2, "#82e0aa"), (0.4, 1, "#d5f5e3"), (-1, -0.4, "#fdebd0"), (-2, -1, "#f5b7b1")]:
            fig.add_hrect(y0=y0, y1=y1, fillcolor=c, opacity=0.35, line_width=0, row=2, col=1)
        fig.update_yaxes(range=[-2, 2], row=2, col=1)
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
        fig.update_layout(title="Composite bias history vs NIFTY", height=620, hovermode="x unified",
                          legend=dict(orientation="h", y=1.07, x=1, xanchor="right"))
        st.plotly_chart(fig, width="stretch")

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

        c1, c2 = st.columns(2)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_scatter(x=m.index, y=m["A1"], name="FII long share %", line=dict(color="#2e86c1"))
        fig.add_scatter(x=m.index, y=m["A5"], name="Client long share %", line=dict(color="#17a589", dash="dot"))
        if not nifty.empty:
            fig.add_scatter(x=nifty.index, y=nifty, name="NIFTY", line=dict(color="#bdc3c7"), secondary_y=True)
        fig.update_layout(title="Who is long the index?", height=380, hovermode="x unified")
        c1.plotly_chart(fig, width="stretch")

        e = m["E_day"].dropna()
        fig = go.Figure(go.Bar(x=e.index, y=e, marker_color=np.where(e >= 0, "#27ae60", "#c0392b"), name="Daily"))
        fig.add_scatter(x=m.index, y=m["E2_20"], name="20-session sum", line=dict(color="#2c3e50"))
        fig.update_layout(title="FII NIFTY futures net ₹ crore (buy − sell)", height=380, hovermode="x unified")
        c2.plotly_chart(fig, width="stretch")

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_bar(x=m.index, y=m["A2"], name="FII net index futures",
                    marker_color=np.where(m["A2"] >= 0, "#27ae60", "#c0392b"), opacity=0.75,
                    hovertemplate="%{y:,.0f} contracts<extra>FII net</extra>")
        if not nifty.empty:
            nn = nifty[nifty.index >= m.index.min()]
            fig.add_scatter(x=nn.index, y=nn, name="NIFTY spot", line=dict(color="#1b4f72", width=2),
                            hovertemplate="%{y:,.0f}<extra>NIFTY</extra>", secondary_y=True)
        fig.update_layout(title="FII net index futures (contracts) vs NIFTY spot - contracts affected by "
                                "lot-size changes", height=420, hovermode="x unified", bargap=0,
                          legend=dict(orientation="h", y=1.08, x=1, xanchor="right"))
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])  # hide weekend gaps
        fig.update_yaxes(title="Net contracts", tickformat=",.0f", secondary_y=False)
        fig.update_yaxes(title="NIFTY", tickformat=",.0f", showgrid=False, secondary_y=True)
        st.plotly_chart(fig, width="stretch")

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
                st.plotly_chart(fig, width="stretch")
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
