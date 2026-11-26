# 🧭 Participant F&O Bias Matrix

A Streamlit app that collects NSE's daily participant-wise F&O data (Client, DII, FII, Pro) and the FII
derivatives statistics, scores them on the Participant F&O Bias Matrix framework, and shows a
**short-term (1–5 sessions)** and **medium-term (2–6 weeks)** NIFTY bias with colour-coded infographics,
alerts and a built-in backtest.

© Dr Shantanu Samanta · Education and research only — not investment advice.

## Files

| File | What it does |
|---|---|
| `app.py` | The Streamlit interface: gauges, matrix, scorecard, charts, backtest |
| `core.py` | Fetches, parses and checks NSE files; computes metrics, scores, composites, alerts, backtest |
| `data/history.csv` | Pre-loaded NSE history (Jan 2015 – 25 Nov 2026) so the app opens instantly |
| `data/missing_dates.txt` | Known holidays / non-trading days (so they are not re-fetched) |
| `requirements.txt` | Python packages |
| `.streamlit/config.toml` | Light theme |
| `docs/Participant_FO_Bias_Matrix_User_Manual.pdf` | Colour-coded user manual |

No API keys are needed.

## Data sources

* `https://nsearchives.nseindia.com/content/nsccl/fao_participant_oi_DDMMYYYY.csv`
* `https://nsearchives.nseindia.com/content/nsccl/fao_participant_vol_DDMMYYYY.csv`
* `https://nsearchives.nseindia.com/content/fo/fii_stats_DD-Mon-YYYY.xls`
* NIFTY 50 close from Yahoo Finance (`^NSEI`) — charts and backtest only

**Data fetching is manual.** The app only reads the saved `data/history.csv`; it makes no network calls
until you press **📥 Fetch new days from NSE** in the sidebar. That button downloads the trading days after
the last stored session (NSE publishes after ~7 PM IST), adds NIFTY closes, and saves them to the CSV.

Sidebar data tools:

| Control | What it does |
|---|---|
| 📥 Fetch new days from NSE | Adds every missing trading day up to today |
| Backfill / re-fetch a date range | Fetches a chosen period (e.g. older history) |
| ⬇️ Download history.csv | Saves the up-to-date CSV to your computer |
| Add data from files | Merges a saved `history.csv` and/or NSE files (original names) into the CSV; uploaded values fill in, never blank, stored ones |

If NSE ever blocks the cloud server, download the files from nseindia.com and use **Add data from files**.

## Deploy on Streamlit Community Cloud (≈5 minutes)

1. On github.com click **New**, name the repo (e.g. `participant-bias-matrix`), make it Public, **Create**.
2. Click **uploading an existing file**, drag in **all** files and folders (including `data/` and `.streamlit/`),
   click **Commit changes**.
3. Go to share.streamlit.io → **New app** → pick the repo → Main file path: `app.py` → **Deploy**.

To keep it awake, add the app URL to your `streamlit-keep-awake` repo.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Keeping the saved CSV (important on Streamlit Cloud)

Streamlit Cloud's disk resets whenever the app restarts, which rolls the CSV back to the copy in GitHub.
After fetching, press **⬇️ Download history.csv** and upload it to GitHub as `data/history.csv`
(replace the old file). Running locally, the CSV on your computer is updated directly.

## Updating the app

Edit / replace files in GitHub → the app redeploys automatically.
