# Free AI + Algorithmic Trading Agent: Build Guide

A hybrid AI + algorithmic **paper-trading** research system for NSE (NIFTY 50, BANK NIFTY and NSE stocks), built only from free and open-source tools. It lives in the `quant/` package next to the existing `arena/` app and reuses arena's tested Indian-charges model and real-money guard.

> **Paper trading only.** No code path can place a real order: broker order URLs and broker SDK imports are blocked (`arena/safety.py`), the simulator asserts the paper-only constant, and every entry point refuses to start if `paper_only` is switched off. Nothing here is investment advice. Simulated results do not predict future returns.

---

## 1. Architecture

```
                   ┌──────────────────── free data ────────────────────┐
                   │ Yahoo Finance (yfinance) · Kaggle/your CSV files    │
                   └───────────────┬────────────────────────────────────┘
                                   ▼
 (1) Data Collection Agent ── fetch, clean, quality-check ──► SQLite: bars (+source, fetched_at)
                                   │
 (8) Paper Trading Agent: sync ── fill yesterday's queued orders at today's OPEN, run stops/targets, mark to CLOSE
                                   │
 (2) Market Trend Agent ─── NIFTY / BANK NIFTY regime, breadth           ┐
 (3) Technical Analysis Agent ── RSI, MACD, EMA/SMA, VWAP, BB, ATR, ADX   │  describe only
 (4) ML Prediction Agent ── walk-forward-validated model, P(up in 5 days) ┘
                                   │
 (5) Strategy Generation Agent ── fixed, backtested hybrid rule ─► PROPOSALS
          └─ optional local LLM (Ollama) reviews BUY proposals: agree / caution / VETO only
                                   │
 (6) Risk Management Agent ── deterministic code: size, stop, target, limits ─► APPROVED / REJECTED
                                   │
 (8) Paper Trading Agent: execute ── queue approved orders for the NEXT open (virtual account)
 (7) Backtesting Agent ── same simulator, same costs, same risk rules, historical windows
 (9) Performance Evaluation Agent ── return, CAGR, drawdown, Sharpe, Sortino, win rate, costs
                                   │
                         Streamlit + Plotly dashboard
```

Orchestration uses a **LangGraph** `StateGraph` (`quant/agents.py`). If LangGraph is missing, the same node functions run sequentially with the same results.

### Who is allowed to do what

| Component | May | May not |
|---|---|---|
| Trend / Technical / ML agents | compute and describe | propose or place orders |
| Local LLM (Ollama) | comment; **veto** a new entry | create orders, change size, stops or limits, override a rejection |
| Strategy agent | turn the backtested rule into proposals | size positions or touch the account |
| Risk agent + `RiskManager` | approve, size, reject; pause trading | be bypassed: the simulator re-checks every entry at the fill |
| Paper trading agent | change the virtual account with approved orders | send anything to a broker (no broker code exists) |

### Timing rules (no look-ahead)

1. Decisions are made on the **close** of day *t* using only bars up to *t*.
2. Orders fill at the **open** of day *t+1*, plus slippage, plus full Indian charges.
3. Stops and targets are checked on later days' high/low. If both are touched on the same day, the **stop** is assumed first. A gap through the stop fills at the (worse) open.
4. ML labels are "buy at the next open, sell `h` sessions later at the open"; walk-forward training **purges** the `h+1` rows whose labels overlap each test block.
5. LLMs are **excluded from backtests**: a model trained on internet text from after the test period already knows what happened.

## 2. Folder structure

```
claudemm/
├── quant.toml                 all settings (watchlist, costs, risk, ML, paper, LLM)
├── requirements-quant.txt     free/open-source dependencies
├── quant/
│   ├── __main__.py            CLI: python -m quant <command>
│   ├── config.py              defaults + quant.toml loader
│   ├── clock.py               IST calendar helpers
│   ├── data/
│   │   ├── sources.py         YahooSource (retries, backoff, throttle), CSVSource (Kaggle etc.)
│   │   ├── store.py           SQLite: bars, fetch_log, models, predictions, backtests
│   │   ├── quality.py         clean(), adjusted(), report(): gaps, stale bars, bad prints
│   │   └── synthetic.py       test-only prices, labelled SYNTHETIC, refused by the store
│   ├── indicators.py          SMA, EMA, RSI, MACD, Bollinger, ATR, ADX/DI, VWAP
│   ├── strategies.py          MA crossover, RSI, MACD, trend following, composite algo, ML, hybrid
│   ├── costs.py               slippage + Indian charges (arena/charges.py)
│   ├── risk.py                RiskManager: sizing, stop, target, R:R, daily loss, drawdown, cooldown
│   ├── sim.py                 PortfolioSim shared by backtests AND paper trading
│   ├── backtest.py            portfolio backtests + buy-and-hold benchmark
│   ├── metrics.py             return, CAGR, drawdown, Sharpe, Sortino, win rate, profit factor
│   ├── ml/
│   │   ├── features.py        causal features + execution-aligned labels
│   │   ├── models.py          logistic, random forest, gradient boosting, XGBoost
│   │   └── walkforward.py     purged expanding-window walk-forward validation
│   ├── pipeline.py            fetch / prepare / train / compare glue
│   ├── llm.py                 optional Ollama advisor (agree / caution / veto)
│   ├── agents.py              the nine agents + LangGraph orchestration
│   ├── paper.py               persistent paper account (SQLite)
│   ├── live.py                live market-hours session (1-minute bars, stops every minute)
│   └── dashboard.py           Streamlit dashboard
├── AUTOPILOT.bat              runs one full paper-trading day (Windows)
├── SCHEDULE-AUTOPILOT.bat     creates the weekday Task Scheduler job (Windows)
├── tests/test_quant.py        offline tests (indicators, causality, fills, risk, ML leakage, agents)
├── data/                      created at runtime (git-ignored): quant.db, paper.db, paper_replay.db
└── docs/QUANT_GUIDE.md        this file
```

## 3. Free data acquisition strategy

| Need | Free source used | Limits you must know |
|---|---|---|
| NIFTY 50 / BANK NIFTY daily | Yahoo `^NSEI`, `^NSEBANK` | Unofficial; no reliable volume for indices; history from 2007 |
| Tradable index exposure | Yahoo `NIFTYBEES.NS`, `BANKBEES.NS` (ETFs) | An index cannot be bought; ETFs track it with small tracking error |
| NSE stocks daily | Yahoo `SYMBOL.NS` | Occasional bad prints and gaps (the cleaner removes and reports them) |
| Intraday | Yahoo 1m / 5m / 15m / 60m | 1m: ~7 days; 2m–30m: 60 days; 60m: ~730 days. Too short for serious ML; quotes delayed |
| Other history | Kaggle datasets, files you are allowed to download | `python -m quant import-csv FILE SYMBOL`; check each dataset's licence |
| Options chain history, tick data | **No free source** | Not assumed anywhere in this project |
| Point-in-time index constituents | **No free source** | Survivorship bias is reported, not hidden |
| Live quotes | Yahoo (delayed) | Not exchange-licensed; labelled DELAYED |

Deliberate exclusions: no scraping of nseindia.com (its terms prohibit automated access), and no Zerodha Kite data (its terms restrict virtual/mock-trading use). The Upstox free API used by `arena` is an option for official data if you have an account; `quant` stays on Yahoo so it needs no account at all.

How the data layer stays honest:

- Every bar is stored with its **source** and **fetched_at** timestamp; the dashboard prints both next to every chart.
- Failures are retried with exponential backoff and a 1-request-per-second throttle, then logged in `fetch_log`. Missing data stays missing. It is never forward-filled or invented.
- `quality.clean()` removes duplicates, non-positive prices and **bad prints**: a jump of more than 25% that reverts within 3 bars. It found real ones: NIFTYBEES and BANKBEES traded at 1/10th scale on 19–20 Dec 2019 in Yahoo's data. Genuine big moves (INFY −21% on 12 Apr 2013, SBIN +27% on 25 Oct 2017) are kept and reported.
- Prices are back-adjusted for splits and dividends with Yahoo's `adj_close` ratio before indicators and backtests.
- Stale data is flagged when the newest bar is older than the last expected weekday session (a holiday is mentioned as the possible reason, since NSE holidays are not free machine-readable data).

---

## 4. Phases

Commands below use the project virtualenv. On Windows replace `source .venv/bin/activate` with `.venv\Scripts\activate`.

### Phase 1: Environment setup and architecture

**Concepts.** A reproducible, isolated Python environment; a single config file; a safety guard that loads before anything else.

```bash
git clone <this repository> && cd claudemm
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-quant.txt
python -m quant check
```

**Verify.** Expected output (values from the run on 25 Nov 2026):

```
Python 3.13.x (ok)
  ok   pandas, numpy, sklearn, yfinance, requests, xgboost, langgraph, streamlit, plotly
  ok   real-order URLs are blocked (paper trading only)
  ok   SQLite store at .../data/quant.db
  ok   Yahoo Finance reachable: ^NSEI last close 25819.25 on 2026-11-25 (end-of-day)
  opt  Ollama not running or model missing (optional)
All required checks passed.
```

Optional local LLM: install Ollama from ollama.com, then `ollama pull llama3.1:8b` (or `qwen2.5:7b`). Set `[llm] model` in `quant.toml` to the model you pulled.

**Common errors.**
- `Python 3.11+ is required` / `tomllib` not found: install Python 3.11 or newer.
- `Yahoo Finance: ... CONNECT tunnel failed` or `possibly delisted; no price data found` for every symbol: your network or proxy blocks `query1/query2.finance.yahoo.com` and `fc.yahoo.com`. Allow them, or use a different network.
- `YFRateLimitError`: Yahoo is throttling you. Wait a few minutes; the fetcher already retries with backoff.

### Phase 2: Free historical data collection

```bash
python -m quant fetch                      # daily bars from 2012 for indices + watchlist
python -m quant fetch --interval 5m        # optional: last 60 days of 5-minute bars
python -m quant import-csv my_kaggle.csv TATAMOTORS.NS
```

`YahooSource.fetch()` downloads, normalises columns and timezones (IST), and retries; `Store.upsert_bars()` writes idempotently, so re-running only refreshes. **Verify:** each symbol shows `ok` with ~3,670 rows ending on the last trading day.

### Phase 3: Cleaning and preprocessing

```bash
python -m quant quality
```

Prints every symbol's status (`OK`, `STALE`, `MISSING`), date range, data kind (HISTORICAL end-of-day / DELAYED / HISTORICAL intraday) and warnings, e.g. `Cleaning: 2 bad prints removed ... 2019-12-19, 2019-12-20`. **Verify:** no `STALE` rows after a fresh fetch on a trading day.

### Phase 4: Technical indicators

```bash
python -m quant indicators ^NSEI --rows 3
```

All indicators are implemented in pandas (`quant/indicators.py`). RSI, ATR and ADX use Wilder smoothing (alpha = 1/n), matching common charting platforms after a warm-up of about three times the period. Daily VWAP is a rolling 20-day VWAP, because a true session VWAP needs intraday data, which free sources keep only for weeks; for 5-minute data `vwap(df)` anchors to each session. **Verify:** the test `test_indicators_and_features_are_causal` recomputes every indicator on truncated history and checks the value at each bar is unchanged.

### Phase 5: Algorithmic strategies

`quant/strategies.py` returns `entry` / `exit` boolean series:

| Name | Rule |
|---|---|
| `ma_crossover` | EMA21 crosses above EMA50; exit on the cross below |
| `rsi_reversion` | RSI crosses up through 30 while above SMA200; exit RSI ≥ 55 |
| `macd_trend` | MACD crosses its signal line, confirmed by close > EMA200 |
| `trend_following` | close > EMA50 > EMA200, ADX > 20, +DI > −DI; exit close < EMA50 |
| `algo` (composite) | EMA21 > EMA50, close > EMA200, MACD hist > 0, RSI 45–70, **volume ≥ 20-day average**, **ATR% below its 85th percentile** |
| `ml` | walk-forward P(up) ≥ 0.55 |
| `hybrid` | `algo` setup **and** P(up) ≥ 0.50; exit on algo exit or P(up) < 0.40 |

### Phase 6: Backtesting

```bash
python -m quant backtest --strategy ma_crossover --start 2015-01-01
```

One `PortfolioSim` (`quant/sim.py`) is used for backtests **and** paper trading, so the paper account follows exactly the rules that were tested. Costs per fill: 5 bps slippage, tick rounding, brokerage, STT, NSE transaction fee, SEBI fee, stamp duty, GST and DP charges (delivery). Every backtest prints an equal-weight **buy-and-hold** benchmark with the same costs.

Why a custom engine instead of Backtrader/VectorBT: neither models Indian delivery charges or the next-open fill and risk-manager re-check the paper account needs, and sharing one engine between backtests and paper trading removes a whole class of "backtest ≠ live" bugs. Both libraries are free and can still be used to cross-check simple strategies.

### Phase 7: Machine learning

```bash
python -m quant train --models logistic,random_forest,gradient_boosting,xgboost
python -m quant train --task regression --models gradient_boosting
```

- **Features** (`ml/features.py`): returns over 1/5/10/20 days, volatility, RSI, normalised MACD histogram, distance to EMA21/50/200, ADX and DI spread, ATR%, Bollinger %B and width, volume ratio, distance to 52-week high/low, the algo-setup flag, plus NIFTY context (20-day return, distance to EMA200, ADX).
- **Labels**: buy at the next open, sell 5 sessions later at the open; "up" means beating ~0.3% round-trip costs.
- **Validation**: purged, expanding-window walk-forward (3 years minimum training, retrain every quarter). No random splits anywhere.
- **Overfitting checks**: train vs test AUC per fold (`overfit_gap`), comparison with the base rate, small regularised models, fixed seeds. A test plants a feature equal to the future label to prove the pipeline would detect leakage (AUC > 0.95), and another confirms a random walk scores AUC ≈ 0.5.
- PyTorch is deliberately not used: with ~250 rows a year of tabular features, trees and logistic regression are the stronger baselines. Add a neural model only if it beats them out of sample.

### Phase 8: AI multi-agent integration

```bash
python -m quant paper run              # fetch -> 9 agents -> queue orders for the next open
python -m quant paper run --no-llm     # skip the local LLM
```

Each run is saved (`agent_runs` table) and shown on the dashboard's **Agent analysis** tab.

### Phase 9: Risk management (`quant/risk.py`)

| Control | Default | Behaviour |
|---|---|---|
| Position sizing | 1% of equity at risk | qty = min(risk ÷ stop distance, 25% of equity ÷ price, cash) |
| Stop-loss | 2 × ATR(14) | from the actual fill price |
| Take-profit | 2 × stop distance | reward/risk 2:1 |
| Minimum reward/risk | 1.5 | otherwise rejected |
| Max daily loss | 2% | no new entries next session |
| Max drawdown | 15% | circuit breaker: no new entries for 20 sessions, then restart from the new equity level |
| Cooldown | 3 sessions | after a losing trade in that stock |
| Max open positions | 5 | |

Exits are never blocked. The circuit breaker was changed from "wait until equity recovers" after testing showed that a flat account can never recover, which had frozen a backtest permanently in 2015.

### Phase 10: Paper trading

```bash
python -m quant paper reset --capital 1000000
python -m quant paper run          # every trading day after 15:30 IST (schedule it with cron / Task Scheduler)
python -m quant paper status
python -m quant paper replay --start 2026-06-01   # forward-test on sessions the models never saw
```

Orders queued tonight fill at the next session's open when you run it the next evening; skipped days are caught up with the same deterministic rules. The account (cash, positions, pending orders, risk state, trade history, equity) lives in `data/paper.db` and survives restarts.

`paper replay` retrains models on data **before** the start date only, then runs the full LangGraph agent cycle once per past session, each time seeing only data up to that evening. It writes to `data/paper_replay.db`, labelled as a historical simulation.

### Phase 10b: Live session and daily automation

The trading day has two halves:

| When | Command | What happens |
|---|---|---|
| 09:15–15:30 IST | `python -m quant live` | Orders queued last evening fill at the **first 1-minute bar's open** (plus slippage and charges). Every minute, new 1-minute bars are checked against each position's stop-loss and target. At 15:30 the account is marked to the last prices and the day is closed. |
| After 15:30 IST | `python -m quant daily` | Waits for Yahoo to publish today's daily bar, **retrains** the model on all data up to today (walk-forward + final model), then runs the nine agents, which queue tomorrow's orders. |
| Both | `python -m quant autopilot` | Runs `live`, waits until 15:45, then runs `daily`. Weekends do nothing; a holiday ends the live part at 10:00 when no bars arrive. |

Why new decisions are only made after the close: the strategies and models were tested on daily bars. Deciding mid-session from a half-finished bar would trade a strategy that was never tested. During the session, only exits (stops and targets) react to live prices.

Live prices come from Yahoo 1-minute bars: free, unofficial and **delayed**. The dashboard's Paper portfolio tab shows each live price with its bar time and a DELAYED label. If you start late, queued orders fill at the latest price instead of the open, and the log says so. The day's bar is never treated as final before 15:30 (an unfinished bar would be look-ahead).

**Retraining every day** keeps the model current. It adds one new day of data, so it does not make the model accurate by itself: walk-forward AUC stays the honest measure of skill.

**Windows: run it automatically every weekday**

1. Double-click `SCHEDULE-AUTOPILOT.bat` once. It creates a Task Scheduler job, "Quant Paper Autopilot", that runs `AUTOPILOT.bat` at 09:00 Monday–Friday (your computer's clock must be on IST).
2. Keep the computer on, awake and logged in from 09:00 to about 16:15 (Settings → System → Power → Sleep: Never while plugged in).
3. Read `data\autopilot.log` or open the dashboard to see what happened.
4. Remove the job any time: `schtasks /Delete /TN "Quant Paper Autopilot" /F`.

**Mac / Linux:** add a cron line (`crontab -e`) on a machine set to IST:

```
0 9 * * 1-5 cd /path/to/claudemm && .venv/bin/python -m quant autopilot >> data/autopilot.log 2>&1
```

### Phase 11: Dashboard and integration

```bash
python -m quant dashboard          # http://127.0.0.1:8501
```

Tabs: Market overview (NIFTY / BANK NIFTY, data-quality table), Charts & indicators (candles, EMA, Bollinger, VWAP, RSI, MACD, ADX), Signals & AI (rule states and ML probabilities with their walk-forward AUC), Agent analysis, Paper portfolio (live or replay), Backtests (metrics, equity and drawdown curves) and Risk. Each panel shows source, last bar date, fetch time and data kind, with warnings for missing, stale or delayed data.

---

## 5. Results from the verification run (real Yahoo data, fetched 25 Nov 2026)

Watchlist: NIFTYBEES, BANKBEES, RELIANCE, HDFCBANK, ICICIBANK, INFY, TCS, SBIN. ₹10,00,000 virtual. Same out-of-sample window for every strategy (17 Dec 2015 – 25 Nov 2026), same costs, same risk rules.

| Strategy | Total return | CAGR | Max drawdown | Sharpe | Trades | Charges paid |
|---|---|---|---|---|---|---|
| Buy and hold (equal weight) | 352.1% | 14.8% | 40.0% | 0.90 | 8 | ₹5,991 |
| MA crossover | 77.2% | 5.4% | 11.4% | 0.85 | 184 | ₹1,25,950 |
| Composite algo | 33.8% | 2.7% | 16.4% | 0.35 | 412 | ₹2,46,224 |
| ML only (gradient boosting) | −11.0% | −1.1% | 30.8% | −0.04 | 1,041 | ₹4,32,648 |
| Hybrid (algo + ML filter) | −6.6% | −0.6% | 17.4% | −0.10 | 258 | ₹1,32,446 |

Walk-forward ML (out of sample, 5-day horizon): AUC between 0.46 and 0.54 for every model and stock, against train AUC of about 0.85 for the tree models.

What this honestly says:

1. **No strategy beat buy-and-hold on return** over this bull-market decade. The rule-based strategies did cut the maximum drawdown from 40% to 11–16%, at a large cost in return.
2. **Costs matter**: delivery STT (0.1% each side) and friends cost the composite strategy about 25% of starting capital over ten years.
3. **The ML models show no real out-of-sample skill** at predicting 5-day direction from price-based features (AUC ≈ 0.5), so the ML filter made the hybrid worse than the plain rules. This is the expected result for liquid large caps, and it is what a leak-free pipeline should report. Picking the "best" model from this table would itself be overfitting to the test period.
4. **Paper replay** (1 Jun – 25 Nov 2026, models trained through 29 May): +0.6% after ₹5,027 of costs, 6 trades, max drawdown 2.6%. The composite algo returned −2.1% and buy-and-hold +12.9% over the same window. Six months and six trades are far too few to conclude anything.

## 6. Where to go next

- Improve the signal before adding complexity: longer horizons (20 days), cross-sectional ranking across a broader universe, sector/market-regime features, and probability calibration.
- Reduce turnover: costs dominate short-horizon delivery trading.
- Use intraday data for execution studies only once you have a longer free history (start collecting 5-minute bars daily with `fetch --interval 5m`, as Yahoo keeps only 60 days).
- Add an NSE holiday list so stale-data checks skip holidays, and a broader point-in-time universe if you obtain a legitimate constituent history.
- Keep paper trading for months before trusting any number, and never switch this system to real money.
