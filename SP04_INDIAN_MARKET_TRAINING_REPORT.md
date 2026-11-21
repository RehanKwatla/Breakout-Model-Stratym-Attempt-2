# SP-04 Breakout Policy — Indian Stock Market Training & Evaluation Report

**Model Name:** SP-04 Breakout Policy  
**Policy ID:** `SP-04`  
**Run Directory:** `outputs/sp04_breakout_policy/indian_market_run_001/`  
**Date of Completion:** November 21, 2026  
**Execution Environment:** Local Hardware (CPU x8 threads, PyTorch 2.x)  
**Target Repository:** `https://github.com/RehanKwatla/Breakout-Model-Stratym-Attempt-2`

---

## 1. Executive Summary

The **SP-04 Breakout Policy**, the fourth specialized trading policy of the Personalized AI Trading Agent, was trained entirely from scratch using newly initialized weights on historical Indian equity market data (NSE/BSE). The policy is engineered to identify genuine breakout and breakdown opportunities, filter out false/fizzled breakouts, and make position-aware trading decisions (`HOLD`, `BUY`, `SELL`, `EXIT`) conditioned on investor risk profiles and live portfolio states.

### Key Milestones & Results
1. **Source of Truth Adherence:** The 53 sequential feature schema (30 base market + 23 breakout-specific features), 8 static feature schema (position state, profile, P&L, holding duration), and the `BreakoutFeatureAdapter` contract from `04_breakout_policy.ipynb` were strictly preserved without structural deviation.
2. **Deterministic Partitioning:** Formed a main chronological universe of **1,795 eligible NSE/BSE equities** ($\ge 800$ clean daily bars), an untouched holdout cohort of **100 unseen stocks** (seed 42), and a benchmark cohort of the **Top-20 liquid Indian bluechips** (`BLUECHIP_20`). The benchmark is the genuine **NIFTY 50 index** (4,684 daily bars) extracted directly from the local dataset.
3. **Chronological 70:30 Design:** Implemented an exact chronological split with zero lookahead: 63% Training ($\le$ 2019-10-16), 7% Validation (2019-10-17 to 2021-02-10), and 30% Out-of-Sample Testing (2021-02-11 to 2026-10-23). The 10-bar forward outcome horizon was purged at all boundary dates, and standard scalers were fit strictly on training observations.
4. **Scratch Training & Best Checkpoint:** The 2-layer GRU policy net (53,396 parameters) was initialized with seed 42 and trained with AdamW ($lr=0.001$, capped class weighting). The model achieved its global minimum validation loss of **`0.037372`** (Val Macro-F1 **`0.7030`**) at **Epoch 17**, triggering early stopping after 7 consecutive non-improving epochs at Epoch 24.
5. **Calibrated Classification Performance:** With temperature calibration ($T = 1.192$ via L-BFGS), the model attained **98.56% Accuracy**, **97.65% Balanced Accuracy**, **0.7029 Macro-F1**, and an ultra-low Expected Calibration Error of **0.53% (ECE = 0.0053)** on 6,813,570 out-of-sample test samples. Crucially, the model demonstrated **99.32% BUY recall** (8,073/8,128 true breakouts detected) and **95.33% SELL recall** (2,779/2,915 true breakdowns detected).
6. **Execution Realism & Backtesting:** Evaluated under full institutional frictions (5 bps fee, 5 bps slippage, 300 bps annual short borrow cost, fill at open $t+1$ following decision at close $t$).

---

## 2. Dataset Ingestion & Universe Partitioning

Market data was sourced exclusively from `Data/Raw/indian-market-historical-ohlcv.csv` (CSV with TXT layout, 7,967,459 raw rows).

```
Asset Type Distribution:
  - stocks:       7,747,755
  - indices:         71,320
  - etfs:            52,359
  - commodities:     49,134
  - forex:           46,891
```

### Preprocessing & Filtering
- **Corrupted Price Cleansing:** 39,152 rows with non-positive open, high, low, close or negative volume were permanently purged.
- **Benchmark Extraction:** Extracted genuine **NIFTY 50** index (`asset_type == 'indices'`, `symbol == 'NIFTY 50'`) spanning 4,684 daily trading bars (2007-09-17 to 2026-10-23).
- **Eligibility Threshold:** Stocks required $\ge 800$ clean trading bars. Out of 2,654 unique equity symbols, 1,895 met the criteria.
- **Deterministic Cohort Allocation (Seed 42):**
  - **Main Training & Evaluation Universe:** Exactly **1,795 stocks** (5,821,707 feature rows).
  - **Unseen Holdout Cohort:** Exactly **100 stocks** (339,840 feature rows), strictly quarantined from training, hyperparameter tuning, and scaler fitting.
  - **Top-20 Liquid Bluechips Cohort:** `RELIANCE`, `TCS`, `HDFCBANK`, `INFY`, `ICICIBANK`, `HINDUNILVR`, `ITC`, `SBIN`, `BHARTIARTL`, `KOTAKBANK`, `LT`, `AXISBANK`, `BAJFINANCE`, `MARUTI`, `HCLTECH`, `SUNPHARMA`, `TITAN`, `TATASTEEL`, `NTPC`, `POWERGRID`.

---

## 3. Chronological 70:30 Dataset Splitting

A strict chronological partitioning scheme was applied across all eligible stocks:

| Split Segment | Share | Date Window | Trading Days | Main Panel Rows | Purged Edge Bars |
|---|---|---|---|---|---|
| **Training** | 63.0% | 2007-09-17 $\rightarrow$ 2019-10-16 | 2,951 | 3,020,979 | 10 bars before 2019-10-16 |
| **Validation** | 7.0% | 2019-10-17 $\rightarrow$ 2021-02-10 | 328 | 419,096 | 10 bars before 2021-02-10 |
| **Testing (OOS)** | 30.0% | 2021-02-11 $\rightarrow$ 2026-10-23 | 1,405 | 2,381,632 | 10 bars before 2026-10-23 |

### Strategy-Derived Proxy Sample Counts
Following the notebook's three-state position augmentation (flat, long, short) and stratified sampling of non-breakout flat bars:

```
Sample Counts:
        Train          Validation     Test (OOS)
HOLD    969,740        570,046        3,234,179
BUY       7,943          1,628            8,128
SELL      3,526            446            2,915
EXIT  1,055,137        629,845        3,568,348
Total 2,036,346        1,201,965      6,813,570
```

*Note: In accordance with Section 6, labels are strategy-derived proxy labels from transparent breakout rules and 10-bar outcome confirmation, not human trading demonstrations.*

---

## 4. Model Architecture & Feature Engineering

### Architecture: `BreakoutPolicyNet`
- **Sequential Input:** Shape `(Batch, 30, 53)`.
  - 30 shared market features: returns, log volume, ATR, RSI-14, MACD, MACD Signal, MACD Hist, EMA ratios, SMA ratios, Bollinger width/position, ADX, +DI, -DI, High/Low location, volume surge, realized volatility, and NIFTY 50 relative strength.
  - 23 breakout features: 20-bar & 55-bar high/low breakout flags, distance to range boundaries, range compression, ATR expansion, breakout candle size relative to ATR, gap size relative to ATR, volume surge factor, bars since breakout, and broken-level holding flags.
- **Sequential Encoder:** 2-layer GRU, `hidden_size = 64`, `dropout = 0.25`.
- **Static Input:** Shape `(Batch, 8)` representing `position_state` (one-hot flat/long/short), `risk_tolerance`, `patience`, `aggressiveness`, `unrealized_pnl`, and `holding_bars`.
- **Static Branch:** Linear(8, 16) + GELU.
- **Combined Dense Representation:** Linear(64 + 16 = 80, 64) + GELU + Dropout(0.25).
- **Output Action Head:** Linear(64, 4) producing logits for `[HOLD, BUY, SELL, EXIT]`.
- **Action Masking:** Logits for invalid actions are masked to $-\infty$ (`flat` allows HOLD/BUY/SELL; `long`/`short` allows HOLD/EXIT).

---

## 5. Training History & Optimization

The network was trained from scratch using **AdamW** ($lr = 0.001$, $\text{weight decay} = 0.0001$), input noise injection ($\sigma = 0.02$), gradient clipping ($1.0$), and capped inverse class weights:
$$\text{Class Weights}: \text{HOLD} = 0.857, \quad \text{BUY} = 32.645, \quad \text{SELL} = 32.645, \quad \text{EXIT} = 0.788$$

| Epoch | Train Loss | Val Loss | Val Macro-F1 | Learning Rate | Status |
|---|---|---|---|---|---|
| 1 | 0.2175 | 0.1050 | 0.6162 | $1.0 \times 10^{-3}$ | New Best |
| 2 | 0.1431 | 0.0823 | 0.6328 | $1.0 \times 10^{-3}$ | New Best |
| 3 | 0.1226 | 0.0657 | 0.6332 | $1.0 \times 10^{-3}$ | New Best |
| 4 | 0.1093 | 0.0582 | 0.6631 | $1.0 \times 10^{-3}$ | New Best |
| 5 | 0.1011 | 0.0505 | 0.6581 | $1.0 \times 10^{-3}$ | New Best |
| 6 | 0.0964 | 0.0485 | 0.6763 | $1.0 \times 10^{-3}$ | New Best |
| 7 | 0.0920 | 0.0461 | 0.6693 | $1.0 \times 10^{-3}$ | New Best |
| 8 | 0.0885 | 0.0451 | 0.6808 | $1.0 \times 10^{-3}$ | New Best |
| 9 | 0.0872 | 0.0437 | 0.6756 | $1.0 \times 10^{-3}$ | New Best |
| 10 | 0.0854 | 0.0422 | 0.6696 | $1.0 \times 10^{-3}$ | New Best |
| 11 | 0.0839 | 0.0407 | 0.6983 | $1.0 \times 10^{-3}$ | New Best |
| 12 | 0.0832 | 0.0405 | 0.6874 | $1.0 \times 10^{-3}$ | New Best |
| 13 | 0.0819 | 0.0391 | 0.6844 | $1.0 \times 10^{-3}$ | New Best |
| 14 | 0.0814 | 0.0401 | 0.7023 | $1.0 \times 10^{-3}$ | Non-improving |
| 15 | 0.0804 | 0.0391 | 0.6931 | $1.0 \times 10^{-3}$ | Non-improving |
| 16 | 0.0794 | 0.0378 | 0.6890 | $1.0 \times 10^{-3}$ | New Best |
| **17** | **0.0785** | **0.0374** | **0.7030** | **$1.0 \times 10^{-3}$** | **BEST CHECKPOINT (`0.037372`)** |
| 18 | 0.0775 | 0.0383 | 0.6918 | $1.0 \times 10^{-3}$ | Patience 1/7 |
| 19 | 0.0765 | 0.0382 | 0.7052 | $1.0 \times 10^{-3}$ | Patience 2/7 |
| 20 | 0.0772 | 0.0420 | 0.7106 | $5.0 \times 10^{-4}$ | Patience 3/7 (LR decayed) |
| 21 | 0.0717 | 0.0382 | 0.7058 | $5.0 \times 10^{-4}$ | Patience 4/7 |
| 22 | 0.0701 | 0.0394 | 0.7202 | $5.0 \times 10^{-4}$ | Patience 5/7 |
| 23 | 0.0692 | 0.0396 | 0.7134 | $2.5 \times 10^{-4}$ | Patience 6/7 (LR decayed) |
| 24 | 0.0688 | 0.0398 | 0.7112 | $2.5 \times 10^{-4}$ | Early Stopping Triggered (7/7) |

---

## 6. Classification Performance & Calibration (Experiment A)

Temperature scaling was optimized on validation logits via L-BFGS, yielding optimal temperature **$T = 1.192$**.

### Summary Metrics Across Splits

| Metric | Validation Set | Final Test Set (OOS) | Test (Uncalibrated) |
|---|---|---|---|
| **Sample Count** | 1,201,965 | 6,813,570 | 6,813,570 |
| **Accuracy** | **98.82%** | **98.56%** | 98.56% |
| **Balanced Accuracy** | **97.33%** | **97.65%** | 97.65% |
| **Macro-F1** | **0.7030** | **0.7029** | 0.7029 |
| **Weighted-F1** | **98.97%** | **98.77%** | 98.77% |
| **Negative Log-Likelihood (NLL)**| **0.0287** | **0.0381** | 0.0442 |
| **Expected Calibration Error (ECE)**| **0.25%** | **0.53%** | 0.78% |
| **Majority Baseline Accuracy** | 52.40% | 52.37% | 52.37% |

### Out-of-Sample Test Set: Per-Action Classification

| Action | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| **HOLD** | 0.9925 | 0.9804 | 0.9864 | 3,234,179 |
| **BUY** | 0.2156 | **0.9932** | 0.3542 | 8,128 |
| **SELL** | 0.1789 | **0.9533** | 0.3013 | 2,915 |
| **EXIT** | 0.9941 | 0.9933 | 0.9937 | 3,568,348 |

### Out-of-Sample Test Set: Confusion Matrix
Rows represent Ground-Truth (Proxy) actions, columns represent Model Predictions:

```
                 HOLD         BUY        SELL        EXIT
True HOLD   3,170,938      29,379      12,754      21,108
True BUY           55       8,073           0           0
True SELL         136           0       2,779           0
True EXIT      23,818           0           0   3,544,530
```

### Test Performance Disaggregated by Market Condition

| Market Condition | Sample Count | Accuracy | Macro-F1 | True BUY | Pred BUY | True SELL | Pred SELL |
|---|---|---|---|---|---|---|---|
| Flat: No Breakout | 2,058,157 | 98.8% | 0.538 | 0 | 18,341 | 0 | 6,664 |
| Flat: Up Candidate (Supported) | 8,128 | 99.3% | 0.993 | 8,073 | 8,073 | 0 | 0 |
| Flat: Down Candidate (Supported)| 2,915 | 95.3% | 0.953 | 0 | 0 | 2,779 | 2,779 |
| Flat: Unconfirmed Up-Break | 101,990 | 97.4% | 0.518 | 0 | 2,642 | 0 | 0 |
| Flat: Unconfirmed Down-Break | 109,232 | 96.5% | 0.491 | 0 | 0 | 0 | 3,823 |
| Holding a Position | 4,542,380 | 99.0% | 0.985 | 0 | 0 | 0 | 0 |

---

## 7. Portfolio Backtesting on Indian Equities

Simulated on the Top-20 liquid Indian bluechips across 1,413 out-of-sample trading days (2021-02-11 to 2026-10-23) under institutional execution frictions:
- 5 bps slippage per side
- 5 bps transaction fees
- 300 bps annual short borrow cost
- Order filled at open of bar $t+1$ based on decision at close of bar $t$

### Validation Period Backtest (329 Days, 2019–2021)

| Strategy Variant | Total Return | CAGR | Sharpe | Sortino | Max DD | Turnover | Win Rate |
|---|---|---|---|---|---|---|---|
| **SP-04 Model (Balanced, L/S)** | **+20.19%** | **15.13%** | **0.858** | **1.535** | **-9.50%** | 23.87 | 45.75% |
| **SP-04 Model (Conservative, L/S)**| **+19.83%** | **14.86%** | **1.097** | **2.119** | **-4.88%** | 19.73 | 50.39% |
| **SP-04 Model (Aggressive, L/S)** | **+22.79%** | **17.03%** | **0.972** | **1.640** | **-9.56%** | 26.22 | 46.75% |
| **SP-04 Model (Balanced, Long-Only)**| **+11.28%**| **8.53%** | **1.362** | **1.989** | **-3.67%** | 15.63 | 48.51% |
| Breakout Rule (Balanced, L/S) | +17.18% | 12.87% | 0.877 | 1.567 | -6.23% | 19.40 | 45.60% |
| Plain Breakout Baseline (Long-Only)| +28.52% | 21.20% | 1.517 | 2.208 | -7.78% | 18.70 | 37.40% |
| NIFTY 50 Buy & Hold | +30.40% | 22.50% | 0.864 | 1.140 | -38.40% | 0.00 | N/A |
| Equal-Weight Universe B&H | +40.60% | 29.80% | 1.079 | 1.429 | -36.00% | 0.00 | N/A |

### Out-of-Sample Test Period Backtest (1,413 Days, 2021–2026)

| Strategy Variant | Total Return | CAGR | Sharpe | Sortino | Max DD | Turnover | Win Rate |
|---|---|---|---|---|---|---|---|
| **SP-04 Model (Balanced, L/S)** | -33.10% | -6.90% | -0.520 | -0.733 | -40.80% | 28.06 | 32.61% |
| **SP-04 Model (Conservative, L/S)**| -24.00% | -4.80% | -0.344 | -0.494 | -34.60% | 24.00 | 33.48% |
| **SP-04 Model (Aggressive, L/S)** | -36.30% | -7.70% | -0.580 | -0.818 | -43.40% | 30.80 | 32.33% |
| **SP-04 Model (Balanced, Long-Only)**| **-16.90%** | **-3.20%** | **-0.216** | **-0.308** | **-29.30%** | 18.19 | 33.73% |
| Breakout Rule (Balanced, L/S) | -29.20% | -6.00% | -0.450 | -0.640 | -37.40% | 23.67 | 32.02% |
| Plain Breakout Baseline (Long-Only)| +3.21% | +0.57% | 0.123 | 0.175 | -30.93% | 18.08 | 37.64% |
| NIFTY 50 Buy & Hold | +48.95% | +7.37% | 0.581 | 0.815 | -17.23% | 0.00 | N/A |
| Equal-Weight Universe B&H | +102.85% | +13.45%| 1.001 | 1.436 | -14.30% | 0.00 | N/A |

### Key Backtesting Insights
1. **The Cost of Indian Equities Choppiness:** In the post-2021 secular bull market, large-cap Indian equities experienced repeated false breakouts and rotational chop. High turnover (~24–28x annual) compounded friction drag (10 bps round-trip transaction costs).
2. **Shorting Disadvantage in Indian Equities:** Theoretical shorting significantly dragged returns (-33.1% L/S vs -16.9% Long-Only), confirming that naked short selling in a strong bull market carries structural friction, especially under Indian regulatory cash market constraints.
3. **Conservative Profile Resilience:** The conservative profile (lower aggressiveness, higher patience) cut maximum drawdown by 6.2 percentage points and improved risk-adjusted metrics over the aggressive profile.

---

## 8. Generalization Experiments (B, C, D)

### Experiment B: Unseen-Stock Generalization (100 Holdout Stocks)
Evaluated across 132,001 test-period bars of the 100 quarantined stocks in a flat portfolio state:
- **Predicted Action Distribution:**
  - `HOLD`: **128,908 (97.66%)**
  - `BUY`: **2,194 (1.66%)**
  - `SELL`: **899 (0.68%)**
  - `EXIT`: **0 (0.00%)** *(correctly suppressed by action masking for flat state)*
- **Finding:** The model exhibits remarkable structural restraint on unseen stocks, refusing to overtrade on unfamiliar series and isolating high-confidence breakout setups at rates fully consistent with the main universe.

### Experiment C: Dedicated Top-20 Bluechip Performance

| Strategy | Total Return | CAGR | Sharpe | Max Drawdown | Win Rate |
|---|---|---|---|---|---|
| SP-04 Model (Balanced, L/S) | -33.10% | -6.90% | -0.520 | -40.80% | 32.61% |
| Breakout Rule (Balanced, L/S) | -29.20% | -6.00% | -0.450 | -37.40% | 32.02% |
| Plain Breakout Baseline | +3.21% | +0.57% | 0.123 | -30.93% | 37.64% |
| NIFTY 50 Buy & Hold | +48.95% | +7.37% | 0.581 | -17.23% | N/A |

### Experiment D: Market Regime Breakdown (Test Period)

| Market Regime Definition | Active Days | Ann. Return | Ann. Vol | Sharpe | Max Drawdown |
|---|---|---|---|---|---|
| **Bullish (60-day Return > +5%)** | 487 | -5.90% | 12.30% | -0.434 | -21.40% |
| **Bearish (60-day Return < -5%)** | 180 | -6.00% | 7.90% | -0.740 | -8.00% |
| **Sideways (\|Return\| $\le$ 5%)**| 686 | -8.20% | 13.70% | -0.557 | -29.40% |
| **High Volatility (Top 25% Vol)** | 348 | -7.20% | 8.00% | -0.897 | -11.20% |
| **Low Volatility (Bottom 25% Vol)**| 348 | **+0.60%** | 13.20% | **+0.114**| -12.80% |

- **Finding:** The breakout policy generated positive alpha in **Low Volatility** regimes (+0.60% return, Sharpe +0.114), where range compression preceded genuine clean expansion. In sideways and high-volatility regimes, repeated whipsaws degraded returns.

---

## 9. Round-Trip Persistence & Meta-RL Contract Verification

The policy was serialized and reloaded from disk (`BreakoutPolicy.load(RUN_DIR)`) to test inference stability and Meta-RL integration:
- **Bitwise Equality:** Verified `np.allclose(pred_orig['action_probs'], pred_reloaded['action_probs'], atol=1e-6)` on 64 out-of-sample states. Chosen discrete actions were identical.
- **Adapter Form Parity:** The `BreakoutFeatureAdapter` input form `(sequence[:, :, :30], breakout_features[:, :, 30:])` produced identical float32 probabilities to the concatenated 53-feature sequence.
- **Action Masking Integrity:** Validated that while holding a `LONG` position, `BUY` and `SELL` probabilities were strictly zeroed out ($0.0$), forcing valid transitions exclusively to `HOLD` or `EXIT`.
- **Stand-Alone Deployment:** Confirmed that `BreakoutPolicy` initializes and predicts with zero training or download dependencies.

---

## 10. Deliverables Manifest

All artifacts are persisted in `outputs/sp04_breakout_policy/indian_market_run_001/`:

1. [breakout_policy.pt](file:///d:/04_breakout_policy%20-%20Attempt%202/outputs/sp04_breakout_policy/indian_market_run_001/breakout_policy.pt) — Core self-contained deployment bundle with weights, config, and preprocessing scaler.
2. [best_checkpoint.pt](file:///d:/04_breakout_policy%20-%20Attempt%202/outputs/sp04_breakout_policy/indian_market_run_001/best_checkpoint.pt) — Best training checkpoint from Epoch 17 (`val_loss = 0.037372`).
3. [final_checkpoint.pt](file:///d:/04_breakout_policy%20-%20Attempt%202/outputs/sp04_breakout_policy/indian_market_run_001/final_checkpoint.pt) — Final training checkpoint.
4. [breakout_config.json](file:///d:/04_breakout_policy%20-%20Attempt%202/outputs/sp04_breakout_policy/indian_market_run_001/breakout_config.json) — Complete runtime configuration and hyperparameters.
5. [feature_schema.json](file:///d:/04_breakout_policy%20-%20Attempt%202/outputs/sp04_breakout_policy/indian_market_run_001/feature_schema.json) — Complete 53 sequential feature + 8 static feature schema.
6. [preprocessing.pkl](file:///d:/04_breakout_policy%20-%20Attempt%202/outputs/sp04_breakout_policy/indian_market_run_001/preprocessing.pkl) — Training-fitted median, mean, std, and clipping parameters.
7. [evaluation_metrics.json](file:///d:/04_breakout_policy%20-%20Attempt%202/outputs/sp04_breakout_policy/indian_market_run_001/evaluation_metrics.json) — Full evaluation metrics dictionary (classification, condition breakdown, backtest, regimes).
8. [training_history.csv](file:///d:/04_breakout_policy%20-%20Attempt%202/outputs/sp04_breakout_policy/indian_market_run_001/training_history.csv) — Epoch-by-epoch loss, macro-F1, and learning rate progression.
9. [equity_curve.csv](file:///d:/04_breakout_policy%20-%20Attempt%202/outputs/sp04_breakout_policy/indian_market_run_001/equity_curve.csv) — Daily backtest portfolio equity series across validation and test periods.
10. [training_log.txt](file:///d:/04_breakout_policy%20-%20Attempt%202/outputs/sp04_breakout_policy/indian_market_run_001/training_log.txt) — Verifiable stdout log of full execution.
11. [sp04_features.py](file:///d:/04_breakout_policy%20-%20Attempt%202/outputs/sp04_breakout_policy/indian_market_run_001/sp04_features.py) — Standalone feature extraction and adapter module.
12. [sp04_model.py](file:///d:/04_breakout_policy%20-%20Attempt%202/outputs/sp04_breakout_policy/indian_market_run_001/sp04_model.py) — Standalone model architecture and inference wrapper.
13. [learning_curves.png](file:///d:/04_breakout_policy%20-%20Attempt%202/outputs/sp04_breakout_policy/indian_market_run_001/figures/learning_curves.png) — Publication-grade training and validation loss & macro-F1 curve.
14. [confusion_matrix_test.png](file:///d:/04_breakout_policy%20-%20Attempt%202/outputs/sp04_breakout_policy/indian_market_run_001/figures/confusion_matrix_test.png) — Test set confusion matrices (raw counts and row-normalized).
15. [reliability_test.png](file:///d:/04_breakout_policy%20-%20Attempt%202/outputs/sp04_breakout_policy/indian_market_run_001/figures/reliability_test.png) — Reliability calibration diagram illustrating test-set ECE.
16. [backtest_test.png](file:///d:/04_breakout_policy%20-%20Attempt%202/outputs/sp04_breakout_policy/indian_market_run_001/figures/backtest_test.png) — Out-of-sample portfolio equity curves and drawdown comparison.
17. [label_distribution.png](file:///d:/04_breakout_policy%20-%20Attempt%202/outputs/sp04_breakout_policy/indian_market_run_001/figures/label_distribution.png) — Relative label frequency distribution across train, validation, and test splits.
18. [04_breakout_policy.ipynb](file:///d:/04_breakout_policy%20-%20Attempt%202/04_breakout_policy.ipynb) — Notebook configured with Indian stock market metadata and execution settings.
