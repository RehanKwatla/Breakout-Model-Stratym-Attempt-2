"""
Hard Test Evaluation for SP-04 Breakout Policy on BSE.NS
Testing the trained model checkpoint against real historical out-of-universe data for BSE.NS.
"""
import sys
import os
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from sp04_features import (
    ACTION_NAMES, DEFAULT_PROFILE, FEATURE_COLUMNS, BASE_FEATURE_COLUMNS, EXTRA_FEATURE_COLUMNS,
    HOLD, BUY, SELL, EXIT, N_ACTIONS, WARMUP_BARS,
    build_features, valid_action_mask
)
from sp04_model import BreakoutPolicy

PROFILES = {
    "balanced": (0.5, 0.5, 0.5),
    "conservative": (0.2, 0.8, 0.2),
    "aggressive": (0.8, 0.2, 0.8),
}

CONFIG = {
    "initial_capital": 1000000.0,
    "fee_bps": 5.0,
    "slippage_bps": 5.0,
    "short_borrow_bps_annual": 300.0,
    "risk_free_annual": 0.06,
    "seq_len": 30,
    "warmup_bars": 240, # Need sufficient bars for sma200 + 30 seq_len
}

FEE = CONFIG["fee_bps"] / 10000.0
SLIP = CONFIG["slippage_bps"] / 10000.0
BORROW_DAILY = CONFIG["short_borrow_bps_annual"] / 10000.0 / 252.0
CAP0 = float(CONFIG["initial_capital"])
L = CONFIG["seq_len"]

def run_simulation(df_feats: pd.DataFrame, decide_fn, allow_short: bool = True) -> dict:
    """
    Simulates single-asset trading with realistic execution:
    - Decision made at close of bar t based on history up to t.
    - Execution at open of bar t+1 with slippage and transaction fees.
    """
    dates = df_feats.index.to_numpy()
    opens = df_feats["Open"].to_numpy(dtype=float)
    closes = df_feats["Close"].to_numpy(dtype=float)
    T = len(dates)
    
    cash = CAP0
    shares = 0.0
    entry_px = 0.0
    entry_idx = 0
    hold_bars = 0
    prev_eq = CAP0
    
    eq_hist = []
    ret_hist = []
    trades = []
    actions_taken = []
    
    # We begin decisions at k = CONFIG["warmup_bars"]
    # So first possible trade entry execution is at k + 1
    start_k = CONFIG["warmup_bars"]
    
    for k in range(start_k, T - 1):
        pos = int(np.sign(shares))
        px_curr = closes[k]
        pnl = pos * (px_curr / entry_px - 1.0) if pos != 0 and entry_px > 0 else 0.0
        
        # Decide action on bar k
        probs = decide_fn(k, pos, pnl, hold_bars)
        valid = valid_action_mask(np.array([pos]), allow_short=allow_short)[0]
        
        # Mask invalid actions
        masked_probs = np.where(valid, probs, -1.0)
        action = int(masked_probs.argmax())
        actions_taken.append(action)
        
        o_next = opens[k + 1]
        c_next = closes[k + 1]
        
        # 1. Process EXIT
        if action == EXIT and shares != 0:
            sh = shares
            px = o_next * (1.0 - SLIP) if sh > 0 else o_next * (1.0 + SLIP)
            trade_val = abs(sh) * px
            comm = trade_val * FEE
            cash += sh * px - comm
            trade_ret = (px / entry_px - 1.0) if sh > 0 else (1.0 - px / entry_px)
            trades.append({
                "side": "LONG" if sh > 0 else "SHORT",
                "entry_date": str(pd.to_datetime(dates[entry_idx]).date()),
                "exit_date": str(pd.to_datetime(dates[k + 1]).date()),
                "entry_px": float(entry_px),
                "exit_px": float(px),
                "shares": abs(float(sh)),
                "bars": int(hold_bars),
                "ret_pct": float(trade_ret * 100.0),
                "pnl_inr": float(sh * (px - entry_px) - 2 * comm) if sh > 0 else float(-sh * (entry_px - px) - 2 * comm)
            })
            shares = 0.0
            entry_px = 0.0
            hold_bars = 0
            
        # 2. Process ENTRY (BUY or SELL)
        elif action in (BUY, SELL) and shares == 0:
            if action == BUY:
                px = o_next * (1.0 + SLIP)
                # Allocate 100% of available cash minus fee
                notional = cash / (1.0 + FEE)
                if notional > 1000.0:
                    shares = notional / px
                    cash -= notional * (1.0 + FEE)
                    entry_px = px
                    entry_idx = k + 1
                    hold_bars = 1
            elif action == SELL and allow_short:
                px = o_next * (1.0 - SLIP)
                notional = cash # 100% margin
                if notional > 1000.0:
                    shares = -notional / px
                    cash += notional * (1.0 - FEE)
                    entry_px = px
                    entry_idx = k + 1
                    hold_bars = 1
        else:
            if shares != 0:
                hold_bars += 1
                
        # End of day k+1 equity calculation
        if shares < 0:
            # Short borrow fee
            cash -= abs(shares * c_next) * BORROW_DAILY
            
        eq = cash + (shares * c_next)
        ret = (eq / prev_eq) - 1.0
        eq_hist.append(eq)
        ret_hist.append(ret)
        prev_eq = eq

    # Force close any open position on the final day
    if shares != 0:
        last_c = closes[-1]
        px = last_c * (1.0 - SLIP) if shares > 0 else last_c * (1.0 + SLIP)
        trade_ret = (px / entry_px - 1.0) if shares > 0 else (1.0 - px / entry_px)
        trades.append({
            "side": "LONG" if shares > 0 else "SHORT",
            "entry_date": str(pd.to_datetime(dates[entry_idx]).date()),
            "exit_date": str(pd.to_datetime(dates[-1]).date()) + " (MKT CLOSE)",
            "entry_px": float(entry_px),
            "exit_px": float(px),
            "shares": abs(float(shares)),
            "bars": int(hold_bars),
            "ret_pct": float(trade_ret * 100.0),
            "pnl_inr": float(shares * (px - entry_px)) if shares > 0 else float(-shares * (entry_px - px))
        })
        eq = cash + (shares * px) - abs(shares * px) * FEE
        eq_hist[-1] = eq
        
    trades_df = pd.DataFrame(trades)
    r = np.array(ret_hist)
    eq_arr = np.array(eq_hist)
    n_days = len(r)
    years = n_days / 252.0
    
    total_ret = (eq_arr[-1] / CAP0 - 1.0) if len(eq_arr) else 0.0
    cagr = ((eq_arr[-1] / CAP0) ** (1.0 / years) - 1.0) if years > 0 and eq_arr[-1] > 0 else 0.0
    sd = r.std(ddof=1) if n_days > 1 else 0.0
    ann_vol = sd * np.sqrt(252.0)
    rf_daily = CONFIG["risk_free_annual"] / 252.0
    ex = r - rf_daily
    sharpe = (ex.mean() / sd * np.sqrt(252.0)) if sd > 1e-8 else 0.0
    
    dd_sd = np.sqrt((np.minimum(ex, 0.0) ** 2).mean()) if n_days else 0.0
    sortino = (ex.mean() / dd_sd * np.sqrt(252.0)) if dd_sd > 1e-8 else 0.0
    
    peak = np.maximum.accumulate(np.concatenate([[CAP0], eq_arr]))[1:]
    mdd = float((eq_arr / peak - 1.0).min()) if len(eq_arr) else 0.0
    calmar = cagr / abs(mdd) if abs(mdd) > 1e-4 else 0.0
    
    win_rate = float((trades_df["ret_pct"] > 0).mean()) if len(trades_df) else 0.0
    gross_gains = trades_df.loc[trades_df["pnl_inr"] > 0, "pnl_inr"].sum() if len(trades_df) else 0.0
    gross_losses = abs(trades_df.loc[trades_df["pnl_inr"] < 0, "pnl_inr"].sum()) if len(trades_df) else 0.0
    profit_factor = (gross_gains / gross_losses) if gross_losses > 1e-6 else (999.0 if gross_gains > 0 else 0.0)
    avg_trade_ret = float(trades_df["ret_pct"].mean()) if len(trades_df) else 0.0
    
    return {
        "dates": dates[start_k+1:],
        "equity": eq_arr,
        "returns": r,
        "trades": trades_df,
        "actions_taken": actions_taken,
        "metrics": {
            "Total Return (%)": total_ret * 100.0,
            "CAGR (%)": cagr * 100.0,
            "Annualized Volatility (%)": ann_vol * 100.0,
            "Sharpe Ratio": sharpe,
            "Sortino Ratio": sortino,
            "Calmar Ratio": calmar,
            "Max Drawdown (%)": mdd * 100.0,
            "Final Equity (INR)": eq_arr[-1] if len(eq_arr) else CAP0,
            "Number of Trades": len(trades_df),
            "Win Rate (%)": win_rate * 100.0,
            "Profit Factor": profit_factor,
            "Avg Trade Return (%)": avg_trade_ret,
            "Avg Holding Bars": float(trades_df["bars"].mean()) if len(trades_df) else 0.0
        }
    }

def main():
    print("=" * 80)
    print("SP-04 BREAKOUT POLICY: HARD TEST ON BSE.NS")
    print("=" * 80)
    
    csv_path = r"C:\Users\Rehan\.gemini\antigravity\brain\14088b97-9ee1-4108-a64e-90d54efe79f1\.user_uploaded\media_1795250441470_9ce0d9d9.csv"
    bench_path = "Data/nifty50_benchmark.csv"
    
    if not Path(bench_path).exists():
        print(f"Waiting for benchmark file {bench_path} to finish...")
        for _ in range(60):
            if Path(bench_path).exists():
                break
            time.sleep(2)
            
    print(f"Loading BSE.NS dataset from {csv_path}...")
    df_raw = pd.read_csv(csv_path)
    df_raw["date"] = pd.to_datetime(df_raw["date"], format="%d-%m-%Y")
    df_raw = df_raw.sort_values("date").reset_index(drop=True)
    
    # Compute adjusted OHLC
    adj_ratio = (df_raw["adjclose"] / df_raw["close"]).values
    df_ohlcv = pd.DataFrame({
        "Open": (df_raw["open"].values * adj_ratio),
        "High": (df_raw["high"].values * adj_ratio),
        "Low": (df_raw["low"].values * adj_ratio),
        "Close": df_raw["adjclose"].values,
        "Volume": df_raw["volume"].values
    }, index=df_raw["date"])
    
    print(f"BSE.NS bars: {len(df_ohlcv):,} from {df_ohlcv.index.min().date()} to {df_ohlcv.index.max().date()}")
    
    print(f"Loading NIFTY 50 benchmark from {bench_path}...")
    bench_df = pd.read_csv(bench_path)
    bench_df["date"] = pd.to_datetime(bench_df["date"])
    bench_df = bench_df.set_index("date").sort_index()
    print(f"Benchmark bars: {len(bench_df):,} from {bench_df.index.min().date()} to {bench_df.index.max().date()}")
    
    print("Engineering 53 Breakout Features via build_features...")
    df_feats = build_features(df_ohlcv, bench_df)
    print(f"Engineered feature shape: {df_feats.shape}")
    
    # Verify no NaN in feature columns after warmup
    post_warmup = df_feats.iloc[CONFIG["warmup_bars"]:][FEATURE_COLUMNS]
    n_nan = post_warmup.isna().sum().sum()
    print(f"Features post-warmup ({len(post_warmup)} bars): {n_nan} NaNs")
    
    # Load trained Breakout Policy
    model_dir = "outputs/sp04_breakout_policy/indian_market_run_001"
    print(f"Loading trained BreakoutPolicy from {model_dir}...")
    policy = BreakoutPolicy.load(model_dir, device="cpu")
    print("Policy successfully loaded.")
    
    # Pre-extract raw feature matrix for high-speed simulation
    feat_mat = df_feats[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    offsets = np.arange(-(L - 1), 1)
    
    # Define decide functions
    def make_model_decide(profile_tuple):
        prof = np.asarray(profile_tuple, dtype=np.float32)
        def fn(k, pos, pnl, hold):
            # sequence slice: (30, 53)
            seq = feat_mat[k + offsets][None, :, :]
            state = {
                "sequence": seq,
                "position": np.array([pos]),
                "profile": prof[None, :],
                "unrealized_pnl": np.array([pnl]),
                "holding_bars": np.array([hold])
            }
            out = policy.predict(state)
            return out["action_probs"][0]
        return fn

    def make_plain_breakout_decide():
        up20 = df_feats["brk_up_flag_20"].to_numpy()
        ema20 = df_feats["close_ema20"].to_numpy()
        def fn(k, pos, pnl, hold):
            probs = np.zeros(N_ACTIONS)
            if pos == 0 and up20[k] > 0:
                probs[BUY] = 1.0
            elif pos == 1 and (ema20[k] < 0 or pnl <= -0.05):
                probs[EXIT] = 1.0
            else:
                probs[HOLD] = 1.0
            return probs
        return fn

    # Evaluate configurations
    experiments = [
        ("SP-04 Policy (Balanced, Long/Short)", make_model_decide(PROFILES["balanced"]), True),
        ("SP-04 Policy (Balanced, Long-Only)", make_model_decide(PROFILES["balanced"]), False),
        ("SP-04 Policy (Aggressive, Long/Short)", make_model_decide(PROFILES["aggressive"]), True),
        ("SP-04 Policy (Aggressive, Long-Only)", make_model_decide(PROFILES["aggressive"]), False),
        ("SP-04 Policy (Conservative, Long-Only)", make_model_decide(PROFILES["conservative"]), False),
        ("Plain Breakout Baseline (Long-Only)", make_plain_breakout_decide(), False),
    ]
    
    results = {}
    print("\n" + "=" * 80)
    print("RUNNING BACKTEST SIMULATIONS (Execution at Open t+1 with 5bps slippage & 5bps fees)")
    print("=" * 80)
    
    start_eval_bar = CONFIG["warmup_bars"]
    eval_dates = df_feats.index[start_eval_bar+1:]
    eval_closes = df_feats["Close"].iloc[start_eval_bar+1:].to_numpy()
    
    # Benchmark 1: BSE.NS Buy-and-Hold
    bse_bh_ret = (eval_closes[-1] / eval_closes[0] - 1.0) * 100.0
    years_eval = len(eval_dates) / 252.0
    bse_bh_cagr = ((eval_closes[-1] / eval_closes[0]) ** (1.0 / years_eval) - 1.0) * 100.0
    peak_bse = np.maximum.accumulate(eval_closes)
    bse_bh_mdd = float(((eval_closes / peak_bse) - 1.0).min()) * 100.0
    
    # Benchmark 2: NIFTY 50 Buy-and-Hold
    bench_reindex = bench_df["Close"].reindex(eval_dates).ffill().bfill().to_numpy()
    nifty_bh_ret = (bench_reindex[-1] / bench_reindex[0] - 1.0) * 100.0
    nifty_bh_cagr = ((bench_reindex[-1] / bench_reindex[0]) ** (1.0 / years_eval) - 1.0) * 100.0
    peak_nifty = np.maximum.accumulate(bench_reindex)
    nifty_bh_mdd = float(((bench_reindex / peak_nifty) - 1.0).min()) * 100.0
    
    for name, dec_fn, allow_s in experiments:
        res = run_simulation(df_feats, dec_fn, allow_short=allow_s)
        results[name] = res
        m = res["metrics"]
        print(f"\n>>> {name}")
        print(f"  Total Return: {m['Total Return (%)']:+.2f}% | CAGR: {m['CAGR (%)']:+.2f}%")
        print(f"  Sharpe: {m['Sharpe Ratio']:.2f} | Sortino: {m['Sortino Ratio']:.2f} | Calmar: {m['Calmar Ratio']:.2f} | Max DD: {m['Max Drawdown (%)']:.2f}%")
        print(f"  Trades: {m['Number of Trades']} | Win Rate: {m['Win Rate (%)']:.1f}% | Profit Factor: {m['Profit Factor']:.2f} | Avg Trade: {m['Avg Trade Return (%)']:+.2f}%")
        print(f"  Final Equity: Rs. {m['Final Equity (INR)']:,.0f} (from Rs. {CAP0:,.0f})")

    print("\n" + "=" * 80)
    print("BENCHMARKS SUMMARY OVER SAME PERIOD:")
    print(f"  BSE.NS Buy-and-Hold: Total Return: {bse_bh_ret:+.2f}% | CAGR: {bse_bh_cagr:+.2f}% | Max DD: {bse_bh_mdd:.2f}%")
    print(f"  NIFTY 50 Buy-and-Hold: Total Return: {nifty_bh_ret:+.2f}% | CAGR: {nifty_bh_cagr:+.2f}% | Max DD: {nifty_bh_mdd:.2f}%")
    print("=" * 80)
    
    # Save detailed trade logs and metrics to json
    summary_table = []
    for name, res in results.items():
        row = {"Strategy": name}
        row.update(res["metrics"])
        summary_table.append(row)
        
    summary_table.append({
        "Strategy": "BSE.NS Buy & Hold",
        "Total Return (%)": bse_bh_ret,
        "CAGR (%)": bse_bh_cagr,
        "Annualized Volatility (%)": float(np.std(eval_closes[1:] / eval_closes[:-1] - 1.0) * np.sqrt(252.0) * 100.0),
        "Sharpe Ratio": float((np.mean(eval_closes[1:] / eval_closes[:-1] - 1.0) - CONFIG['risk_free_annual']/252.0) / np.std(eval_closes[1:] / eval_closes[:-1] - 1.0) * np.sqrt(252.0)),
        "Sortino Ratio": np.nan,
        "Calmar Ratio": bse_bh_cagr / abs(bse_bh_mdd),
        "Max Drawdown (%)": bse_bh_mdd,
        "Final Equity (INR)": CAP0 * (1.0 + bse_bh_ret / 100.0),
        "Number of Trades": 1,
        "Win Rate (%)": 100.0,
        "Profit Factor": np.nan,
        "Avg Trade Return (%)": bse_bh_ret,
        "Avg Holding Bars": len(eval_dates)
    })
    
    summary_table.append({
        "Strategy": "NIFTY 50 Buy & Hold",
        "Total Return (%)": nifty_bh_ret,
        "CAGR (%)": nifty_bh_cagr,
        "Annualized Volatility (%)": float(np.std(bench_reindex[1:] / bench_reindex[:-1] - 1.0) * np.sqrt(252.0) * 100.0),
        "Sharpe Ratio": float((np.mean(bench_reindex[1:] / bench_reindex[:-1] - 1.0) - CONFIG['risk_free_annual']/252.0) / np.std(bench_reindex[1:] / bench_reindex[:-1] - 1.0) * np.sqrt(252.0)),
        "Sortino Ratio": np.nan,
        "Calmar Ratio": nifty_bh_cagr / abs(nifty_bh_mdd),
        "Max Drawdown (%)": nifty_bh_mdd,
        "Final Equity (INR)": CAP0 * (1.0 + nifty_bh_ret / 100.0),
        "Number of Trades": 1,
        "Win Rate (%)": 100.0,
        "Profit Factor": np.nan,
        "Avg Trade Return (%)": nifty_bh_ret,
        "Avg Holding Bars": len(eval_dates)
    })

    df_summary = pd.DataFrame(summary_table)
    df_summary.to_csv("outputs/bse_hard_test_summary.csv", index=False)
    
    # Save trade logs for the primary strategy
    best_strat = "SP-04 Policy (Balanced, Long/Short)"
    results[best_strat]["trades"].to_csv("outputs/bse_hard_test_trades_balanced_ls.csv", index=False)
    results["SP-04 Policy (Balanced, Long-Only)"]["trades"].to_csv("outputs/bse_hard_test_trades_balanced_lo.csv", index=False)
    results["SP-04 Policy (Aggressive, Long-Only)"]["trades"].to_csv("outputs/bse_hard_test_trades_aggressive_lo.csv", index=False)
    print("\nSaved summary to outputs/bse_hard_test_summary.csv and trade logs to outputs/.")

if __name__ == "__main__":
    main()
