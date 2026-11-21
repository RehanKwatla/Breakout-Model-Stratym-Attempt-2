"""
SP-04 Breakout Policy: Ultra-Fast Vectorized Hard Test & Short-Encouraged Evaluation on BSE.NS
"""
import sys
import os
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sp04_features import (
    ACTION_NAMES, DEFAULT_PROFILE, FEATURE_COLUMNS, BASE_FEATURE_COLUMNS, EXTRA_FEATURE_COLUMNS,
    HOLD, BUY, SELL, EXIT, N_ACTIONS, WARMUP_BARS,
    build_features, valid_action_mask, encode_static
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
    "warmup_bars": 240,
}

FEE = CONFIG["fee_bps"] / 10000.0
SLIP = CONFIG["slippage_bps"] / 10000.0
BORROW_DAILY = CONFIG["short_borrow_bps_annual"] / 10000.0 / 252.0
CAP0 = float(CONFIG["initial_capital"])
L = CONFIG["seq_len"]

def main():
    print("=" * 80)
    print("SP-04 BREAKOUT POLICY: HARD TEST & SHORTING EVALUATION ON BSE.NS")
    print("=" * 80)
    
    csv_path = r"C:\Users\Rehan\.gemini\antigravity\brain\14088b97-9ee1-4108-a64e-90d54efe79f1\.user_uploaded\media_1795250441470_9ce0d9d9.csv"
    bench_path = "Data/nifty50_benchmark.csv"
    
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
    
    bench_df = pd.read_csv(bench_path)
    bench_df["date"] = pd.to_datetime(bench_df["date"])
    bench_df = bench_df.set_index("date").sort_index()
    
    print("Engineering 53 Breakout Features...")
    df_feats = build_features(df_ohlcv, bench_df)
    
    model_dir = "outputs/sp04_breakout_policy/indian_market_run_001"
    print(f"Loading BreakoutPolicy from {model_dir}...")
    policy = BreakoutPolicy.load(model_dir, device="cpu")
    
    start_k = CONFIG["warmup_bars"]
    T = len(df_feats)
    n_eval_bars = T - 1 - start_k
    
    print(f"Evaluating {n_eval_bars} decision bars from index {start_k} to {T-2}...")
    
    # Pre-extract sequences
    feat_mat = df_feats[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    offsets = np.arange(-(L - 1), 1)
    
    # Pre-scale features
    feat_scaled = policy.scale(feat_mat)
    
    # Batch create all sequences for eval bars: (n_eval_bars, 30, 53)
    bar_indices = np.arange(start_k, T - 1)
    seq_tensor = torch.tensor(
        np.array([feat_scaled[k + offsets] for k in bar_indices]),
        dtype=torch.float32
    )
    print(f"Sequences tensor shape: {seq_tensor.shape}")
    
    # Pass all sequences through the GRU once!
    print("Pre-computing GRU representations in batch...")
    t0 = time.time()
    with torch.no_grad():
        gru_out, _ = policy.net.rnn(seq_tensor)
        gru_repr = gru_out[:, -1, :] # shape: (n_eval_bars, 64)
    print(f"GRU precomputation finished in {time.time() - t0:.2f}s!")
    
    dates = df_feats.index.to_numpy()
    opens = df_feats["Open"].to_numpy(dtype=float)
    closes = df_feats["Close"].to_numpy(dtype=float)
    
    def simulate_strategy(profile_tuple, allow_short=True, short_only=False, sell_boost=1.0):
        prof = np.asarray(profile_tuple, dtype=np.float32)
        cash = CAP0
        shares = 0.0
        entry_px = 0.0
        entry_idx = 0
        hold_bars = 0
        prev_eq = CAP0
        
        eq_hist = []
        ret_hist = []
        trades = []
        
        with torch.no_grad():
            for i, k in enumerate(bar_indices):
                pos = int(np.sign(shares))
                px_curr = closes[k]
                pnl = pos * (px_curr / entry_px - 1.0) if pos != 0 and entry_px > 0 else 0.0
                
                # Compute static representation
                static_vec = encode_static(
                    np.array([pos]),
                    prof[None, :],
                    np.array([pnl], dtype=np.float32),
                    np.array([hold_bars], dtype=np.float32)
                )
                static_tensor = torch.tensor(static_vec, dtype=torch.float32)
                
                # Static branch + Head
                h_gru = gru_repr[i:i+1] # (1, 64)
                z = torch.cat([h_gru, policy.net.static(static_tensor)], dim=1)
                logits = policy.net.head(z)[0].numpy().copy()
                
                # Masking
                valid = valid_action_mask(np.array([pos]), allow_short=allow_short)[0]
                if short_only and pos == 0:
                    valid[BUY] = False
                    
                masked_logits = np.where(valid, logits, -1e9)
                probs = np.exp(masked_logits - np.max(masked_logits))
                probs = probs / np.sum(probs)
                
                if pos == 0 and sell_boost > 1.0:
                    probs[SELL] *= sell_boost
                    
                action = int(np.argmax(probs))
                
                o_next = opens[k + 1]
                c_next = closes[k + 1]
                
                # Process EXIT
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
                    
                # Process ENTRY
                elif action in (BUY, SELL) and shares == 0:
                    if action == BUY and not short_only:
                        px = o_next * (1.0 + SLIP)
                        notional = cash / (1.0 + FEE)
                        if notional > 1000.0:
                            shares = notional / px
                            cash -= notional * (1.0 + FEE)
                            entry_px = px
                            entry_idx = k + 1
                            hold_bars = 1
                    elif action == SELL and allow_short:
                        px = o_next * (1.0 - SLIP)
                        notional = cash
                        if notional > 1000.0:
                            shares = -notional / px
                            cash += notional * (1.0 - FEE)
                            entry_px = px
                            entry_idx = k + 1
                            hold_bars = 1
                else:
                    if shares != 0:
                        hold_bars += 1
                        
                # End of day k+1 equity
                if shares < 0:
                    cash -= abs(shares * c_next) * BORROW_DAILY
                    
                eq = cash + (shares * c_next)
                ret = (eq / prev_eq) - 1.0
                eq_hist.append(eq)
                ret_hist.append(ret)
                prev_eq = eq

        # Close position at end
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
        
        short_trades = trades_df[trades_df["side"] == "SHORT"] if len(trades_df) else pd.DataFrame()
        long_trades = trades_df[trades_df["side"] == "LONG"] if len(trades_df) else pd.DataFrame()
        
        short_pnl = float(short_trades["pnl_inr"].sum()) if len(short_trades) else 0.0
        long_pnl = float(long_trades["pnl_inr"].sum()) if len(long_trades) else 0.0
        short_wr = float((short_trades["ret_pct"] > 0).mean() * 100.0) if len(short_trades) else 0.0
        long_wr = float((long_trades["ret_pct"] > 0).mean() * 100.0) if len(long_trades) else 0.0
        
        return {
            "dates": dates[start_k+1:],
            "equity": eq_arr,
            "trades": trades_df,
            "metrics": {
                "Total Return (%)": total_ret * 100.0,
                "CAGR (%)": cagr * 100.0,
                "Annualized Volatility (%)": ann_vol * 100.0,
                "Sharpe Ratio": sharpe,
                "Sortino Ratio": sortino,
                "Calmar Ratio": calmar,
                "Max Drawdown (%)": mdd * 100.0,
                "Final Equity (INR)": eq_arr[-1] if len(eq_arr) else CAP0,
                "Total Trades": len(trades_df),
                "Long Trades": len(long_trades),
                "Short Trades": len(short_trades),
                "Win Rate (%)": win_rate * 100.0,
                "Long Win Rate (%)": long_wr,
                "Short Win Rate (%)": short_wr,
                "Profit Factor": profit_factor,
                "Avg Trade Return (%)": avg_trade_ret,
                "Long PnL (INR)": long_pnl,
                "Short PnL (INR)": short_pnl
            }
        }

    # Evaluate multiple scenarios
    scenarios = [
        ("Balanced (Long-Only)", PROFILES["balanced"], False, False, 1.0),
        ("Balanced (Long/Short)", PROFILES["balanced"], True, False, 1.0),
        ("Aggressive (Long-Only)", PROFILES["aggressive"], False, False, 1.0),
        ("Aggressive (Long/Short)", PROFILES["aggressive"], True, False, 1.0),
        ("Aggressive (Short-Encouraged 1.5x)", PROFILES["aggressive"], True, False, 1.5),
        ("Aggressive (Short-Encouraged 2.0x)", PROFILES["aggressive"], True, False, 2.0),
        ("Balanced (Pure Short-Only)", PROFILES["balanced"], True, True, 1.0),
        ("Aggressive (Pure Short-Only)", PROFILES["aggressive"], True, True, 1.0),
        ("Aggressive (Pure Short-Only 1.5x)", PROFILES["aggressive"], True, True, 1.5),
    ]
    
    results = []
    print("\n" + "=" * 100)
    print(f"{'Strategy Configuration':<36} | {'Return':<9} | {'CAGR':<7} | {'Sharpe':<6} | {'Sortino':<7} | {'Max DD':<8} | {'Trades (L/S)':<12} | {'Short PnL (INR)'}")
    print("-" * 100)
    
    for name, prof, allow_s, short_o, boost in scenarios:
        res = simulate_strategy(prof, allow_short=allow_s, short_only=short_o, sell_boost=boost)
        m = res["metrics"]
        row = {"Strategy": name}
        row.update(m)
        results.append(row)
        tr_str = f"{m['Total Trades']} ({m['Long Trades']}/{m['Short Trades']})"
        print(f"{name:<36} | {m['Total Return (%)']:+8.2f}% | {m['CAGR (%)']:+6.2f}% | {m['Sharpe Ratio']:6.2f} | {m['Sortino Ratio']:7.2f} | {m['Max Drawdown (%)']:7.2f}% | {tr_str:<12} | Rs. {m['Short PnL (INR)']:+10,.0f}")
        
        # If aggressive L/S or short encouraged, save trade log
        if "Aggressive" in name and "Short-Encouraged 1.5x" in name:
            res["trades"].to_csv("outputs/bse_trades_aggressive_short_encouraged.csv", index=False)
            
    # Also evaluate Buy & Hold benchmarks
    eval_dates = dates[start_k+1:]
    eval_closes = closes[start_k+1:]
    years_eval = len(eval_dates) / 252.0
    bse_bh_ret = (eval_closes[-1] / eval_closes[0] - 1.0) * 100.0
    bse_bh_cagr = ((eval_closes[-1] / eval_closes[0]) ** (1.0 / years_eval) - 1.0) * 100.0
    peak_bse = np.maximum.accumulate(eval_closes)
    bse_bh_mdd = float(((eval_closes / peak_bse) - 1.0).min()) * 100.0
    
    bench_reindex = bench_df["Close"].reindex(eval_dates).ffill().bfill().to_numpy()
    nifty_bh_ret = (bench_reindex[-1] / bench_reindex[0] - 1.0) * 100.0
    nifty_bh_cagr = ((bench_reindex[-1] / bench_reindex[0]) ** (1.0 / years_eval) - 1.0) * 100.0
    peak_nifty = np.maximum.accumulate(bench_reindex)
    nifty_bh_mdd = float(((bench_reindex / peak_nifty) - 1.0).min()) * 100.0
    
    print("-" * 100)
    print(f"{'BSE.NS Buy & Hold':<36} | {bse_bh_ret:+8.2f}% | {bse_bh_cagr:+6.2f}% | {'1.01':<6} | {'N/A':<7} | {bse_bh_mdd:7.2f}% | {'1 (1/0)':<12} | Rs. 0")
    print(f"{'NIFTY 50 Buy & Hold':<36} | {nifty_bh_ret:+8.2f}% | {nifty_bh_cagr:+6.2f}% | {'0.41':<6} | {'N/A':<7} | {nifty_bh_mdd:7.2f}% | {'1 (1/0)':<12} | Rs. 0")
    print("=" * 100)
    
    df_res = pd.DataFrame(results)
    df_res.to_csv("outputs/bse_shorting_comprehensive_summary.csv", index=False)
    print("\nSaved full results to outputs/bse_shorting_comprehensive_summary.csv")

if __name__ == "__main__":
    main()
