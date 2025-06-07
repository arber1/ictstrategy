import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import itertools
from datetime import datetime
from talib import RSI, STOCH, ATR

# 1) MT5 initialization
if not mt5.initialize():
    print("MT5 init failed:", mt5.last_error())
    quit()

def backtest(symbol: str,
             timeframe: int,
             history_bars: int,
             ema_period: int,
             stoch_rsi_period: int,
             atr_period: int,
             atr_mult: float,
             risk_pct: float,
             initial_balance: float = 10000.0) -> float:
    """
    Run a backtest of your EMA + StochRSI + ATR trailing‐stop strategy.
    Returns net profit (final balance – initial_balance).
    """
    print(f"[DEBUG] Loading {history_bars} bars of historical data for {symbol}...")
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, history_bars)
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    print(f"[DEBUG] Data loaded successfully. Shape: {df.shape}")

    # --- indicators
    print("[DEBUG] Calculating indicators...")
    df['ema'] = df['close'].ewm(span=ema_period, adjust=False).mean()
    rsi = RSI(df['close'], timeperiod=stoch_rsi_period)
    k, d = STOCH(rsi, rsi, rsi,
                 fastk_period=stoch_rsi_period,
                 slowk_period=3,
                 slowd_period=3)
    df['stoch_k'], df['stoch_d'] = k, d
    df['atr'] = ATR(df['high'], df['low'], df['close'], timeperiod=atr_period)
    print("[DEBUG] Indicators calculated successfully")

    # --- simulation state
    balance = 100
    in_pos = False
    entry_price = sl = tp = 0.0
    risk_per_trade = risk_pct / 100.0
    point = mt5.symbol_info(symbol).point
    lot_step = mt5.symbol_info(symbol).trade_contract_size or 1.0

    # --- iterate bar-by-bar
    print("[DEBUG] Starting bar-by-bar simulation...")
    for i in range(1, len(df)):
        if i % 1000 == 0:
            print(f"[DEBUG] Progress: {(i / len(df) * 100):.1f}%")
        bar = df.iloc[i]
        prev = df.iloc[i-1]

        # entry condition
        if not in_pos:
            if (bar['close'] > bar['ema']
                    and bar['stoch_k'] < 20
                    and prev['stoch_k'] < bar['stoch_k']):
                entry_price = bar['close']
                sl = entry_price - atr_mult * bar['atr']
                # position size in lots
                risk_amount = balance * risk_per_trade
                stop_dist = (entry_price - sl) / point
                volume = max(lot_step, round(risk_amount / stop_dist) * lot_step)
                in_pos = True

        # once in position, check exits
        else:
            low, high, close = bar['low'], bar['high'], bar['close']

            # 1) fixed SL hit
            if low <= sl:
                balance -= balance * risk_per_trade
                in_pos = False

            # 2) trailing SL
            else:
                new_sl = close - atr_mult * bar['atr']
                if new_sl > sl:
                    sl = new_sl  # move stop up

    final_profit = balance - initial_balance
    print(f"[DEBUG] Simulation complete. Final profit: {final_profit:.2f}")
    return final_profit


if __name__ == "__main__":
    # --- backtest settings
    SYMBOL       = "BTCUSD"
    TIMEFRAME    = mt5.TIMEFRAME_H1
    HISTORY_BARS = 50000

    # --- grid to sweep
    param_grid = {
        'ema_period':       [20, 50, 100],
        'stoch_rsi_period': [14, 21],
        'atr_period':       [14, 21],
        'atr_mult':         [2.0, 3.0, 4.0],
        'risk_pct':         [1.0, 2.0, 3.0],
    }

 
    # --- run grid search
    results = []
    for combo in itertools.product(*param_grid.values()):
        params = dict(zip(param_grid.keys(), combo))
        profit = backtest(
            symbol=SYMBOL,
            timeframe=TIMEFRAME,
            history_bars=HISTORY_BARS,
            **params
        )
        results.append({**params, 'profit': profit})

    # --- rank and display
    df_res = pd.DataFrame(results)
    df_top = df_res.sort_values('profit', ascending=False).head(10)
    print("\nTop 10 parameter sets by net profit:")
    print(df_top.to_string(index=False))

    mt5.shutdown()
