"""
Professional ICT/SMC Trading Bot
- Wick-based BOS/CHOCH validation
- 3-candle swing confirmation
- Fibonacci retracement levels
- Volume spread analysis
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from smartmoneyconcepts.smc import smc

# ========== Configuration ==========
SYMBOL = "BTCUSD"
TIMEFRAME = "M15"
NUM_CANDLES = 100
RISK_PERCENT = 1.0
ICT_CONFIG = {
    "fvg_expiration": timedelta(hours=12),
    "bos_confirm_bars": 3,
    "fib_levels": [0.236, 0.382, 0.5, 0.618, 0.786]
}

class SMCTrader:
    def __init__(self, symbol, timeframe):
        self.symbol = symbol
        self.timeframe = timeframe
        self.historical_data = pd.DataFrame()
        self.pip_size = mt5.symbol_info(symbol).point * 10

    def calculate_fvg(self, df):
        """Fair Value Gaps with proper expiration handling"""
        fvg_data = smc.fvg(df, join_consecutive=True)
        fvg_data.index = df.index  # Preserve original datetime index

        # Expiration check
        now = pd.Timestamp.now(tz='UTC')
        valid_fvg = fvg_data[
            (now - fvg_data.index.tz_convert('UTC')) <= ICT_CONFIG["fvg_expiration"]
            ]
        return valid_fvg

    def validate_bos(self, df, swings):
        """Wick-based Break of Structure validation"""
        if swings.empty or 'HighLow' not in swings.columns:
            return pd.DataFrame()

        try:
            # Ensure integer index
            swings = swings.reset_index(drop=True)
            bos_data = smc.bos_choch(df, swings)
        except KeyError as e:
            print(f"BOS calculation error: {e}")
            return pd.DataFrame()

        if bos_data.empty:
            return bos_data

        # Wick confirmation with n-bar momentum
        bos_data["ValidBullBOS"] = (df["high"] > bos_data["Level"]) & \
                                   (df["close"].rolling(ICT_CONFIG["bos_confirm_bars"]).min() > bos_data["Level"])
        bos_data["ValidBearBOS"] = (df["low"] < bos_data["Level"]) & \
                                   (df["close"].rolling(ICT_CONFIG["bos_confirm_bars"]).max() < bos_data["Level"])

        return bos_data[
            (bos_data["BOS"] == 1 & bos_data["ValidBullBOS"]) |
            (bos_data["BOS"] == -1 & bos_data["ValidBearBOS"])
        ]

    def validate_swings(self, df):
        """ICT-valid 3-candle swing points"""
        raw_swings = smc.swing_highs_lows(df, swing_length=14)
        valid_swings = pd.DataFrame()

        # Add empty check
        if raw_swings.empty:
            return valid_swings

        # Reset index for safe iteration
        raw_swings = raw_swings.reset_index(drop=True)

        for i in range(2, len(raw_swings) - 2):
            if raw_swings.iloc[i]["HighLow"] == 1:  # Swing high
                left = df.iloc[:i]["high"].max()
                right = df.iloc[i+1:]["high"].max()
                if df.iloc[i]["high"] > left and df.iloc[i]["high"] > right:
                    valid_swings = pd.concat([valid_swings, raw_swings.iloc[[i]]])
            else:  # Swing low
                left = df.iloc[:i]["low"].min()
                right = df.iloc[i+1:]["low"].min()
                if df.iloc[i]["low"] < left and df.iloc[i]["low"] < right:
                    valid_swings = pd.concat([valid_swings, raw_swings.iloc[[i]]])

        return valid_swings

    def calculate_fib_levels(self, swings):
        """Fibonacci retracement levels between latest swings"""
        if len(swings) < 2:
            return {}

        swing_high = swings[swings["HighLow"] == 1]["Level"].iloc[-1]
        swing_low = swings[swings["HighLow"] == -1]["Level"].iloc[-1]
        fib_levels = {}

        for level in ICT_CONFIG["fib_levels"]:
            price = swing_high - (swing_high - swing_low) * level
            fib_levels[f"Fib_{level*100:.1f}%"] = price

        return fib_levels

    def validate_liquidity(self, df, swings):
        """Liquidity sweep detection with wick confirmation"""
        liq_data = smc.liquidity(df, swings)
        if liq_data.empty:
            return liq_data

        liq_data["Swept"] = False
        for idx, row in liq_data.iterrows():
            if row["Liquidity"] == 1:  # Bullish liquidity
                liq_data.at[idx, "Swept"] = df["low"].loc[idx:] < row["Level"]
            else:  # Bearish liquidity
                liq_data.at[idx, "Swept"] = df["high"].loc[idx:] > row["Level"]

        return liq_data[~liq_data["Swept"]]

    def volume_spike_check(self, df):
        """ICT Volume Spread Analysis"""
        current_volume = df["volume"].iloc[-1]
        vol_ma = df["volume"].rolling(20).mean().iloc[-1]
        return current_volume > vol_ma * 1.5

    def generate_signals(self, df):
        """Composite signal generation with all validations"""
        signals = pd.Series(index=df.index, dtype=int)
        swings = self.validate_swings(df)

        # Core components
        fvg = self.calculate_fvg(df)
        bos = self.validate_bos(df, swings)
        liq = self.validate_liquidity(df, swings)
        fib_levels = self.calculate_fib_levels(swings)
        volume_spike = self.volume_spike_check(df)

        # Signal logic
        buy_conditions = (
                (fvg["FVG"] == 1) &
                (bos["BOS"] == 1) &
                (liq["Liquidity"] == 1) &
                volume_spike
        )

        sell_conditions = (
                (fvg["FVG"] == -1) &
                (bos["BOS"] == -1) &
                (liq["Liquidity"] == -1) &
                volume_spike
        )

        signals[buy_conditions] = 1
        signals[sell_conditions] = -1

        return signals

class ICTExecution:
    def __init__(self, symbol):
        self.symbol = symbol
        self.trader = SMCTrader(symbol, TIMEFRAME)

    def execute_trades(self, df):
        signals = self.trader.generate_signals(df)
        current_signal = signals.iloc[-1]

        if current_signal != 0:
            price = mt5.symbol_info_tick(self.symbol).ask if current_signal == 1 \
                else mt5.symbol_info_tick(self.symbol).bid

            # Risk management
            sl = self.calculate_stop_loss(current_signal, df)
            tp = self.calculate_take_profit(price, current_signal)
            size = self.calculate_position_size(price, sl)

            self.send_order(current_signal, price, sl, tp, size)

    def calculate_stop_loss(self, signal, df):
        """Dynamic liquidity-based SL"""
        swings = self.trader.validate_swings(df)
        if signal == 1:
            return swings[swings["HighLow"] == -1]["Level"].min() - 3 * self.trader.pip_size
        return swings[swings["HighLow"] == 1]["Level"].max() + 3 * self.trader.pip_size

    def calculate_position_size(self, price, sl):
        """Risk-adjusted sizing"""
        risk_amount = mt5.account_info().balance * (RISK_PERCENT / 100)
        risk_pips = abs(price - sl) / self.trader.pip_size
        return round(risk_amount / (risk_pips * self.trader.pip_size), 2)

    def send_order(self, signal, price, sl, tp, size):
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": size,
            "type": mt5.ORDER_TYPE_BUY if signal == 1 else mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "comment": "ICT Strategy"
        }
        mt5.order_send(request)

# ========== Main Loop ==========
def main():
    mt5.initialize()
    executor = ICTExecution(SYMBOL)

    while True:
        try:
            # if not in_session():  # Time check commented as requested
            #     continue

            rates = mt5.copy_rates_from_pos(SYMBOL,
                                            getattr(mt5, f'TIMEFRAME_{TIMEFRAME}'), 0, NUM_CANDLES)

            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
            df.set_index('time', inplace=True)

            executor.execute_trades(df)
            mt5.sleep(300)

        except KeyboardInterrupt:
            break

    mt5.shutdown()

if __name__ == "__main__":
    main()