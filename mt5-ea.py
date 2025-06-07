import MetaTrader5 as mt5
import pandas as pd
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
import time
from smartmoneyconcepts.smc import smc

# Initialize MT5 connection
if not mt5.initialize():
    print("❌ Failed to connect to MetaTrader 5.")
    quit()

# Define symbol and timeframe
symbol = "BTCUSD"  # Change to BTCUSD, EURUSD, etc.
timeframe = mt5.TIMEFRAME_M15  # 15-minute timeframe
num_bars = 500  # Number of bars to fetch

# Store last known price to detect changes
last_bid = None
last_ask = None


# Function to fetch real-time M5 candlestick data from MT5
def get_m5_candles(symbol, num_bars):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_bars)

    if rates is None or len(rates) == 0:
        print("❌ No M5 data received.")
        return None

    # Convert to DataFrame
    df = pd.DataFrame(rates)
    df.columns = ["Date", "open", "high", "low", "close", "tick_volume", "spread", "volume"]
    df["Date"] = pd.to_datetime(df["Date"], unit="s")
    df.set_index("Date", inplace=True)

    return df


# Main loop to fetch and display data in real time
while True:
    # Fetch last 500 M5 candles
    df = get_m5_candles(symbol, num_bars=500)
    if df is not None:
        print("\n500 Real-Time M5 Candles for BTCUSD:")
        print(df.tail())  # Display latest candles


    if df is not None:
        # Apply SMC indicators
        swing_data = pd.DataFrame(smc.swing_highs_lows(df, swing_length=10))
        fvg_data = smc.fvg(df, join_consecutive=True)
        bos_choch_data = smc.bos_choch(df, swing_data)
        ob_data = smc.ob(df, swing_data, close_mitigation=False)
        liquidity_data = smc.liquidity(df, swing_data, range_percent=0.01)
        previous_high_low_data = smc.previous_high_low(df, time_frame="1D")
        retracements_data = smc.retracements(df, swing_data)

        # Print last 10 values for each indicator
        print("\n📌 Fair Value Gap (FVG):")
        print(fvg_data.dropna())

        print("\n📌 Swing Highs & Lows:")
        print(swing_data.dropna() if not swing_data.empty else "No swing data available")

        print("\n📌 Break of Structure (BOS) & Change of Character (CHoCH):")
        print(bos_choch_data.dropna())

        print("\n📌 Order Blocks (OB):")
        print(ob_data.dropna())

        print("\n📌 Liquidity Zones:")
        print(liquidity_data.dropna())

        print("\n📌 Previous High & Low:")
        print(previous_high_low_data.dropna())

        print("\n📌 Retracements:")
        print(retracements_data.dropna())

    tick = mt5.symbol_info_tick(symbol)

    if tick:
        bid, ask = tick.bid, tick.ask

        # Only print if price changed
        if bid != last_bid or ask != last_ask:
            print(f"🔄 Price Updated - Bid: {bid}, Ask: {ask}")
            last_bid, last_ask = bid, ask