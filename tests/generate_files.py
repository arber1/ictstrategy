import os
import sys
import time
import pandas as pd
import unittest

BASE_DIR = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "test_data")))
from smartmoneyconcepts.smc import smc

# define and import test data
test_instrument = "BTCUSD"
instrument_data = f"{test_instrument}_15M.csv"
TEST_DATA_DIR = os.path.join(BASE_DIR, "test_data", test_instrument)
df = pd.read_csv(os.path.join(TEST_DATA_DIR, instrument_data))
df = df.set_index("Date")
df.index = pd.to_datetime(df.index)


def generate_results_data():
    fvg_data = smc.fvg(df)
    fvg_data.to_csv(
        os.path.join(TEST_DATA_DIR, "fvg_result_data.csv"), index=False
    )

    fvg_data = smc.fvg(df, join_consecutive=True)
    fvg_data.to_csv(
        os.path.join(TEST_DATA_DIR, "fvg_consecutive_result_data.csv"), index=False
    )

    swing_highs_lows_data = smc.swing_highs_lows(df, swing_length=5)
    swing_highs_lows_data.to_csv(
        os.path.join(TEST_DATA_DIR, "swing_highs_lows_result_data.csv"),
        index=False,
    )

    bos_choch_data = smc.bos_choch(df, swing_highs_lows_data)
    bos_choch_data.to_csv(
        os.path.join(TEST_DATA_DIR, "bos_choch_result_data.csv"),
        index=False,
    )

    ob_data = smc.ob(df, swing_highs_lows_data)
    ob_data.to_csv(
        os.path.join(TEST_DATA_DIR, "ob_result_data.csv"), index=False
    )

    liquidity_data = smc.liquidity(df, swing_highs_lows_data)
    liquidity_data.to_csv(
        os.path.join(TEST_DATA_DIR, "liquidity_result_data.csv"),
        index=False,
    )

    previous_high_low_data = smc.previous_high_low(df, time_frame="4h")
    previous_high_low_data.to_csv(
        os.path.join(TEST_DATA_DIR, "previous_high_low_result_data_4h.csv"),
        index=False,
    )

    previous_high_low_data = smc.previous_high_low(df, time_frame="1D")
    previous_high_low_data.to_csv(
        os.path.join(TEST_DATA_DIR, "previous_high_low_result_data_1D.csv"),
        index=False,
    )

    previous_high_low_data = smc.previous_high_low(df, time_frame="W")
    previous_high_low_data.to_csv(
        os.path.join(TEST_DATA_DIR, "previous_high_low_result_data_W.csv"),
        index=False,
    )

    sessions = smc.sessions(df, session="London")
    sessions.to_csv(
        os.path.join(TEST_DATA_DIR, "sessions_result_data.csv"),
        index=False,
    )

    retracements = smc.retracements(df, swing_highs_lows_data)
    retracements.to_csv(
        os.path.join(TEST_DATA_DIR, "retracements_result_data.csv"),
        index=False,
    )


generate_results_data()
#
# import MetaTrader5 as mt5
# import pandas as pd
# import os
# from datetime import datetime
#
#
# def fetch_mt5_btc_data():
#     # Create base directory paths
#     BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#     test_instrument = "BTCUSD"
#     test_data_dir = os.path.join(BASE_DIR, "test_data", test_instrument)
#     os.makedirs(test_data_dir, exist_ok=True)
#
#     # Initialize MT5 connection
#     if not mt5.initialize():
#         print("MT5 initialization failed")
#         mt5.shutdown()
#         return
#
#     try:
#         # Request 500 15-minute candles
#         rates = mt5.copy_rates_from_pos("BTCUSD", mt5.TIMEFRAME_M15, 0, 500)
#
#         if rates is None:
#             print("Failed to get data from MT5")
#             return
#
#         # Convert to DataFrame
#         df = pd.DataFrame(rates)
#
#         # Convert timestamp to datetime
#         df['Date'] = pd.to_datetime(df['time'], unit='s')
#
#         # Keep and rename only needed columns
#         df = df[['Date', 'open', 'high', 'low', 'close', 'tick_volume']]
#         df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
#
#         # Round prices to 2 decimal places
#         for col in ['Open', 'High', 'Low', 'Close']:
#             df[col] = df[col].round(2)
#
#         # Save to CSV
#         csv_path = os.path.join(test_data_dir, f"{test_instrument}_15M.csv")
#         df.to_csv(csv_path, index=False)
#
#         print(f"Successfully downloaded and saved {len(df)} candles to {csv_path}")
#         print(f"Date range: {df['Date'].iloc[0]} to {df['Date'].iloc[-1]}")
#         print("\nFirst few rows:")
#         print(df.head())
#
#     except Exception as e:
#         print(f"Error fetching data: {e}")
#
#     finally:
#         # Shut down MT5 connection
#         mt5.shutdown()
#
#
# if __name__ == "__main__":
#     fetch_mt5_btc_data()
#
