import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
from smartmoneyconcepts.smc import smc
from datetime import datetime, timedelta

# ========== Configuration ==========
SYMBOL = "XAUUSD"
TIMEFRAME = "M15"
NUM_CANDLES = 500
DRY_RUN = False
DEBUG_MODE = True

# ========== ICT Configuration ==========
ICT_CONFIG = {
    "SWING_LENGTHS": {
        "M1": 50,
        "M5": 10,
        "M15": 4,
        "H1": 3,
        "D1": 3
    },
    "FVG_EXPIRATION": {
        "M1": timedelta(hours=4),
        "M5": timedelta(hours=6),
        "M15": timedelta(hours=120),
        "H1": timedelta(days=1),
        "D1": timedelta(days=3)
    },
    "TRADING_SESSIONS": {
        "London": (7, 16),  # UTC hours
        "NewYork": (13, 22)  # UTC hours
    },
    "RISK_PERCENT": 1.0
}

def initialize_mt5():
    if not mt5.initialize():
        print("Failed to initialize MT5")
        return False
    print("✅ Successfully connected to MT5")
    return True


# ========== Enhanced Debugging ==========
def fetch_realtime_data():
    """Debuggable data fetcher"""
    try:
        print("\n=== FETCHING DATA ===")
        print(f"[DEBUG] MT5 initialized: {mt5.initialize()}")

        rates = mt5.copy_rates_from_pos(SYMBOL,
                                        getattr(mt5, f'TIMEFRAME_{TIMEFRAME}'),
                                        0,
                                        NUM_CANDLES)
        print(f"[DEBUG] Raw rates received: {type(rates)} | Length: {len(rates) if rates else 0}")

        df = pd.DataFrame(rates)
        print(f"[DEBUG] Initial columns: {df.columns.tolist()}")

        # Column processing
        df.columns = [col.lower().strip() for col in df.columns]
        df.rename(columns={'time': 'datetime'}, inplace=True)
        print(f"[DEBUG] After rename: {df.columns.tolist()}")

        # Validate OHLC
        missing_ohlc = [col for col in ['open', 'high', 'low', 'close'] if col not in df.columns]
        if missing_ohlc:
            print(f"🔴 Missing critical columns: {missing_ohlc}")
            return pd.DataFrame()

        # Convert datetime
        df['datetime'] = pd.to_datetime(df['datetime'], unit='s', utc=True)
        df.set_index('datetime', inplace=True)
        print(f"[DEBUG] Time range: {df.index.min()} to {df.index.max()}")

        # Validate prices
        price_check = {
            'high > low': (df['high'] > df['low']).all(),
            'close > 0': (df['close'] > 0).all(),
            'nans_in_ohlc': df[['open', 'high', 'low', 'close']].isna().sum().sum()
        }
        print("[DEBUG] Price validation:")
        for k, v in price_check.items():
            print(f"  {k}: {v}")

        return df[['open', 'high', 'low', 'close', 'volume']]

    except Exception as e:
        print(f"🔴 Fetch error: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


class SMCTrader:
    def get_swing_data(self, df):
        """Debuggable swing detection"""
        print("\n=== SWING DETECTION ===")
        print(f"[INPUT] Data shape: {df.shape}")
        print(f"[INPUT] Columns: {df.columns.tolist()}")
        print(f"[INPUT] First 3 rows:\n{df.head(3)}")
        print(f"[INPUT] Last 3 rows:\n{df.tail(3)}")

        validated_df = self._validate_input_data(df)
        print(f"[VALIDATION] Cleaned data shape: {validated_df.shape}")

        if validated_df.empty:
            print("🔴 No valid data for swings!")
            return self._create_synthetic_swings()

        try:
            temp = validated_df.reset_index(drop=True)
            print(f"[PROCESSING] Temp data:\nIndex: {temp.index[:5]}...\nColumns: {temp.columns}")

            swings = smc.swing_highs_lows(temp, swing_length=self.swing_length)
            print(f"[SWINGS RAW] {swings.shape} | Types: {swings['HighLow'].value_counts().to_dict()}")

            processed = self._process_swings(swings, validated_df)
            print(f"[SWINGS FINAL] {processed.shape} | First: {processed.iloc[0]} | Last: {processed.iloc[-1]}")

            return processed

        except Exception as e:
            print(f"🔴 Swing processing error: {e}")
            return self._create_synthetic_swings()

    def _validate_input_data(self, df):
        """Data validation with debug"""
        print("\n=== DATA VALIDATION ===")
        if df.empty:
            print("🔴 Empty input DataFrame!")
            return df

        # Column check
        original_cols = df.columns.tolist()
        df.columns = [col.strip().lower() for col in df.columns]
        print(f"[COLUMNS] Before: {original_cols} | After: {df.columns.tolist()}")

        # Required columns
        missing = [col for col in ['open', 'high', 'low', 'close'] if col not in df.columns]
        if missing:
            print(f"🔴 Missing columns: {missing}")
            return pd.DataFrame()

        # Price validation
        valid_mask = (
                (df['high'] >= df['low']) &
                (df['high'] >= df['close']) &
                (df['low'] <= df['open']) &
                (df['close'] > 0)
        )
        print(f"[VALIDATION] Valid rows: {valid_mask.sum()}/{len(df)}")
        print(f"[VALIDATION] Invalid examples:\n{df[~valid_mask].head(2)}")

        return df[valid_mask].dropna()

    def _create_synthetic_swings(self):
        """Debug-friendly synthetic data"""
        print("\n⚠️ GENERATING SYNTHETIC SWINGS")
        now = pd.Timestamp.now(tz='UTC')
        synthetic = pd.DataFrame({
            'highlow': [1, -1],
            'level': [1800.00, 1750.00],  # Realistic XAUUSD levels
            'datetime': [now - pd.Timedelta(minutes=30), now]
        }).set_index('datetime')
        print(f"[SYNTHETIC] Created swings:\n{synthetic}")
        return synthetic


def main_loop():
    if not initialize_mt5():
        return

    trader = SMCTrader(SYMBOL, TIMEFRAME)
    print("🚀 Starting Trading Bot...")

    # Initial data load
    df = fetch_realtime_data()
    print(f"[MAIN] Initial data loaded: {len(df)} candles")

    if not df.empty:
        trader.historical_data = df.iloc[-NUM_CANDLES:]
        print(
            f"[MAIN] Historical data range: {trader.historical_data.index.min()} - {trader.historical_data.index.max()}")
    else:
        print("🔴 No initial data!")

    while True:
        try:
            new_data = fetch_realtime_data()
            print(f"[MAIN] New data received: {len(new_data)} candles")

            if not new_data.empty:
                combined = pd.concat([trader.historical_data, new_data]).drop_duplicates()
                trader.historical_data = combined.iloc[-NUM_CANDLES:]
                print(
                    f"[MAIN] Updated history: {len(trader.historical_data)} candles | New range: {trader.historical_data.index.min()} - {trader.historical_data.index.max()}")

            # Test swing detection
            test_swings = trader.get_swing_data(trader.historical_data.tail(100))
            print("\n[TEST] Latest swings:")
            print(test_swings[['HighLow', 'Level']].tail(5))

            time.sleep(5)

        except Exception as e:
            print(f"🔴 Main loop error: {e}")
            time.sleep(10)