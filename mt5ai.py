"""
Smart Money Concepts (SMC) Trading Bot
This module implements SMC trading strategies using MetaTrader5.
"""

try:
    import MetaTrader5 as mt5
except ImportError:
    raise ImportError("MetaTrader5 package is not installed. Please install it first.")

import pandas as pd
# pd.set_option("display.max_columns", None)
# pd.set_option("display.max_rows", None)

import numpy as np
import time
from smartmoneyconcepts.smc import smc
from datetime import datetime, timedelta
from typing import Dict, Union, Optional

# ========== Configuration ==========
SYMBOL: str = ("BTCUSD")
TIMEFRAME: str = "M15"
NUM_CANDLES: int = 100
DRY_RUN: bool = False
DEBUG_MODE: bool = True

# ========== ICT Configuration ==========
ICT_CONFIG: Dict[str, Union[Dict, float]] = {
    "SWING_LENGTHS": {
        "M1": 75,  # Number of candles to look back for swing detection
        "M5": 20,
        "M15": 20,
        "H1": 50,
        "D1": 15
    },
    "FVG_EXPIRATION": {
        "M1": timedelta(hours=4),  # Fair Value Gap expiration times
        "M5": timedelta(hours=6),
        "M15": timedelta(hours=3),
        "H1": timedelta(days=1),
        "D1": timedelta(days=3)
    },
    "TRADING_SESSIONS": {
        "London": (8, 16),
        "NewYork": (13, 22)
    },
    "RISK_PERCENT": 1.0  # Risk percentage per trade
}
# ky funksioni eshte ok
def initialize_mt5() -> bool:
    """
    Initialize connection to MetaTrader 5 terminal.
    
    Returns:
        bool: True if connection successful, False otherwise
    """
    print("\n=== MT5 INITIALIZATION START ===")
    try:
        print("[DEBUG] Attempting MT5 initialization...")
        if not mt5.initialize():
            print("❌ Failed to initialize MT5: Connection error")
            print("[DEBUG] MT5 initialization returned False")
            return False

        print("[DEBUG] Getting account info...")
        account_info = mt5.account_info()
        if account_info is None:
            print("❌ Failed to get account info")
            print("[DEBUG] Account info is None")
            return False

        print("[DEBUG] Account details:")
        print(f"[DEBUG] - Login: {account_info.login}")
        print(f"[DEBUG] - Balance: {account_info.balance}")
        print(f"[DEBUG] - Equity: {account_info.equity}")
        print(f"✅ Successfully connected to MT5 (Server: {account_info.server})")
        print("=== MT5 INITIALIZATION END ===\n")
        return True

    except Exception as e:
        print("=== MT5 INITIALIZATION ERROR ===")
        print(f"❌ MT5 initialization error: {str(e)}")
        import traceback
        print("[DEBUG] Full traceback:")
        traceback.print_exc()
        print("=== MT5 INITIALIZATION ERROR END ===\n")
        return False

# ky funksioni eshte ok
def fetch_realtime_data():
    """Fetch live data from MT5 with UTC timezone."""
    try:
        if DEBUG_MODE:
            print("\n=== FETCHING DATA ===")
            print(f"[DEBUG] MT5 initialized: {mt5.initialize()}")
            print(f"[DEBUG] Fetching {NUM_CANDLES} {TIMEFRAME} candles for {SYMBOL}")

        tf = getattr(mt5, f'TIMEFRAME_{TIMEFRAME}')
        if tf is None:
            print(f"❌ Invalid timeframe: {TIMEFRAME}")
            return pd.DataFrame()

        rates = mt5.copy_rates_from_pos(SYMBOL, tf, 0, NUM_CANDLES)
        if rates is None:
            print("❌ MT5 returned no data") if DEBUG_MODE else None
            return pd.DataFrame()

        df = pd.DataFrame(rates)
        if DEBUG_MODE:
            print(f"[DEBUG] Raw data shape: {df.shape}")
            print(f"[DEBUG] Initial columns: {df.columns.tolist()}")

        # Standardize column names
        df.columns = [col.lower().strip() for col in df.columns]
        required_cols = ['time', 'open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            print(f"Missing required columns: {[col for col in required_cols if col not in df.columns]}")
            return pd.DataFrame()

        # Rename time column
        df.rename(columns={'time': 'datetime'}, inplace=True)

        # Handle volume data
        if 'real_volume' in df.columns and df['real_volume'].sum() > 0:
            df['volume'] = df['real_volume']
        elif 'tick_volume' in df.columns and df['tick_volume'].sum() > 0:
            df['volume'] = df['tick_volume']
        else:
            df['volume'] = 1

        # Convert datetime
        df['datetime'] = pd.to_datetime(df['datetime'], unit='s', utc=True)
        df.set_index('datetime', inplace=True)
        df = df.sort_index(ascending=True)

        # Validate data
        if DEBUG_MODE:
            print(f"[DEBUG] Time range: {df.index.min()} to {df.index.max()}")
            print("[DEBUG] Price validation:")
            print(f"  High > Low: {(df['high'] > df['low']).all()}")
            print(f"  Close > 0: {(df['close'] > 0).all()}")
            print(f"  NaNs in OHLC: {df[['open', 'high', 'low', 'close']].isna().sum().sum()}")

        final_df = df[['open', 'high', 'low', 'close', 'volume']]
        print(final_df.tail()) if DEBUG_MODE else None
        return final_df

    except Exception as e:
        print(f"Data error: {str(e)}")
        if DEBUG_MODE:
            import traceback
            traceback.print_exc()
        return pd.DataFrame()

class SMCTrader:
    def __init__(self, symbol, timeframe):
        self.symbol = symbol
        self.timeframe = timeframe
        self.swing_length = ICT_CONFIG["SWING_LENGTHS"][timeframe]
        self.fvg_expiration = ICT_CONFIG["FVG_EXPIRATION"][timeframe]
        self.historical_data = pd.DataFrame()
        self.active_fvgs = pd.DataFrame()
        self.pip_size = mt5.symbol_info(symbol).point * 10

    # def is_trading_session(self):
    #     """Check active trading sessions"""
    #     try:
    #         now = datetime.utcnow().hour
    #         london = ICT_CONFIG["TRADING_SESSIONS"]["London"]
    #         ny = ICT_CONFIG["TRADING_SESSIONS"]["NewYork"]
    #         return (london[0] <= now <= london[1]) or (ny[0] <= now <= ny[1])
    #     except Exception as e:
    #         print(f"Session check error: {e}")
    #         return True
    # ky funksioni eshte pjeserisht ok sepse duhet mbajtur ne monitorim ne kohe reale
    def get_swing_data(self, df):
        """Swing detection with call counter"""
        if not hasattr(self, "_swing_warn_counter"):
            self._swing_warn_counter = 0

        if df.empty or len(df) < 2:
            self._swing_warn_counter += 1
            if self._swing_warn_counter % 3 != 0:
                return self._generate_fallback_swings(df)
            print("⚠️ Insufficient data for swing calculation")
            return self._generate_fallback_swings(df)

        self._swing_warn_counter = 0

        try:
            required_cols = ['open', 'high', 'low', 'close']
            if not all(col in df.columns for col in required_cols):
                print(f"⚠️ Missing required columns: {required_cols}")
                return pd.DataFrame(columns=['HighLow', 'Level'])

            temp = df[required_cols].reset_index(drop=True)
            swing_length = min(self.swing_length, len(temp) - 2)
            swing_length = max(swing_length, 2)

            if DEBUG_MODE:
                print(f"[DEBUG] Swing data shape: {temp.shape}")

            swings = smc.swing_highs_lows(temp, swing_length=swing_length)

            if isinstance(swings, pd.DataFrame) and not swings.empty:
                swings = swings.dropna(subset=['HighLow', 'Level'])
                swings = swings[~swings.index.duplicated(keep='last')]
                swings['HighLow'] = swings['HighLow'].astype(int)
                swings['Level'] = swings['Level'].astype(float)

                # Map integer indices to df's datetime index
                swings.index = df.index[swings.index.astype(int)]
                print(f"[DEBUG] Swings detected: {swings.shape[0]} valid swings") if DEBUG_MODE else None
            else:
                swings = pd.DataFrame(columns=['HighLow', 'Level'])

            if len(swings) < 2:
                print("⚠️ Insufficient swings, adding fallback")
                return self._generate_fallback_swings(df)

            return swings.iloc[-10:].copy()

        except Exception as e:
            print(f"Swing detection critical error: {e}")
            if not df.empty and len(df) >= 2:
                return pd.DataFrame({
                    'HighLow': [1, -1],
                    'Level': [df['high'].iloc[-2], df['low'].iloc[-1]]
                }, index=[df.index[-2], df.index[-1]])
            elif not df.empty and len(df) == 1:
                return pd.DataFrame({
                    'HighLow': [1],
                    'Level': [df['high'].iloc[-1]]
                }, index=[df.index[-1]])
            return pd.DataFrame(columns=['HighLow', 'Level'])

    # def get_swing_data(self, df):
    #     """Swing detection with improved error handling and fallback"""
    #     if not hasattr(self, "_swing_warn_counter"):
    #         self._swing_warn_counter = 0
    #
    #     # Basic data validation with detailed logging
    #     if df is None or not isinstance(df, pd.DataFrame):
    #         print("⚠️ Invalid dataframe provided to get_swing_data")
    #         return pd.DataFrame(columns=['HighLow', 'Level'])
    #
    #     if df.empty:
    #         self._swing_warn_counter += 1
    #         if self._swing_warn_counter % 3 == 0:
    #             print("⚠️ Empty dataframe provided to get_swing_data")
    #         return pd.DataFrame(columns=['HighLow', 'Level'])
    #
    #     # Check minimum required rows (need at least swing_length + 1 rows for valid swing detection)
    #     min_required = max(self.swing_length + 1, 2)
    #     if len(df) < min_required:
    #         self._swing_warn_counter += 1
    #         if self._swing_warn_counter % 3 == 0:
    #             print(f"⚠️ Insufficient data for swing calculation (needs {min_required} rows, got {len(df)})")
    #         return self._generate_fallback_swings(df)
    #
    #     # Reset counter on valid data
    #     self._swing_warn_counter = 0
    #
    #     try:
    #         # Validate required columns
    #         required_cols = ['open', 'high', 'low', 'close']
    #         missing_cols = [col for col in required_cols if col not in df.columns]
    #         if missing_cols:
    #             print(f"⚠️ Missing required columns: {missing_cols}")
    #             return pd.DataFrame(columns=['HighLow', 'Level'])
    #
    #         # Create working copy with numeric index
    #         temp = df.copy()
    #         temp = temp.reset_index(drop=True)
    #
    #         # Calculate safe swing length
    #         safe_swing_length = min(max(self.swing_length, 2), len(temp) - 2)
    #
    #         try:
    #             # Prepare data for swing detection
    #             swing_data = temp[required_cols].copy()
    #
    #             if DEBUG_MODE:
    #                 print(f"[DEBUG] Processing swing data: shape={swing_data.shape}")
    #
    #             # Calculate swings
    #             swings = smc.swing_highs_lows(swing_data, swing_length=safe_swing_length)
    #
    #             if not isinstance(swings, pd.DataFrame):
    #                 print("⚠️ Invalid swing detection result")
    #                 return self._generate_fallback_swings(df)
    #
    #             if swings.empty:
    #                 print("⚠️ No swings detected")
    #                 return self._generate_fallback_swings(df)
    #
    #             # Process valid swings
    #             swings = swings.dropna(subset=['HighLow', 'Level'])
    #             if swings.empty:
    #                 return self._generate_fallback_swings(df)
    #
    #             # Clean up swing data
    #             swings = swings[~swings.index.duplicated(keep='last')]
    #             swings['HighLow'] = swings['HighLow'].astype(int)
    #             swings['Level'] = swings['Level'].astype(float)
    #
    #             # Map indices back to original datetime
    #             try:
    #                 swings.index = df.index[swings.index.astype(int)]
    #             except Exception as idx_error:
    #                 print(f"⚠️ Index mapping error: {idx_error}")
    #                 return self._generate_fallback_swings(df)
    #
    #             # Return last 10 valid swings
    #             return swings.iloc[-10:]
    #
    #         except Exception as swing_error:
    #             print(f"⚠️ Swing calculation error: {swing_error}")
    #             return self._generate_fallback_swings(df)
    #
    #     except Exception as e:
    #         print(f"⚠️ Critical error in swing detection: {e}")
    #         return self._generate_fallback_swings(df)

    # def get_swing_data(self, df):
    #     """Get highest and lowest swings within the swing length"""
    #     try:
    #         if df.empty or len(df) < 2:
    #             print("⚠️ Insufficient data for swing calculation")
    #             return pd.DataFrame({
    #                 'HighLow': [1, -1],
    #                 'Level': [df['high'].iloc[-1], df['low'].iloc[-1]]
    #             }, index=[df.index[-1], df.index[-1]])
    #
    #         # Create working copy with integer index
    #         temp = df.copy()
    #         temp = temp.reset_index()
    #         temp_index = temp.index
    #
    #         # Calculate swing length with safety bounds
    #         calculated_swing_length = min(self.swing_length, len(temp) - 2)
    #         calculated_swing_length = max(calculated_swing_length, 2)
    #
    #         # Get raw swing data
    #         swing_data = temp[['open', 'high', 'low', 'close']].copy()
    #         swing_data.index = np.arange(len(swing_data))
    #
    #         swings = smc.swing_highs_lows(swing_data, swing_length=calculated_swing_length)
    #
    #         # Get the highest high and lowest low within the swing length
    #         highest_swing = swings[swings['HighLow'] == 1]
    #         lowest_swing = swings[swings['HighLow'] == -1]
    #
    #         if not highest_swing.empty and not lowest_swing.empty:
    #             highest_idx = highest_swing['Level'].idxmax()
    #             lowest_idx = lowest_swing['Level'].idxmin()
    #
    #             result = pd.DataFrame({
    #                 'HighLow': [1, -1],
    #                 'Level': [highest_swing.loc[highest_idx, 'Level'],
    #                           lowest_swing.loc[lowest_idx, 'Level']]
    #             }, index=[df.index[highest_idx], df.index[lowest_idx]])
    #
    #             return result.sort_index()
    #
    #         else:
    #             # Fallback if no swings found
    #             return pd.DataFrame({
    #                 'HighLow': [1, -1],
    #                 'Level': [df['high'].iloc[-1], df['low'].iloc[-1]]
    #             }, index=[df.index[-1], df.index[-1]])
    #
    #     except Exception as e:
    #         print(f"Swing detection error: {str(e)}")
    #         # Return fallback values on error
    #         return pd.DataFrame({
    #             'HighLow': [1, -1],
    #             'Level': [df['high'].iloc[-1], df['low'].iloc[-1]]
    #         }, index=[df.index[-1], df.index[-1]])
    #     # """Detect swing points with robust index handling"""
    #     # try:
    #     #     if df.empty or len(df) < 2:
    #     #         print("⚠️ Insufficient data for swing calculation")
    #     #         return pd.DataFrame(columns=['HighLow', 'Level'])
    #     #
    #     #     # Validate required columns
    #     #     required_cols = ['open', 'high', 'low', 'close']
    #     #     missing_cols = [col for col in required_cols if col not in df.columns]
    #     #     if missing_cols:
    #     #         print(f"⚠️ Missing required columns: {missing_cols}")
    #     #         return pd.DataFrame(columns=['HighLow', 'Level'])
    #     #
    #     #     # Create working copy
    #     #     temp = df.copy()
    #     #     temp = temp.reset_index()
    #     #     temp_index = temp.index
    #     #     original_length = len(temp)
    #     #
    #     #     # Calculate swing length with safety bounds
    #     #     calculated_swing_length = min(self.swing_length, original_length - 2)
    #     #     calculated_swing_length = max(calculated_swing_length, 2)
    #     #
    #     #     try:
    #     #         # Prepare data for swing detection
    #     #         swing_data = temp[['open', 'high', 'low', 'close']].copy()
    #     #         swing_data.index = np.arange(len(swing_data))
    #     #
    #     #         if DEBUG_MODE:
    #     #             print(f"[DEBUG] Swing data columns: {swing_data.columns}")
    #     #             print(f"[DEBUG] Swing data shape: {swing_data.shape}")
    #     #
    #     #         # Calculate swings using integer indices
    #     #         swings = smc.swing_highs_lows(swing_data, swing_length=calculated_swing_length)
    #     #
    #     #         if isinstance(swings, pd.DataFrame) and not swings.empty:
    #     #             # Convert swing indices to integer positions
    #     #             swing_positions = pd.Index(swings.index).astype(int)
    #     #             valid_mask = (swing_positions >= 0) & (swing_positions < len(df))
    #     #             valid_positions = swing_positions[valid_mask]
    #     #
    #     #             if len(valid_positions) > 0:
    #     #                 # Map back to original datetime index
    #     #                 swings = swings.iloc[valid_mask]
    #     #                 swings.index = df.index[valid_positions]
    #     #             else:
    #     #                 print("⚠️ No valid swing positions")
    #     #                 swings = pd.DataFrame()
    #     #         else:
    #     #             print("⚠️ No valid swings detected")
    #     #             swings = pd.DataFrame()
    #     #
    #     #     except Exception as e:
    #     #         print(f"Swing detection error: {e}")
    #     #         swings = pd.DataFrame()
    #     #
    #     #     # Process valid swings
    #     #     swings = swings.dropna(subset=['HighLow', 'Level'])
    #     #     swings = swings[~swings.index.duplicated(keep='last')]
    #     #     swings['HighLow'] = swings['HighLow'].astype(int)
    #     #     swings['Level'] = swings['Level'].astype(float)
    #     #
    #     #     # Ensure minimum number of swings
    #     #     if len(swings) < 2:
    #     #         print("⚠️ Insufficient swings, adding fallback")
    #     #         new_index = df.index[-1]
    #     #         new_row = pd.DataFrame({
    #     #             'HighLow': [-1 if not swings.empty and swings['HighLow'] == 1 else 1],
    #     #             'Level': [df['low'] if not swings.empty and swings['HighLow'] == 1
    #     #                       else df['high']]
    #     #         }, index=[new_index])
    #     #         swings = pd.concat([swings, new_row])
    #     #
    #     #     # Get last 10 valid swings
    #     #     valid_swings = swings.iloc[-10:].copy()
    #     #
    #     #     return valid_swings
    #     #
    #     # except Exception as e:
    #     #     print(f"Swing detection critical error: {e}")
    #     #     return pd.DataFrame({
    #     #         'HighLow': [1, -1],
    #     #         'Level': [df['high'].iloc[-2], df['low']]
    #     #     }, index=[df.index[-2], df.index[-1]])

    # ky funksioni eshte pjeserisht ok sepse duhet mbajtur ne monitorim ne kohe reale
    def _generate_fallback_swings(self, df):
        """Generate emergency swings"""
        if df.empty:
            return pd.DataFrame(columns=['HighLow', 'Level'])
        print(pd.DataFrame({
            'HighLow': [1, -1],
            'Level': [
                df['high'].iloc[-1] if not df.empty else 0,
                df['low'].iloc[-1] if not df.empty else 0
            ]
        }, index=[df.index[-1]] * 2))
        return pd.DataFrame({
            'HighLow': [1, -1],
            'Level': [
                df['high'].iloc[-1] if not df.empty else 0,
                df['low'].iloc[-1] if not df.empty else 0
            ]
        }, index=[df.index[-1]] * 2)

    def validate_fvg(self, fvg_data):
        """Validate FVGs"""
        if not isinstance(fvg_data, pd.DataFrame):
            print("⚠️ Invalid FVG data type")
            return pd.DataFrame()

        if fvg_data.empty:
            return fvg_data
        try:
            fvg_data = fvg_data.reindex(self.historical_data.index)
            current_time = self.historical_data.index[-1]
            valid_fvg = fvg_data[
                (current_time - fvg_data.index) <= self.fvg_expiration
                ].copy()
            last_close = self.historical_data['close'].iloc[-1]
            valid_fvg['Mitigated'] = np.where(
                (valid_fvg['FVG'] == 1) & (last_close < valid_fvg['Bottom']),
                True,
                np.where((valid_fvg['FVG'] == -1) & (last_close > valid_fvg['Top']),
                         True, False
                         ))
            return valid_fvg[~valid_fvg['Mitigated']]
        except Exception as e:
            print(f"FVG validation error: {e}")
            return pd.DataFrame()

    def validate_ob(self, ob_data):
        """Validate Order Blocks with enhanced NaN handling"""
        if not isinstance(ob_data, pd.DataFrame) or ob_data.empty:
            print("⚠️ Invalid OB input data")
            return pd.DataFrame()

        try:
            # Ensure required columns with fallbacks
            required_cols = ['OB', 'Top', 'Bottom']
            current_price = self.historical_data['close'].iloc[-1]

            # Create missing columns with safe defaults
            for col in required_cols:
                if col not in ob_data.columns:
                    if col == 'OB':
                        ob_data['OB'] = 0  # Default to neutral
                    else:
                        ob_data[col] = current_price  # Use current price as fallback

            # Convert numeric types safely
            ob_data[required_cols] = ob_data[required_cols].apply(
                pd.to_numeric, errors='coerce'
            )

            # Fill numeric NaNs with price-based values
            ob_data['Top'] = ob_data['Top'].fillna(current_price)
            ob_data['Bottom'] = ob_data['Bottom'].fillna(current_price)
            ob_data['OB'] = ob_data['OB'].fillna(0)

            # Calculate mitigation with price guardrails
            ob_data['Mitigated'] = np.where(
                ob_data['OB'] == 1,
                current_price > ob_data['Top'],
                np.where(
                    ob_data['OB'] == -1,
                    current_price < ob_data['Bottom'],
                    False  # Neutral OB never mitigates
                )
            )

            return ob_data[~ob_data['Mitigated']]

        except Exception as e:
            print(f"OB Validation Error: {str(e)}")
            return pd.DataFrame()

    # def validate_ob(self, ob_data):
    #     """Validate Order Blocks"""
    #     if not isinstance(ob_data, pd.DataFrame):
    #         print("⚠️ validate_ob: Input is not a DataFrame")
    #         return pd.DataFrame()
    #
    #     if ob_data.empty:
    #         print("⚠️ validate_ob: Input DataFrame is empty")
    #         return pd.DataFrame()
    #
    #     try:
    #         # Check if required columns exist
    #         required_cols = ['OB', 'Top', 'Bottom']
    #         if not all(col in ob_data.columns for col in required_cols):
    #             missing = [col for col in required_cols if col not in ob_data.columns]
    #             print(f"⚠️ validate_ob: Missing columns: {missing}")
    #
    #             # Try to create missing columns with defaults if possible
    #             for col in missing:
    #                 if col == 'OB':
    #                     # Try to infer OB direction from Top/Bottom if available
    #                     if 'Top' in ob_data.columns and 'Bottom' in ob_data.columns:
    #                         high_diff = ob_data['Top'] - self.historical_data['high'].iloc[-1]
    #                         low_diff = self.historical_data['low'].iloc[-1] - ob_data['Bottom']
    #                         ob_data['OB'] = np.where(high_diff > low_diff, 1, -1)
    #                     else:
    #                         ob_data['OB'] = 0
    #                 elif col in ['Top', 'Bottom']:
    #                     ob_data[col] = self.historical_data['close'].iloc[-1]
    #
    #         # Convert critical columns to float
    #         for col in required_cols:
    #             if col in ob_data.columns:
    #                 ob_data[col] = pd.to_numeric(ob_data[col], errors='coerce')
    #
    #         # Remove rows with NaN values after conversion
    #         ob_data = ob_data.dropna(subset=['Top', 'Bottom'])
    #
    #         if ob_data.empty:
    #             print("⚠️ validate_ob: DataFrame became empty after removing NaN values")
    #             return pd.DataFrame()
    #
    #         # Try to reindex with historical data
    #         try:
    #             # Use valid subset of indices for reindexing
    #             valid_indices = ob_data.index.intersection(self.historical_data.index)
    #             if len(valid_indices) == 0:
    #                 print("⚠️ validate_ob: No common indices with historical data")
    #                 # In this case, we can't reindex, so return the original data
    #                 return ob_data
    #
    #             ob_data = ob_data.loc[valid_indices]
    #         except Exception as idx_error:
    #             print(f"⚠️ validate_ob: Reindexing error: {idx_error}")
    #             # Continue with original data
    #
    #         # Get current price for mitigation check
    #         current_price = self.historical_data['close'].iloc[-1]
    #
    #         # Create mitigation conditions
    #         try:
    #             ob_data['Mitigated'] = np.where(
    #                 (ob_data['OB'] == 1) & (current_price > ob_data['Top']),
    #                 True,
    #                 np.where((ob_data['OB'] == -1) & (current_price < ob_data['Bottom']),
    #                          True, False)
    #             )
    #             return ob_data[~ob_data['Mitigated']]
    #         except Exception as mitigation_error:
    #             print(f"⚠️ validate_ob: Mitigation calculation error: {mitigation_error}")
    #             return ob_data
    #
    #     except Exception as e:
    #         print(f"⚠️ OB validation error: {e}")
    #         import traceback
    #         traceback.print_exc()
    #         return pd.DataFrame()

    def validate_bos(self, bos_data):
        """Validate BOS/CHOCH"""
        if not isinstance(bos_data, pd.DataFrame):
            print("⚠️ Invalid BOS data type")
            return pd.DataFrame()

        if bos_data.empty:
            return bos_data
        try:
            bos_data = bos_data.reindex(self.historical_data.index)
            current_price = self.historical_data['close'].iloc[-1]
            bos_data['Confirmed'] = np.where(
                ((bos_data['BOS'] == 1) & (current_price > bos_data['Level'])) |
                ((bos_data['BOS'] == -1) & (current_price < bos_data['Level'])),
                1, 0
            )
            return bos_data[bos_data['Confirmed'] == 1]
        except Exception as e:
            print(f"BOS validation error: {e}")
            return pd.DataFrame()

    def validate_liquidity(self, liq_data):
        """Validate Liquidity"""
        if not isinstance(liq_data, pd.DataFrame):
            print("⚠️ Invalid Liquidity data type")
            return pd.DataFrame()

        if liq_data.empty:
            return liq_data
        try:
            liq_data = liq_data.reindex(self.historical_data.index)
            current_price = self.historical_data['close'].iloc[-1]
            liq_data['Mitigated'] = np.where(
                (liq_data['Liquidity'] == 1) & (current_price > liq_data['Level']),
                True,
                np.where((liq_data['Liquidity'] == -1) & (current_price < liq_data['Level']),
                         True, False
                         ))
            return liq_data[~liq_data['Mitigated']]
        except Exception as e:
            print(f"Liquidity validation error: {e}")
            return pd.DataFrame()

    def process_smc_data(self):
        """Process all SMC data with index alignment"""
        try:
            if self.historical_data.empty or len(self.historical_data) < 2:
                 print("⚠️ Insufficient historical data for processing.")
                 return pd.DataFrame()

            # Ensure the index is unique before processing
            self.historical_data = self.historical_data[~self.historical_data.index.duplicated(keep='last')]

            # Get swings with INTEGER index relative to historical_data
            print("\n=== SMC Processing Start ===") # Added header
            swings = self.get_swing_data(self.historical_data)
            # print(f"SWINGS: {swings}")

            if swings.empty or len(swings) < 2:
                 print("⚠️ Swing data generation failed or insufficient. Cannot calculate dependent SMC.")
                 return pd.DataFrame()
            # print(f"[DEBUG]: Swings (integer index):\n{swings}") # Debug swings

            temp_data = self.historical_data.reset_index(drop=True)
            print(f"[DEBUG]: Processing temp_data shape: {temp_data.shape}")

            if not swings.empty:
                # Ensure index compatibility
                swings = swings.reset_index(drop=True).sort_index()
                try:
                    bos_raw = smc.bos_choch(temp_data, swings, close_break=True)
                except KeyError as e:
                    print(f"Index alignment error: {e}")
                    # Reindex with forward fill
                    aligned_swings = swings.reindex(temp_data.index)
                    bos_raw = smc.bos_choch(temp_data, aligned_swings, close_break=True)

            # --- SMC Calculations ---
            bos_data = pd.DataFrame()
            ob_data = pd.DataFrame()
            fvg_data = pd.DataFrame()
            liq_data = pd.DataFrame()

            # BOS - Uses integer index
            print("\n--- BOS Calculation ---") # Header
            try:
                # Ensure integer indices for both inputs
                temp_data_int = temp_data.reset_index(drop=True)
                temp_data_int.index = np.arange(len(temp_data_int), dtype=int)

                swings_int = swings.reset_index(drop=True)
                swings_int.index = np.arange(len(swings_int), dtype=int)

                bos_raw = smc.bos_choch(temp_data_int, swings_int, close_break=True)

                # Check raw output type and content
                if isinstance(bos_raw, pd.DataFrame):
                    print(f"[DEBUG]: bos_raw shape: {bos_raw.shape}, is_empty: {bos_raw.empty}")
                    if not bos_raw.empty:
                        # Ensure integer indices and filter valid ones
                        bos_indices = bos_raw.index.astype(int)
                        valid_mask = (bos_indices >= 0) & (bos_indices < len(self.historical_data))
                        valid_bos_indices = bos_indices[valid_mask]

                        if len(valid_bos_indices) > 0:
                            bos_raw_filtered = bos_raw.loc[valid_bos_indices]
                            bos_raw_filtered.index = self.historical_data.index[valid_bos_indices]
                            bos_raw_dedup = bos_raw_filtered[~bos_raw_filtered.index.duplicated(keep='last')]
                            print(f"[DEBUG]: bos_raw mapped/dedup shape: {bos_raw_dedup.shape}")
                            bos_data = self.validate_bos(bos_raw_dedup)
                            print(f"[DEBUG]: bos_data after validation shape: {bos_data.shape}")
                        else:
                            print("[DEBUG]: No valid BOS indices found.")
                # Handle non-dataframe return explicitly
                elif bos_raw is None:
                     print("⚠️ BOS function returned None.")
                else:
                     print(f"⚠️ BOS function returned non-DataFrame: type={type(bos_raw)}, value={bos_raw}")

            except Exception as e:
                print(f"💥 BOS Calculation Exception:")
                import traceback; traceback.print_exc() # Print full traceback

            # Order Blocks - Uses integer index
            print("\n--- OB Calculation ---") # Header
            try:
                # Make sure swings is properly formatted and has required columns
                if not isinstance(swings, pd.DataFrame) or swings.empty:
                    print("⚠️ Swings data is empty or not a DataFrame. Cannot calculate OB.")
                    ob_raw = pd.DataFrame()
                elif 'HighLow' not in swings.columns or 'Level' not in swings.columns:
                    print("⚠️ Swings data missing required columns ('HighLow' or 'Level')")
                    ob_raw = pd.DataFrame()
                else:
                    # Convert any numpy values to Python native types
                    swings_copy = swings.copy()
                    swings_copy['HighLow'] = swings_copy['HighLow'].astype(float)
                    swings_copy['Level'] = swings_copy['Level'].astype(float)
                    swings_copy = swings_copy.reset_index(drop=True)

                    # Print debugging information
                    # print(f"[DEBUG]: Swings data before OB calculation: \n{swings_copy.head()}")
                    print(f"[DEBUG]: Swings data shape: {swings_copy.shape}")
                    print(f"[DEBUG]: Temp data shape: {temp_data.shape}")

                    # Make the call with error handling
                    try:
                        ob_raw = smc.ob(temp_data, swings_copy, close_mitigation = False)
                    except TypeError as te:
                        print(f"⚠️ Type error in OB function: {te}")
                        # Try again with integer indices
                        temp_data_int = temp_data.reset_index(drop=True)
                        ob_raw = smc.ob(temp_data_int, swings_copy)
                    except Exception as e:
                        print(f"⚠️ Error in OB function: {e}")
                        ob_raw = pd.DataFrame()

                if isinstance(ob_raw, pd.DataFrame):
                    print(f"[DEBUG]: ob_raw shape: {ob_raw.shape}, is_empty: {ob_raw.empty}")
                    if not ob_raw.empty:
                        # Ensure indices are valid
                        try:
                            valid_indices = ob_raw.index[ob_raw.index < len(self.historical_data)]
                            ob_raw_filtered = ob_raw.loc[valid_indices]
                            if not ob_raw_filtered.empty:
                                ob_raw_filtered.index = self.historical_data.index[valid_indices]
                                ob_raw_dedup = ob_raw_filtered[~ob_raw_filtered.index.duplicated(keep='last')]
                                print(f"[DEBUG]: ob_raw mapped/dedup shape: {ob_raw_dedup.shape}")
                                ob_data = self.validate_ob(ob_raw_dedup)
                                print(f"[DEBUG]: ob_data after validation shape: {ob_data.shape}")
                            else:
                                print("[DEBUG]: ob_raw empty after index filtering.")
                                ob_data = pd.DataFrame()
                        except Exception as e:
                            print(f"⚠️ Error processing OB results: {e}")
                            ob_data = pd.DataFrame()
                    else:
                        print("[DEBUG]: ob_raw is empty")
                        ob_data = pd.DataFrame()
                elif ob_raw is None:
                    print("⚠️ OB function returned None.")
                    ob_data = pd.DataFrame()
                else:
                    print(f"⚠️ OB function returned non-DataFrame: type={type(ob_raw)}, value={ob_raw}")
                    ob_data = pd.DataFrame()
            except Exception as e:
                 print(f"💥 OB Calculation Exception:")
                 import traceback; traceback.print_exc()

            # Fair Value Gaps - Uses integer index
            print("\n--- FVG Calculation ---") # Header
            try:
                fvg_raw = smc.fvg(temp_data, join_consecutive=False)
                if isinstance(fvg_raw, pd.DataFrame):
                    print(f"[DEBUG]: fvg_raw shape: {fvg_raw.shape}, is_empty: {fvg_raw.empty}")
                    if not fvg_raw.empty:
                        valid_indices = fvg_raw.index[fvg_raw.index < len(self.historical_data)]
                        fvg_raw_filtered = fvg_raw.loc[valid_indices]
                        if not fvg_raw_filtered.empty:
                             fvg_raw_filtered.index = self.historical_data.index[valid_indices]
                             fvg_raw_dedup = fvg_raw_filtered[~fvg_raw_filtered.index.duplicated(keep='last')]
                             print(f"[DEBUG]: fvg_raw mapped/dedup shape: {fvg_raw_dedup.shape}")
                             fvg_data = self.validate_fvg(fvg_raw_dedup)
                             print(f"[DEBUG]: fvg_data after validation shape: {fvg_data.shape}")
                        else:
                             print("[DEBUG]: fvg_raw empty after index filtering.")
                elif fvg_raw is None:
                     print("⚠️ FVG function returned None.")
                else:
                     print(f"⚠️ FVG function returned non-DataFrame: type={type(fvg_raw)}, value={fvg_raw}")
            except Exception as e:
                 print(f"💥 FVG Calculation Exception:")
                 import traceback; traceback.print_exc()

            # Liquidity - Uses integer index
            print("\n--- Liquidity Calculation ---") # Header
            try:
                liq_raw = smc.liquidity(temp_data, swings, range_percent = 0.01)
                if isinstance(liq_raw, pd.DataFrame):
                     print(f"[DEBUG]: liq_raw shape: {liq_raw.shape}, is_empty: {liq_raw.empty}")
                     if not liq_raw.empty:
                        valid_indices = liq_raw.index[liq_raw.index < len(self.historical_data)]
                        liq_raw_filtered = liq_raw.loc[valid_indices]
                        if not liq_raw_filtered.empty:
                            liq_raw_filtered.index = self.historical_data.index[valid_indices]
                            liq_raw_dedup = liq_raw_filtered[~liq_raw_filtered.index.duplicated(keep='last')]
                            print(f"[DEBUG]: liq_raw mapped/dedup shape: {liq_raw_dedup.shape}")
                            liq_data = self.validate_liquidity(liq_raw_dedup)
                            print(f"[DEBUG]: liq_data after validation shape: {liq_data.shape}")
                        else:
                             print("[DEBUG]: liq_raw empty after index filtering.")

                elif liq_raw is None:
                     print("⚠️ Liquidity function returned None.")
                else:
                     print(f"⚠️ Liquidity function returned non-DataFrame: type={type(liq_raw)}, value={liq_raw}")
            except Exception as e:
                 print(f"💥 Liquidity Calculation Exception:")
                 import traceback; traceback.print_exc()

            # --- Combine Data ---
            print("\n--- Combining Data ---") # Header
            combined = self.historical_data.copy()
            dfs_to_join = {
                'bos_': bos_data, 'ob_': ob_data, 'fvg_': fvg_data, 'liq_': liq_data
            }

            for prefix, df_join in dfs_to_join.items():
                 if not df_join.empty:
                      df_join_unique = df_join[~df_join.index.duplicated(keep='last')]
                      if not df_join_unique.empty:
                           print(f"[DEBUG]: Joining {prefix[:-1]} data (shape: {df_join_unique.shape})")
                           combined = combined.join(df_join_unique.add_prefix(prefix), how='left')
                      else:
                           print(f"[DEBUG]: Skipping join for {prefix[:-1]} (empty after dedup)")
                 else:
                     print(f"[DEBUG]: Skipping join for {prefix[:-1]} (empty)")


            # combined = combined.ffill().fillna(0)
            print(f"[DEBUG]: Final combined shape before return: {combined.shape}")
            print("=== SMC Processing End ===") # Footer

            print("\n=== TEMPORAL VALIDATION ===")
            print(f"First timestamp: {self.historical_data.index[0]}")
            print(f"Last timestamp: {self.historical_data.index[-1]}")
            print(f"Is sorted: {self.historical_data.index.is_monotonic_increasing}")
            print(f"Time delta: {self.historical_data.index[-1] - self.historical_data.index[0]}")

            return combined

        except Exception as e:
            print(f"⛔ Top-level Processing error: {str(e)}")
            import traceback; traceback.print_exc()
            return pd.DataFrame()

    def calculate_position_size(self):
        """Calculate position size with 1% risk"""
        print("\n=== POSITION SIZE CALCULATION START ===")
        try:
            print("[DEBUG] Getting account balance...")
            balance = mt5.account_info().balance
            print(f"[DEBUG] Account balance: {balance}")

            if balance <= 0:
                print("[DEBUG] Invalid balance detected")
                raise ValueError("Balance cannot be zero or negative!")

            risk_amount = balance * ICT_CONFIG["RISK_PERCENT"] / 100
            print(f"[DEBUG] Risk amount ({ICT_CONFIG['RISK_PERCENT']}%): {risk_amount}")

            risk_pips = 20  # 20 pips
            print(f"[DEBUG] Risk in pips: {risk_pips}")

            pip_value = 1 / self.pip_size  # Për BTCUSD 1 pip = 0.1
            print(f"[DEBUG] Pip value: {pip_value}")

            size = risk_amount / (risk_pips * pip_value)
            final_size = round(max(size, 0.01), 2)
            print(f"[DEBUG] Calculated position size: {size}")
            print(f"[DEBUG] Final position size (after rounding): {final_size}")
            print("=== POSITION SIZE CALCULATION END ===\n")
            return final_size

        except Exception as e:
            print("=== POSITION SIZE CALCULATION ERROR ===")
            print(f"Position size error: {e}")
            import traceback
            print("[DEBUG] Full traceback:")
            traceback.print_exc()
            print("=== POSITION SIZE CALCULATION ERROR END ===\n")
        return 0.01

    # def generate_signals(self):
    #     """Generate trading signals - MODIFIED VERSION"""
    #     processed = self.process_smc_data()
    #     print(processed)
    #     if processed.empty:
    #         print(pd.Series())
    #         return pd.Series()
    #
    #     try:
    #         # === EXPLICIT TYPE CONVERSION ===
    #         signal_columns = [
    #             'bos_BOS', 'bos_CHOCH', 'ob_OB',
    #             'fvg_FVG', 'liq_Liquidity'
    #         ]
    #
    #         for col in signal_columns:
    #             processed[col] = (
    #                 pd.to_numeric(processed[col], errors='coerce')
    #                 .fillna(0)
    #                 .astype(int)
    #             )
    #
    #         # === MODIFIED SIGNAL CONDITIONS ===
    #         buy_conditions = (
    #                 (processed['bos_BOS'] == 1) &
    #                 (processed['bos_CHOCH'] == 1) &
    #                 (processed['ob_OB'] == 1) &
    #                 (processed['fvg_FVG'] == 1) &
    #                 (processed['liq_Liquidity'] == 1)
    #         )
    #
    #
    #         sell_conditions = (
    #                 (processed['bos_BOS'] == -1) &
    #                 (processed['bos_CHOCH'] == -1) &
    #                 (processed['ob_OB'] == -1) &
    #                 (processed['fvg_FVG'] == -1) &
    #                 (processed['liq_Liquidity'] == -1)
    #         )
    #
    #
    #         # === SWING FALLBACK ===
    #         swings = self.get_swing_data(processed)
    #         if not swings.empty:
    #             last_swing = swings.iloc[-1]
    #             current_close = processed['close'].iloc[-1]
    #
    #             if last_swing['HighLow'] == 1:  # Last swing high
    #                 sell_conditions |= (current_close < last_swing['Level'])
    #             elif last_swing['HighLow'] == -1:  # Last swing low
    #                 buy_conditions |= (current_close > last_swing['Level'])
    #
    #         # === FINAL SIGNAL GENERATION ===
    #         processed['signal'] = 0
    #         processed.loc[buy_conditions, 'signal'] = 1
    #         processed.loc[sell_conditions, 'signal'] = -1
    #
    #         # Remove false signals at boundaries
    #         processed['signal'] = processed['signal'].where(
    #             abs(processed['close'].pct_change()) > 0.0001,  # Filter < 0.01% moves
    #             0
    #         )
    #
    #         return processed['signal'].ffill().fillna(0)
    #
    #     except Exception as e:
    #         print(f"Signal generation error: {e}")
    #         return pd.Series()

    # def generate_signals(self):
    #     """Generate trading signals using all SMC columns with proper mitigation checks"""
    #     if DEBUG_MODE:
    #         print("\n=== SIGNAL GENERATION ===")
    #
    #     processed = self.process_smc_data()
    #     if processed is None or processed.empty:
    #         return pd.Series()
    #
    #     # Add integer position column for index-based calculations
    #     processed = processed.reset_index(drop=False)
    #     processed['_position'] = np.arange(len(processed))
    #
    #     # Ensure all required SMC columns exist with proper defaults
    #     smc_columns = {
    #         # From FVG calculation
    #         'fvg_FVG': 0, 'fvg_Top': np.nan, 'fvg_Bottom': np.nan, 'fvg_MitigatedIndex': -1,
    #         # From OB calculation
    #         'ob_OB': 0, 'ob_Top': np.nan, 'ob_Bottom': np.nan, 'ob_MitigatedIndex': -1,
    #         # From BOS/CHOCH calculation
    #         'bos_BOS': 0, 'bos_CHOCH': 0, 'bos_Level': np.nan, 'bos_BrokenIndex': -1,
    #         # From Liquidity calculation
    #         'liq_Liquidity': 0, 'liq_Level': np.nan, 'liq_Swept': -1
    #     }
    #
    #     for col, default in smc_columns.items():
    #         if col not in processed.columns:
    #             processed[col] = default
    #             if DEBUG_MODE:
    #                 print(f"Added missing column: {col}")
    #
    #     try:
    #         # ===== Mitigation Checks =====
    #         current_positions = processed['_position']
    #
    #         # FVG Validation
    #         fvg_active = (
    #              (processed['fvg_FVG'] != 0) &
    #              ((processed['fvg_MitigatedIndex'] == -1) |
    #              (current_positions < processed['fvg_MitigatedIndex']))
    #         )
    #
    #         # OB Validation
    #         ob_active = (
    #              (processed['ob_OB'] != 0) &
    #              ((processed['ob_MitigatedIndex'] == -1) |
    #              (current_positions < processed['ob_MitigatedIndex'])))
    #
    #         # BOS Validation
    #         bos_confirmed = (
    #                 (processed['bos_BOS'] != 0) &
    #                 (processed['bos_BrokenIndex'] != -1) &
    #                 (current_positions >= processed['bos_BrokenIndex']))
    #
    #         # Liquidity Validation
    #         liq_active = (
    #                 (processed['liq_Liquidity'] != 0) &
    #                 ((processed['liq_Swept'] == -1) |
    #                  (current_positions < processed['liq_Swept'])))
    #
    #         # ===== Signal Calculations =====
    #         # 1. FVG Signals (price must stay within FVG range)
    #         fvg_bull = (
    #                 (processed['fvg_FVG'] == 1) &
    #                 (processed['low'] > processed['fvg_Bottom']) &
    #                 fvg_active
    #         )
    #         fvg_bear = (
    #                 (processed['fvg_FVG'] == -1) &
    #                 (processed['high'] < processed['fvg_Top']) &
    #                 fvg_active
    #         )
    #
    #         # 2. OB Signals (price reacts at OB level)
    #         ob_bull = (
    #                 (processed['ob_OB'] == 1) &
    #                 (processed['low'] > processed['ob_Top']) &
    #                 ob_active
    #         )
    #         ob_bear = (
    #                 (processed['ob_OB'] == -1) &
    #                 (processed['high'] < processed['ob_Bottom']) &
    #                 ob_active
    #         )
    #
    #         # 3. BOS/CHOCH Signals (with confirmation)
    #         bos_bull = (
    #                 (processed['bos_BOS'] == 1) &
    #                 (processed['close'] > processed['bos_Level']) &
    #                 bos_confirmed
    #         )
    #         bos_bear = (
    #                 (processed['bos_BOS'] == -1) &
    #                 (processed['close'] < processed['bos_Level']) &
    #                 bos_confirmed
    #         )
    #
    #         # 4. Liquidity Signals (price approaches liquidity pool)
    #         liq_bull = (
    #                 (processed['liq_Liquidity'] == 1) &
    #                 (processed['low'] <= processed['liq_Level']) &
    #                 liq_active
    #         )
    #         liq_bear = (
    #                 (processed['liq_Liquidity'] == -1) &
    #                 (processed['high'] >= processed['liq_Level']) &
    #                 liq_active
    #         )
    #
    #         # ===== Combine Signals =====
    #         buy_signals = fvg_bull | ob_bull | bos_bull | liq_bull
    #         sell_signals = fvg_bear | ob_bear | bos_bear | liq_bear
    #
    #         # ===== Swing Structure Filter =====
    #         swings = self.get_swing_data(processed)
    #         if len(swings) >= 2:
    #             last_swing = swings.iloc[-1]
    #         if last_swing['HighLow'] == 1:  # Last swing high
    #             sell_signals &= processed['close'] < last_swing['Level']
    #         elif last_swing['HighLow'] == -1:  # Last swing low
    #             buy_signals &= processed['close'] > last_swing['Level']
    #
    #         # ===== Apply Signals =====
    #         processed['signal'] = 0
    #         processed.loc[buy_signals, 'signal'] = 1
    #         processed.loc[sell_signals, 'signal'] = -1
    #
    #         # ===== Validation =====
    #         if DEBUG_MODE:
    #             print("\n=== VALID SIGNALS ===")
    #         print(processed[['datetime', 'signal', 'fvg_FVG', 'ob_OB',
    #                          'bos_BOS', 'liq_Liquidity', 'close']].tail())
    #
    #         return processed.set_index('datetime')['signal']
    #
    #     except Exception as e:
    #         if DEBUG_MODE:
    #             print(f"Signal generation error: {str(e)}")
    #             import traceback
    #             traceback.print_exc()
    #         return pd.Series()

    def generate_signals(self):
        """Generate trading signals independently for each indicator"""
        if DEBUG_MODE:
            print("\n=== SIGNAL GENERATION ===")

        processed = self.process_smc_data()
        if processed is None:
            if DEBUG_MODE:
                print("Process SMC data returned None")
            return pd.Series()

        if processed.empty:
            if DEBUG_MODE:
                print("No processed data available for signal generation")
            return pd.Series()

        processed['signal'] = 0
        swing_signal = 0

        try:
            swings = self.get_swing_data(processed)
            if len(swings) >= 2:
                last_two = swings.iloc[-30:]
                first_swing = last_two.iloc[0]
                second_swing = last_two.iloc[1]

                if first_swing['HighLow'] == -1 and second_swing['HighLow'] == 1:
                    swing_signal = 1
                elif first_swing['HighLow'] == 1 and second_swing['HighLow'] == -1:
                    swing_signal = -1
        except Exception as e:
            if DEBUG_MODE:
                print(f"Swing signal error: {e}")

        required_columns = {
            'ob_OB': 0,
            'ob_Top': 0,
            'ob_Bottom': 0,
            'fvg_FVG': 0,
            'fvg_MitigatedIndex': 0,
            'bos_BOS': 0,
            'bos_CHOCH': 0,
            'bos_Level': 0,
            'liq_Liquidity': 0,
            'liq_Swept': 0
        }
        for col, default in required_columns.items():
            if col not in processed.columns:
                processed[col] = default

        try:
            # Convert index and mitigation columns to datetime with UTC for safe comparison
            index_series = processed.index.to_series()
            fvg_mit = pd.to_datetime(processed['fvg_MitigatedIndex'], utc=True)
            liq_mit = pd.to_datetime(processed['liq_Swept'], utc=True)

            # ==== Indipendent Signals with Mitigation ====
            # 1. Fair Value Gap (FVG) only if it's not mitigated
            fvg_unmitigated = processed['fvg_FVG'].notna() & (
                    processed['fvg_MitigatedIndex'].isna() | (index_series < fvg_mit)
            )
            fvg_signal = np.where(
                (processed['fvg_FVG'] == 1) & fvg_unmitigated, 1,
                np.where(
                    (processed['fvg_FVG'] == -1) & fvg_unmitigated, -1,
                    0
                )
            )

            # 2. Order Block (OB)
            ob_signal = np.where(
                (processed['ob_OB'] == 1) & (processed['close'] > processed['ob_Top']), 1,
                np.where(
                    (processed['ob_OB'] == -1) & (processed['close'] < processed['ob_Bottom']),
                    -1, 0
                )
            )

            # 3. Liquidity only if it's not swept
            liq_unmitigated = processed['liq_Liquidity'].notna() & (
                    processed['liq_Swept'].isna() | (index_series < liq_mit)
            )
            liq_signal = np.where(
                (processed['liq_Liquidity'] == 1) & liq_unmitigated, 1,
                np.where(
                    (processed['liq_Liquidity'] == -1) & liq_unmitigated, -1,
                    0
                )
            )

            # 4. BOS & CHOCH (structure)
            bos_signal = np.where(
                (processed['bos_BOS'] == 1) & (processed['close'] > processed['bos_Level']), 1,
                np.where(
                    (processed['bos_BOS'] == -1) & (processed['close'] < processed['bos_Level']),
                    -1, 0
                )
            )
            choch_signal = np.where(
                (processed['bos_CHOCH'] == 1) & (processed['close'] > processed['bos_Level']), 1,
                np.where(
                    (processed['bos_CHOCH'] == -1) & (processed['close'] < processed['bos_Level']),
                    -1, 0
                )
            )

            # ==== Structure confirmation and take 2 last swings ====
            swings = self.get_swing_data(processed)
            if len(swings) >= 2:
                # Take swings
                recent = swings.iloc[-2:]
                # Identify High and Low swing
                last_high = recent.loc[recent['HighLow'] == 1, 'Level'].iloc[-1]
                last_low = recent.loc[recent['HighLow'] == -1, 'Level'].iloc[-1]
                midpoint = (last_high + last_low) / 2
            else:
                midpoint = np.nan

            # ==== PD Array added (Premium/Discount) based ICT ====
            pd_ok = np.zeros(len(processed), dtype=bool)
            if not np.isnan(midpoint):
                pd_ok = np.where(
                    (swing_signal == 1) & (processed['close'] < midpoint), True,
                    np.where(
                        (swing_signal == -1) & (processed['close'] > midpoint), True,
                        False
                    )
                )

            # ==== Signals combinations ====
            # buy_condition = (
            #         # (swing_signal == 1) & pd_ok & (
            #         (fvg_signal == 1) | (ob_signal == 1) |
            #         (liq_signal == 1) | (bos_signal == 1) | (choch_signal == 1)
            #     # )
            # )
            # sell_condition = (
            #         # (swing_signal == -1) & pd_ok & (
            #         (fvg_signal == -1) | (ob_signal == -1) |
            #         (liq_signal == -1) | (bos_signal == -1) | (choch_signal == -1)
            #     # )
            # )

            buy_condition = ((fvg_signal == 1) | (ob_signal == 1) | (liq_signal == 1)
                             | (bos_signal == 1) | (choch_signal == 1))

            sell_condition = ((fvg_signal == -1) | (ob_signal == -1) | (liq_signal == -1)
                              | (bos_signal == -1) | (choch_signal == -1))

            processed.loc[buy_condition, 'signal'] = 1
            processed.loc[sell_condition, 'signal'] = -1

            # ==== Kod validimi për testim ====
            if len(processed) > 1:
                last_valid_idx = processed.index[-2]
                valid_signals = processed.loc[:last_valid_idx].copy()

                if DEBUG_MODE:
                    print("\n=== DEBUG SIGNALS (ME MITIGATION) ===")
                    print(valid_signals[
                              [ 'signal',
                               'fvg_FVG', 'fvg_Top', 'fvg_Bottom', 'fvg_MitigatedIndex',
                               'ob_OB', 'ob_Top', 'ob_Bottom', 'ob_MitigatedIndex',
                               'bos_BOS', 'bos_Level',
                               'liq_Liquidity', 'liq_Level', 'liq_Swept',
                               'close']
                          ])
                return valid_signals['signal']
            return processed['signal']

        except Exception as e:
            if DEBUG_MODE:
                print("Signal generation error:", e)
            return pd.Series()


# ========== Enhanced ICT Strategy Implementation ==========
class ICTExecution:
    def __init__(self, symbol, timeframe):
        self.symbol = symbol
        self.max_positions = 3
        self.trader = SMCTrader(symbol, timeframe)

    def manage_trades(self):
        """Manage open positions"""
        trader = SMCTrader(SYMBOL, TIMEFRAME)
        try:
            positions = mt5.positions_get(symbol=self.symbol)
            if positions is None:
                return

            for position in positions:
                try:
                    pip = mt5.symbol_info(self.symbol).point * 10
                    current_price = mt5.symbol_info_tick(
                        self.symbol).ask if position.type == 0 else mt5.symbol_info_tick(
                        self.symbol).bid

                    new_sl = position.sl
                    if position.type == 0:  # Long
                        profit_pips = (current_price - position.price_open) / pip
                        # Get latest swing low for stop loss
                        swings = trader.get_swing_data(trader.historical_data)
                        if not swings.empty and len(swings[swings['HighLow'] == -1]) > 0:
                            swing_low = swings[swings['HighLow'] == -1]['Level'].values[-1]

                            if profit_pips >= 30:
                                min_sl = position.price_open + (1.5 * pip)  # Breakeven + 1.5 pips
                                trailing_sl = current_price - (20 * pip)  # Trail 50 pips behind
                                swing_sl = min(swing_low, position.sl)  # Use swing low if tighter
                                new_sl = max(min_sl, trailing_sl, swing_sl)
                    else:  # Short
                        profit_pips = (position.price_open - current_price) / pip
                        # Get latest swing high for stop loss
                        swings = trader.get_swing_data(trader.historical_data)
                        if not swings.empty and len(swings[swings['HighLow'] == 1]) > 0:
                            swing_high = swings[swings['HighLow'] == 1]['Level'].values[-1]

                            if profit_pips >= 30:
                                min_sl = position.price_open - (1.5 * pip)  # Breakeven + 1.5 pips
                                trailing_sl = current_price + (20 * pip)  # Trail 50 pips behind
                                swing_sl = max(swing_high, position.sl)  # Use swing high if tighter
                                new_sl = min(min_sl, trailing_sl, swing_sl)

                    if new_sl != position.sl:
                        request = {
                            "action": mt5.TRADE_ACTION_SLTP,
                            "position": position.ticket,
                            "sl": new_sl,
                            "deviation": 20
                        }
                        result = mt5.order_send(request)
                        if result.retcode != mt5.TRADE_RETCODE_DONE:
                            print(f"Failed to update SL: {result.comment}")
                except Exception as pe:
                    print(f"Position management error: {pe}")
                    continue

        except Exception as e:
            print(f"Trade management error: {e}")

    def execute_order(self, signal, price, sl, tp, size):
        """Execute trade order"""
        print("\n=== ORDER EXECUTION START ===")
        print(f"[DEBUG] Checking positions count...")
        open_trades = self.get_open_trades()
        print(f"[DEBUG] Current open trades: {open_trades}/{self.max_positions}")

        if open_trades >= self.max_positions:
            print("[DEBUG] Maximum positions reached, skipping order")
            print("=== ORDER EXECUTION CANCELLED ===\n")
            return None

        order_type = mt5.ORDER_TYPE_BUY if signal == 1 else mt5.ORDER_TYPE_SELL
        print(f"[DEBUG] Order type: {'BUY' if signal == 1 else 'SELL'}")

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": round(size, 2),
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": 10032023,
            "comment": "ICT Strategy",
            "type_time": mt5.ORDER_TIME_GTC,
        }
        print("[DEBUG] Order request details:")
        for key, value in request.items():
            print(f"[DEBUG] - {key}: {value}")

        if not DRY_RUN:
            print("[DEBUG] Sending order to MT5...")
            result = mt5.order_send(request)
            if DEBUG_MODE:
                print(f"[DEBUG] MT5 response: {result}")
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    print("[DEBUG] Order executed successfully")
                else:
                    print(f"[DEBUG] Order failed with code: {result.retcode}")
            print("=== ORDER EXECUTION END ===\n")
            return result
        else:
            print(f"DRY RUN: {order_type} {self.symbol} {size} lots @ {price}")
            print("=== ORDER EXECUTION END (DRY RUN) ===\n")
        return None

    def get_open_trades(self):
        """Count open positions"""
        try:
            return len(mt5.positions_get(symbol=self.symbol) or [])
        except Exception as e:
            print(f"Position count error: {e}")
            return 0
# def main_loop():
#     if not initialize_mt5():
#         return
#
#     trader = SMCTrader(SYMBOL, TIMEFRAME)
#     executor = ICTExecution(SYMBOL)
#     print("🚀 Starting Trading Bot...")
#
#     # Load initial data
#     start_time = time.time()
#     new_data = fetch_realtime_data()  # Re-fetch in loop
#     while time.time() - start_time < 60:
#         df = fetch_realtime_data()
#         if not df.empty:
#             # trader.historical_data = df.iloc[-NUM_CANDLES:]
#             trader.historical_data = pd.concat([trader.historical_data, new_data]) \
#                 .sort_index() \
#                 .last(f'{NUM_CANDLES}min')
#             if len(trader.historical_data) >= NUM_CANDLES:
#                 print(f"✅ Loaded {len(trader.historical_data)} candles")
#                 break
#         time.sleep(5)
#     else:
#         print("❌ Data load timeout")
#         mt5.shutdown()
#         return
#
#     # Main trading loop
#     while True:
#         try:
#             if not trader.is_trading_session():
#                 if DEBUG_MODE:
#                     print("💤 Outside trading session")
#                 time.sleep(60)
#                 continue
#
#             new_data = fetch_realtime_data()
#             if new_data.empty:
#                 time.sleep(1)
#                 continue
#
#             # Update historical data
#             trader.historical_data = pd.concat([
#                 trader.historical_data,
#                 new_data
#             ]).drop_duplicates().iloc[-NUM_CANDLES:]
#
#             # Generate and execute signals
#             signals = trader.generate_signals()
#             current_signal = signals.iloc[-1] if not signals.empty else 0
#
#             if current_signal != 0:
#                 price = mt5.symbol_info_tick(SYMBOL).ask if current_signal == 1 else mt5.symbol_info_tick(SYMBOL).bid
#                 swings = trader.get_swing_data(trader.historical_data)
#
#                 if current_signal == 1:
#                     sl = swings[swings['HighLow'] == -1]['Level'].values[-1]
#                     tp = price + 2 * (price - sl)
#                 else:
#                     sl = swings[swings['HighLow'] == 1]['Level'].values[-1]
#                     tp = price - 2 * (sl - price)
#
#                 size = trader.calculate_position_size()
#                 if abs(price - sl) / trader.pip_size > 10:
#                     if DEBUG_MODE:
#                         print(f"🚨 Signal {current_signal} | Price: {price} | SL: {sl} | TP: {tp} | Size: {size}")
#                     executor.execute_order(current_signal, price, sl, tp, size)
#
#             executor.manage_trades()
#             time.sleep(5)
#
#         except KeyboardInterrupt:
#             print("\n🛑 Bot stopped by user")
#             break
#         except Exception as e:
#             print(f"Main loop error: {e}")
#             time.sleep(5)
#
#     mt5.shutdown()
#     print("✅ Disconnected from MT5")
def main_loop():
    if not initialize_mt5():
        return

    trader = SMCTrader(SYMBOL, TIMEFRAME)
    executor = ICTExecution(SYMBOL, TIMEFRAME)
    print("🚀 Starting Trading Bot...")

    # Calculate timeframe parameters
    timeframe_minutes = int(TIMEFRAME[1:])  # Extract numeric value from TIMEFRAME (e.g., M5 -> 5)
    total_minutes = NUM_CANDLES * timeframe_minutes

    # Load initial data with time-based window
    start_time = time.time()

    # In the initial data loading section:
    while time.time() - start_time < 60:
        new_data = fetch_realtime_data()
        if not new_data.empty:
            trader.historical_data = pd.concat([trader.historical_data, new_data]) \
                                         .drop_duplicates() \
                                         .sort_index() \
                                         .iloc[-NUM_CANDLES * 2:]  # Load extra data buffer

            if len(trader.historical_data) >= NUM_CANDLES:
                print(f"✅ Loaded {len(trader.historical_data)} candles")
                break
        time.sleep(5)
    else:
        print("❌ Data load timeout")
        mt5.shutdown()
        return

    # Main trading loop
    while True:
        try:
            # if not trader.is_trading_session():
            #     if DEBUG_MODE:
            #         print("💤 Outside trading session")
            #     time.sleep(60)
            #     continue

            new_data = fetch_realtime_data()
            if new_data.empty:
                time.sleep(1)
                continue

            # Update historical data with time-based window
            combined = pd.concat([trader.historical_data, new_data]) \
                .sort_index() \
                .drop_duplicates()

            if not combined.empty:
                cutoff = combined.index[-1] - pd.Timedelta(minutes=total_minutes)
                trader.historical_data = combined.loc[cutoff:]

            if len(trader.historical_data) < 10:  # Minimum candles for processing
                print("⏳ Waiting for historical data...")
                time.sleep(5)
                continue

            # Generate and execute signals
            signals = trader.generate_signals()
            current_signal = signals.iloc[-1] if not signals.empty else 0

            if current_signal != 0:
                price = mt5.symbol_info_tick(SYMBOL).ask if current_signal == 1 else mt5.symbol_info_tick(SYMBOL).bid
                swings = trader.get_swing_data(trader.historical_data)

                # Validate swings exist
                if not swings.empty:
                    if current_signal == 1:
                        # Get swing lows and handle cases with insufficient data
                        swing_lows = swings[swings['HighLow'] == -1]['Level'].values
                        if len(swing_lows) > 1:
                            sl = swing_lows[-2]  # Use second-to-last swing low if available
                        else:
                            sl = swing_lows[-1] if len(swing_lows) > 0 else (price - 20 * trader.pip_size)
                        tp = price + 2 * (price - sl)
                    else:
                        # Get swing highs and handle cases with insufficient data
                        swing_highs = swings[swings['HighLow'] == 1]['Level'].values
                        if len(swing_highs) > 1:
                            sl = swing_highs[-2]  # Use second-to-last swing high if available
                        else:
                            sl = swing_highs[-1] if len(swing_highs) > 0 else (price + 20 * trader.pip_size)
                        tp = price - 2 * (sl - price)

                    size = trader.calculate_position_size()
                    if abs(price - sl) / trader.pip_size > 10:
                        if DEBUG_MODE:
                            print(f"🚨 Signal {current_signal} | Price: {price} | SL: {sl} | TP: {tp} | Size: {size}")
                        executor.execute_order(current_signal, price, sl, tp, size)

            executor.manage_trades()
            time.sleep(5)

        except KeyboardInterrupt:
            print("\n🛑 Bot stopped by user")
            break
        except Exception as e:
            print(f"Main loop error: {e}")
            time.sleep(5)

    mt5.shutdown()
    print("✅ Disconnected from MT5")

if __name__ == "__main__":
    main_loop()