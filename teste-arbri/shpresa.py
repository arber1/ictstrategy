"""
Smart Money Concepts (SMC) Trading Bot
This module implements SMC trading strategies using MetaTrader5.
"""

try:
    import MetaTrader5 as mt5
except ImportError:
    raise ImportError("MetaTrader5 package is not installed. Please install it first.")

import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

import numpy as np
import time
from smartmoneyconcepts.smc import smc
from datetime import datetime, timedelta
from typing import Dict, Union, Optional

# ========== Configuration ==========
SYMBOL: str = "BTCUSD"
TIMEFRAME: str = "M15"
NUM_CANDLES: int = 500
DRY_RUN: bool = False
DEBUG_MODE: bool = True

ICT_CONFIG: Dict[str, Union[Dict, float]] = {
    "SWING_LENGTHS": {
        "M1": 75,
        "M5": 10,
        "M15": 30,
        "H1": 7,
        "D1": 5
    },
    "FVG_EXPIRATION": {
        "M1": timedelta(hours=4),
        "M5": timedelta(hours=6),
        "M15": timedelta(hours=72),
        "H1": timedelta(days=1),
        "D1": timedelta(days=3)
    },
    "TRADING_SESSIONS": {
        "London": (7, 16),
        "NewYork": (13, 22)
    },
    "RISK_PERCENT": 1.0
}


def initialize_mt5() -> bool:
    """Initialize connection to MetaTrader 5 terminal."""
    print("\n=== MT5 INITIALIZATION START ===")
    try:
        if not mt5.initialize():
            print("❌ Failed to initialize MT5: Connection error")
            return False

        account_info = mt5.account_info()
        if account_info is None:
            print("❌ Failed to get account info")
            return False

        print(f"✅ Successfully connected to MT5 (Server: {account_info.server})")
        print("=== MT5 INITIALIZATION END ===\n")
        return True

    except Exception as e:
        print(f"MT5 initialization error: {str(e)}")
        return False


def fetch_realtime_data():
    """Fetch live data from MT5 with UTC timezone."""
    try:
        tf = getattr(mt5, f'TIMEFRAME_{TIMEFRAME}')
        rates = mt5.copy_rates_from_pos(SYMBOL, tf, 0, NUM_CANDLES)
        if rates is None or len(rates) == 0:
            return pd.DataFrame()

        df = pd.DataFrame(rates)
        df.columns = [col.lower().strip() for col in df.columns]
        required_cols = ['time', 'open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            return pd.DataFrame()

        df.rename(columns={'time': 'datetime'}, inplace=True)
        df['datetime'] = pd.to_datetime(df['datetime'], unit='s', utc=True)
        df.set_index('datetime', inplace=True)
        result = df[['open', 'high', 'low', 'close', 'tick_volume']].rename(columns={'tick_volume': 'volume'})
        return result if not result.empty else pd.DataFrame()

    except Exception as e:
        print(f"Data error: {str(e)}")
        return pd.DataFrame()


class SMCTrader:
    def __init__(self, symbol, timeframe):
        self.symbol = symbol
        self.timeframe = timeframe
        self.swing_length = ICT_CONFIG["SWING_LENGTHS"][timeframe]
        self.fvg_expiration = ICT_CONFIG["FVG_EXPIRATION"][timeframe]
        self.historical_data = pd.DataFrame()
        self.pip_size = mt5.symbol_info(symbol).point * 10

    def get_swing_data(self, df):
        """Get all swings with zero division protection."""
        try:
            if df.empty or len(df) < 2:
                return pd.DataFrame({
                    'HighLow': [1, -1],
                    'Level': [df['high'].iloc[-1], df['low'].iloc[-1]],
                    'Confirmed': [True, True],
                    'PairIndex': [-1, -1],
                    'Retracement': [0.0, 0.0]
                }, index=[df.index[-1], df.index[-1]])

            temp = df.reset_index()
            calculated_swing_length = min(self.swing_length, len(temp) - 2)
            calculated_swing_length = max(calculated_swing_length, 2)

            swings = smc.swing_highs_lows(temp[['open', 'high', 'low', 'close']],
                                          swing_length=calculated_swing_length)

            all_swings_list = []
            prev_swing = None
            for idx, row in swings.iterrows():
                if prev_swing is not None:
                    try:
                        if prev_swing['HighLow'] == 1 and row['HighLow'] == -1:
                            denominator = prev_swing['Level'] - row['Level']
                            current_low = temp.loc[idx, 'low']
                            retracement = (prev_swing['Level'] - current_low) / denominator if denominator != 0 else 0.0
                        elif prev_swing['HighLow'] == -1 and row['HighLow'] == 1:
                            denominator = row['Level'] - prev_swing['Level']
                            current_high = temp.loc[idx, 'high']
                            retracement = (current_high - prev_swing[
                                'Level']) / denominator if denominator != 0 else 0.0
                        else:
                            retracement = 0.0
                    except KeyError:
                        retracement = 0.0
                else:
                    retracement = 0.0

                all_swings_list.append({
                    'HighLow': row['HighLow'],
                    'Level': row['Level'],
                    'Confirmed': True,
                    'PairIndex': prev_swing.name if prev_swing is not None else -1,
                    'Retracement': retracement
                })
                prev_swing = row

            return pd.DataFrame(all_swings_list).set_index(df.index)

        except Exception as e:
            print(f"Swing detection error: {str(e)}")
            return pd.DataFrame({
                'HighLow': [1, -1],
                'Level': [df['high'].iloc[-1], df['low'].iloc[-1]],
                'Confirmed': [True, True],
                'PairIndex': [-1, -1],
                'Retracement': [0.0, 0.0]
            }, index=[df.index[-1], df.index[-1]])

    def validate_fvg(self, fvg_data):
        """Validate FVGs"""
        if not isinstance(fvg_data, pd.DataFrame):
            print("⚠️ Invalid FVG data type")
            return pd.DataFrame()

        if fvg_data.empty:
            return pd.DataFrame()

        try:
            # Reindex to match historical data
            fvg_data = fvg_data.reindex(self.historical_data.index)

            # Get current time and last close price
            current_time = self.historical_data.index[-1]
            last_close = self.historical_data['close'].iloc[-1]

            # Filter FVGs within expiration period
            valid_fvg = fvg_data[
                (current_time - fvg_data.index) <= self.fvg_expiration
                ].copy()

            # Mark mitigated FVGs
            valid_fvg['Mitigated'] = np.where(
                (valid_fvg['FVG'] == 1) & (last_close < valid_fvg['Bottom']), True,
                np.where(
                    (valid_fvg['FVG'] == -1) & (last_close > valid_fvg['Top']), True, False
                )
            )

            # Return only non-mitigated FVGs
            active_fvgs = valid_fvg[~valid_fvg['Mitigated']].copy()

            if not active_fvgs.empty:
                print(f"Active FVGs found: {len(active_fvgs)}")
                return active_fvgs

            return pd.DataFrame()

        except Exception as e:
            print(f"FVG validation error: {e}")
            return pd.DataFrame()

    def validate_retracements(self, swings: pd.DataFrame) -> pd.DataFrame:
        """Validate retracements with enhanced error handling and validation"""
        try:
            if swings.empty or len(swings) < 2:
                if DEBUG_MODE:
                    print("⚠️ Insufficient swing data for retracement calculation")
                return pd.DataFrame()

            # Validate required columns
            required_cols = ['HighLow', 'Level']
            if not all(col in swings.columns for col in required_cols):
                print("⚠️ Missing required columns for retracement calculation")
                return pd.DataFrame()

            retracements = smc.retracements(self.historical_data, swings)
            if retracements is None or retracements.empty:
                if DEBUG_MODE:
                    print("⚠️ No retracements calculated")
                return pd.DataFrame()

            # Validate index bounds
            valid_mask = np.logical_and(
                retracements.index >= 0,
                retracements.index < len(self.historical_data)
            )

            # Additional validation for retracement values
            validated_ret = retracements[valid_mask].copy()
            if not validated_ret.empty:
                validated_ret['Valid'] = np.logical_and(
                    validated_ret['Level'] > self.historical_data['low'].min(),
                    validated_ret['Level'] < self.historical_data['high'].max()
                )
                return validated_ret[validated_ret['Valid']]

            return pd.DataFrame()

        except Exception as e:
            print(f"Retracement validation error: {str(e)}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    def validate_ob(self, ob_data):
        """Validate Order Blocks"""
        if not isinstance(ob_data, pd.DataFrame):
            print("⚠️ validate_ob: Input is not a DataFrame")
            return pd.DataFrame()
        if ob_data.empty:
            print("⚠️ validate_ob: Input DataFrame is empty")
            return pd.DataFrame()

        try:
            # Get retracement data
            swings = self.get_swing_data(self.historical_data)
            retracements = self.validate_retracements(swings)

            # Check if required columns exist
            required_cols = ['OB', 'Top', 'Bottom', 'OBVolume', 'Percentage']
            if not all(col in ob_data.columns for col in required_cols):
                missing = [col for col in required_cols if col not in ob_data.columns]
                print(f"⚠️ validate_ob: Missing columns: {missing}")

                # Try to create missing columns with defaults if possible
                for col in missing:
                    if col == 'OB':
                        if 'Top' in ob_data.columns and 'Bottom' in ob_data.columns:
                            high_diff = ob_data['Top'] - self.historical_data['high'].iloc[-1]
                            low_diff = self.historical_data['low'].iloc[-1] - ob_data['Bottom']
                            ob_data['OB'] = np.where(high_diff > low_diff, 1, -1)
                        else:
                            ob_data['OB'] = 0
                    elif col in ['Top', 'Bottom']:
                        ob_data[col] = self.historical_data['close'].iloc[-1]
                    elif col == 'OBVolume':
                        # Sum of current and previous 2 volumes
                        ob_data['OBVolume'] = self.historical_data['volume'].rolling(3).sum()
                    elif col == 'Percentage':
                        # Calculate strength based on volume ratio
                        high_vol = self.historical_data['volume'].iloc[-2:].max()
                        low_vol = self.historical_data['volume'].iloc[-2:].min()
                        ob_data['Percentage'] = low_vol / high_vol

            # Convert critical columns to float
            for col in required_cols:
                if col in ob_data.columns:
                    ob_data[col] = pd.to_numeric(ob_data[col], errors='coerce')

            # Remove rows with NaN values after conversion
            ob_data = ob_data.dropna(subset=required_cols)

            if ob_data.empty:
                print("⚠️ validate_ob: DataFrame became empty after removing NaN values")
                return pd.DataFrame()

            # Try to reindex with historical data
            try:
                valid_indices = ob_data.index.intersection(self.historical_data.index)
                if len(valid_indices) == 0:
                    print("⚠️ validate_ob: No common indices with historical data")
                    return ob_data

                ob_data = ob_data.loc[valid_indices]
            except Exception as idx_error:
                print(f"⚠️ validate_ob: Reindexing error: {idx_error}")

            # Get current price for level check
            current_price = self.historical_data['close'].iloc[-1]

            # Add strength based on volume and retracement levels
            if not retracements.empty:
                # Get closest retracement level for each OB
                ob_data['RetracementLevel'] = 0.0
                ob_data['RetracementStrength'] = 0.0

                for idx in ob_data.index:
                    ob_price = (ob_data.loc[idx, 'Top'] + ob_data.loc[idx, 'Bottom']) / 2
                    ret_diffs = abs(retracements['Level'] - ob_price)
                    closest_ret = retracements.iloc[ret_diffs.argmin()]

                    # Store retracement info
                    ob_data.loc[idx, 'RetracementLevel'] = closest_ret['Level']

                    # Calculate strength based on retracement % and fibonacci ratios
                    ret_strength = 1.0
                    if closest_ret['CurrentRetracement%'] >= 0.618:
                        ret_strength = 1.5
                    elif closest_ret['CurrentRetracement%'] >= 0.5:
                        ret_strength = 1.2

                    ob_data.loc[idx, 'RetracementStrength'] = ret_strength

                # Combined strength calculation incorporating retracements
                ob_data['Strength'] = ob_data['Percentage'] * ob_data['OBVolume'] * ob_data['RetracementStrength']
            else:
                # Fallback to original strength calculation
                ob_data['Strength'] = ob_data['Percentage'] * ob_data['OBVolume']

            # Return OBs sorted by strength
            active_obs = ob_data.sort_values('Strength', ascending=False)
            if not active_obs.empty:
                print(f"Active OBs found: {len(active_obs)}\n{active_obs}")
            return active_obs

        except Exception as e:
            print(f"⚠️ OB validation error: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    def validate_bos(self, bos_data):
        """Validate BOS/CHOCH with swing highs/lows and price breaks"""
        if not isinstance(bos_data, pd.DataFrame):
            print("⚠️ Invalid BOS data type")
            return pd.DataFrame()

        if bos_data.empty:
            return bos_data

        try:
            # Reindex with historical data preserving original index
            bos_data = bos_data.reindex(self.historical_data.index)
            print(bos_data)

            current_price = self.historical_data['close'].iloc[-1]
            current_high = self.historical_data['high'].iloc[-1]
            current_low = self.historical_data['low'].iloc[-1]

            # Validate bull BOS/CHOCH break
            bull_break = np.where(
                (bos_data['BOS'] == 1) &
                ((current_price > bos_data['Level']) if bos_data.get('close_break', True) else (
                        current_high > bos_data['Level'])),
                1, 0
            )

            # Validate bear BOS/CHOCH break
            bear_break = np.where(
                (bos_data['BOS'] == -1) &
                ((current_price < bos_data['Level']) if bos_data.get('close_break', True) else (
                        current_low < bos_data['Level'])),
                1, 0
            )

            bos_data['Confirmed'] = bull_break | bear_break
            bos_data['BrokenIndex'] = np.where(bos_data['Confirmed'] == 1, bos_data.index, pd.NaT)

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
            range_percent = 0.01

            # Group highs and lows that are within range_percent of each other
            high_levels = liq_data[liq_data['Liquidity'] == 1]['Level']
            low_levels = liq_data[liq_data['Liquidity'] == -1]['Level']

            # Function to group levels within range
            def group_levels(levels):
                groups = []
                if len(levels) > 0:
                    current_group = [levels.iloc[0]]
                    for level in levels.iloc[1:]:
                        if abs(level - current_group[0]) / current_group[0] <= range_percent:
                            current_group.append(level)
                        else:
                            if len(current_group) > 1:
                                groups.append(current_group)
                            current_group = [level]
                    if len(current_group) > 1:
                        groups.append(current_group)
                return groups

            # Get liquidity groups
            high_groups = group_levels(high_levels)
            low_groups = group_levels(low_levels)

            # Create validated liquidity DataFrame
            valid_liq = []

            # Add bullish liquidity levels
            for group in high_groups:
                level = sum(group) / len(group)
                swept = current_price > level
                valid_liq.append({
                    'Liquidity': 1,
                    'Level': level,
                    'End': liq_data.index[high_levels[high_levels.isin(group)].index[-1]],
                    'Swept': liq_data.index[-1] if swept else None
                })

            # Add bearish liquidity levels
            for group in low_groups:
                level = sum(group) / len(group)
                swept = current_price < level
                valid_liq.append({
                    'Liquidity': -1,
                    'Level': level,
                    'End': liq_data.index[low_levels[low_levels.isin(group)].index[-1]],
                    'Swept': liq_data.index[-1] if swept else None
                })

            if len(valid_liq) > 0:
                return pd.DataFrame(valid_liq)
            print(f"Valid liquidity: {len(valid_liq)}\n{valid_liq}")
            return pd.DataFrame()

        except Exception as e:
            print(f"Liquidity validation error: {e}")
            return pd.DataFrame()

    def process_smc_data(self):
        """Main processing with enhanced error handling."""
        try:
            if self.historical_data.empty:
                return pd.DataFrame()

            self.historical_data = self.historical_data[~self.historical_data.index.duplicated(keep='last')]
            swings = self.get_swing_data(self.historical_data)

            # BOS/CHOCH Calculation
            bos_data = pd.DataFrame()
            if len(swings) >= 4:
                try:
                    bos_raw = smc.bos_choch(self.historical_data.reset_index(drop=True),
                                            swings.reset_index(drop=True),
                                            close_break=True)
                    if isinstance(bos_raw, pd.DataFrame):
                        valid_indices = bos_raw.index[bos_raw.index < len(self.historical_data)]
                        bos_data = bos_raw.loc[valid_indices]
                except Exception as e:
                    print(f"BOS calculation error: {e}")

            # Order Blocks Calculation
            ob_data = pd.DataFrame()
            if not swings.empty:
                try:
                    ob_raw = smc.ob(self.historical_data.reset_index(drop=True),
                                    swings.reset_index(drop=True))
                    if isinstance(ob_raw, pd.DataFrame):
                        valid_indices = ob_raw.index[ob_raw.index < len(self.historical_data)]
                        ob_data = ob_raw.loc[valid_indices]
                except Exception as e:
                    print(f"OB calculation error: {e}")

            # Fair Value Gap Calculation
            fvg_data = pd.DataFrame()
            if len(self.historical_data) >= 3:
                try:
                    if all(col in self.historical_data.columns for col in ['high', 'low', 'close']):
                        fvg_raw = smc.fvg(self.historical_data.reset_index(drop=True))
                        if isinstance(fvg_raw, pd.DataFrame):
                            valid_indices = fvg_raw.index[fvg_raw.index < len(self.historical_data)]
                            fvg_data = fvg_raw.loc[valid_indices]
                    else:
                        print("Required columns missing for FVG calculation")
                except Exception as e:
                    print(f"FVG calculation error: {e}")

            # Combine data
            combined = self.historical_data.copy()
            dfs_to_join = {'bos_': bos_data, 'ob_': ob_data, 'fvg_': fvg_data}

            for prefix, df_join in dfs_to_join.items():
                if not df_join.empty:
                    combined = combined.join(df_join.add_prefix(prefix), how='left')

            return combined.ffill().fillna(0)

        except Exception as e:
            print(f"Processing error: {e}")
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
            # return final_size
            return 0.01  # Fixed position size for testing

        except Exception as e:
            print("=== POSITION SIZE CALCULATION ERROR ===")
            print(f"Position size error: {e}")
            import traceback
            print("[DEBUG] Full traceback:")
            traceback.print_exc()
            print("=== POSITION SIZE CALCULATION ERROR END ===\n")
        return 0.01

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
            'bos_BOS': 0,
            'bos_CHOCH': 0,
            'bos_Level': 0,
            'liq_Liquidity': 0
        }
        for col, default in required_columns.items():
            if col not in processed.columns:
                processed[col] = default

        # try:
        #     # Convert index and mitigation columns to datetime with UTC for safe comparison
        #     index_series = processed.index.to_series()
        #     fvg_mit = pd.to_datetime(processed['fvg_MitigatedIndex'], utc=True)
        #     liq_mit = pd.to_datetime(processed['liq_Swept'], utc=True)
        #
        #     # ==== Indipendent Signals with Mitigation ====
        #     # 1. Fair Value Gap (FVG) only if it's not mitigated
        #     fvg_unmitigated = processed['fvg_FVG'].notna() & (
        #             processed['fvg_MitigatedIndex'].isna() | (index_series < fvg_mit)
        #     )
        #     fvg_signal = np.where(
        #         (processed['fvg_FVG'] == 1) & fvg_unmitigated, 1,
        #         np.where(
        #             (processed['fvg_FVG'] == -1) & fvg_unmitigated, -1,
        #             0
        #         )
        #     )
        #
        #     # 2. Order Block (OB)
        #     ob_signal = np.where(
        #         (processed['ob_OB'] == 1) & (processed['close'] > processed['ob_Top']), 1,
        #         np.where(
        #             (processed['ob_OB'] == -1) & (processed['close'] < processed['ob_Bottom']),
        #             -1, 0
        #         )
        #     )
        #
        #     # 3. Liquidity only if it's not swept
        #     liq_unmitigated = processed['liq_Liquidity'].notna() & (
        #             processed['liq_Swept'].isna() | (index_series < liq_mit)
        #     )
        #     liq_signal = np.where(
        #         (processed['liq_Liquidity'] == 1) & liq_unmitigated, 1,
        #         np.where(
        #             (processed['liq_Liquidity'] == -1) & liq_unmitigated, -1,
        #             0
        #         )
        #     )
        #
        #     # 4. BOS & CHOCH (structure)
        #     bos_signal = np.where(
        #         (processed['bos_BOS'] == 1) & (processed['close'] > processed['bos_Level']), 1,
        #         np.where(
        #             (processed['bos_BOS'] == -1) & (processed['close'] < processed['bos_Level']),
        #             -1, 0
        #         )
        #     )
        #     choch_signal = np.where(
        #         (processed['bos_CHOCH'] == 1) & (processed['close'] > processed['bos_Level']), 1,
        #         np.where(
        #             (processed['bos_CHOCH'] == -1) & (processed['close'] < processed['bos_Level']),
        #             -1, 0
        #         )
        #     )
        #
        #     # ==== Structure confirmation and take 2 last swings ====
        #     swings = self.get_swing_data(processed)
        #     if len(swings) >= 2:
        #         # Take swings
        #         recent = swings.iloc[-2:]
        #         # Identify High and Low swing
        #         last_high = recent.loc[recent['HighLow'] == 1, 'Level'].iloc[-1]
        #         last_low = recent.loc[recent['HighLow'] == -1, 'Level'].iloc[-1]
        #         midpoint = (last_high + last_low) / 2
        #     else:
        #         midpoint = np.nan
        #
        #     # ==== PD Array added (Premium/Discount) based ICT ====
        #     pd_ok = np.zeros(len(processed), dtype=bool)
        #     if not np.isnan(midpoint):
        #         pd_ok = np.where(
        #             (swing_signal == 1) & (processed['close'] < midpoint), True,
        #             np.where(
        #                 (swing_signal == -1) & (processed['close'] > midpoint), True,
        #                 False
        #             )
        #         )
        #
        #     # ==== Signals combinations ====
        #     # buy_condition = (
        #     #         # (swing_signal == 1) & pd_ok & (
        #     #         (fvg_signal == 1) | (ob_signal == 1) |
        #     #         (liq_signal == 1) | (bos_signal == 1) | (choch_signal == 1)
        #     #     # )
        #     # )
        #     # sell_condition = (
        #     #         # (swing_signal == -1) & pd_ok & (
        #     #         (fvg_signal == -1) | (ob_signal == -1) |
        #     #         (liq_signal == -1) | (bos_signal == -1) | (choch_signal == -1)
        #     #     # )
        #     # )
        #     buy_condition = (fvg_signal == 1) | (ob_signal == 1) | (liq_signal == 1) | (bos_signal == 1) | (choch_signal == 1)
        #
        #     sell_condition = (fvg_signal == -1) | (ob_signal == -1) | (liq_signal == -1) | (bos_signal == -1) | (choch_signal == -1)
        #
        #     processed.loc[buy_condition, 'signal'] = 1
        #     processed.loc[sell_condition, 'signal'] = -1
        #
        #     # ==== Kod validimi për testim ====
        #     if len(processed) > 1:
        #         last_valid_idx = processed.index[-2]
        #         valid_signals = processed.loc[:last_valid_idx].copy()
        #
        #         if DEBUG_MODE:
        #             print("\n=== DEBUG SIGNALS (ME MITIGATION) ===")
        #             print(valid_signals[
        #                       ['signal', 'fvg_FVG', 'fvg_MitigatedIndex', 'ob_OB', 'liq_Liquidity', 'liq_Swept',
        #                        'bos_BOS', 'bos_CHOCH']].tail(10))
        #         return valid_signals['signal']
        #
        #     return processed['signal']

        try:
            # ==== Independent Signals ====
            # 1. Fair Value Gap (FVG)
            fvg_signal = np.where(
                processed['fvg_FVG'] == 1, 1,
                np.where(
                    processed['fvg_FVG'] == -1, -1,
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

            # 3. Liquidity
            liq_signal = np.where(
                processed['liq_Liquidity'] == 1, 1,
                np.where(
                    processed['liq_Liquidity'] == -1, -1,
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
                recent = swings.iloc[-2:]
                last_high = recent.loc[recent['HighLow'] == 1, 'Level'].iloc[-1]
                last_low = recent.loc[recent['HighLow'] == -1, 'Level'].iloc[-1]
                midpoint = (last_high + last_low) / 2
            else:
                midpoint = np.nan

            # ==== PD Array (Premium/Discount) based ICT ====
            pd_ok = np.zeros(len(processed), dtype=bool)
            if not np.isnan(midpoint):
                pd_ok = np.where(
                    (swing_signal == 1) & (processed['close'] < midpoint), True,
                    np.where(
                        (swing_signal == -1) & (processed['close'] > midpoint), True,
                        False
                    )
                )

            # Get retracement signals
            swings = self.get_swing_data(processed)
            retracements = self.validate_retracements(swings)
            if not retracements.empty:
                ret_signal = np.where(
                    (retracements['Direction'] == 1) & retracements['Valid'], 1,
                    np.where(
                        (retracements['Direction'] == -1) & retracements['Valid'], -1,
                        0
                    )
                )
            else:
                ret_signal = np.zeros(len(processed))

            # ==== Signals combinations ====
            buy_condition = (fvg_signal == 1) | (ob_signal == 1) | (liq_signal == 1) | (bos_signal == 1) | (
                    choch_signal == 1) | (ret_signal == 1)

            sell_condition = (fvg_signal == -1) | (ob_signal == -1) | (liq_signal == -1) | (bos_signal == -1) | (
                    choch_signal == -1) | (ret_signal == -1)

            processed.loc[buy_condition, 'signal'] = 1
            processed.loc[sell_condition, 'signal'] = -1

            # ==== Validation testing ====
            if len(processed) > 1:
                last_valid_idx = processed.index[-2]
                valid_signals = processed.loc[:last_valid_idx].copy()

                if DEBUG_MODE:
                    print("\n=== DEBUG SIGNALS ===")
                    print(
                        valid_signals[['signal', 'fvg_FVG', 'ob_OB', 'liq_Liquidity', 'bos_BOS', 'bos_CHOCH']].tail(10))
                return valid_signals['signal']

            return processed['signal']

        except Exception as e:
            if DEBUG_MODE:
                print("Signal generation error:", e)
            return pd.Series()


class ICTExecution:
    def __init__(self, symbol):
        self.symbol = symbol
        self.max_positions = 3

    def execute_order(self, signal, price, sl, tp, size):
        """Safe order execution with position limits."""
        try:
            open_trades = len(mt5.positions_get(symbol=self.symbol) or 0)
            if open_trades >= self.max_positions:
                return None

            order_type = mt5.ORDER_TYPE_BUY if signal == 1 else mt5.ORDER_TYPE_SELL
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": round(size, 2),
                "type": order_type,
                "price": price,
                "sl": sl,
                "tp": tp,
                "deviation": 20,
                "comment": "ICT Strategy"
            }
            return mt5.order_send(request) if not DRY_RUN else None

        except Exception as e:
            print(f"Order execution error: {e}")
            return None


def main_loop():
    if not initialize_mt5():
        return

    trader = SMCTrader(SYMBOL, TIMEFRAME)
    executor = ICTExecution(SYMBOL)
    print("🚀 Starting Trading Bot...")

    try:
        while True:
            new_data = fetch_realtime_data()
            if new_data.empty:
                time.sleep(1)
                continue

            trader.historical_data = pd.concat([trader.historical_data, new_data]) \
                                         .sort_index() \
                                         .drop_duplicates() \
                                         .iloc[-NUM_CANDLES:]

            signals = trader.generate_signals()
            current_signal = signals.iloc[-1] if not signals.empty else 0

            if current_signal != 0:
                price = mt5.symbol_info_tick(SYMBOL).ask if current_signal == 1 else mt5.symbol_info_tick(SYMBOL).bid
                swings = trader.get_swing_data(trader.historical_data)

                if not swings.empty:
                    try:
                        if current_signal == 1:
                            sl = swings[swings['HighLow'] == -1]['Level'].iloc[-1]
                            tp = price + 2 * (price - sl)
                        else:
                            sl = swings[swings['HighLow'] == 1]['Level'].iloc[-1]
                            tp = price - 2 * (sl - price)

                        size = 0.01  # Fixed size for example
                        executor.execute_order(current_signal, price, sl, tp, size)
                    except IndexError:
                        print("⚠️ Insufficient swings for SL calculation")

            time.sleep(5)

    except KeyboardInterrupt:
        print("\n🛑 Bot stopped by user")
    finally:
        mt5.shutdown()
        print("✅ Disconnected from MT5")


if __name__ == "__main__":
    main_loop()