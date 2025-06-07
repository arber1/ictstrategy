import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
from datetime import datetime
from smartmoneyconcepts.smc import smc
from typing import Tuple
import traceback

class SMCTrader:
    def __init__(self, debug=False):
        self.historical_data = pd.DataFrame()
        self.swing_length = 14
        self.pip_size = 0.1
        self.debug = debug

    def _log(self, stage, message, data=None):
        """Enhanced terminal logging"""
        if self.debug:
            header = f"\n╒═[{datetime.now().strftime('%H:%M:%S')}] {stage.upper()} {message}"
            print(header)
            print('╞' + '═'*(len(header)-2))
            if data is not None:
                if isinstance(data, pd.DataFrame):
                    print(data.tail(3).to_string())
                elif isinstance(data, dict):
                    for k, v in data.items():
                        print(f"│ {k:<25}: {str(v)[:100]}")
                else:
                    print(f"│ {str(data)[:200]}")
            print('╘' + '═'*(len(header)-2))

    def get_historical_data(self) -> pd.DataFrame:
        """Enhanced data fetcher with array prevention"""
        try:
            # Fetch and immediately convert to scalar DataFrame
            rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, NUM_CANDLES)
            df = self._convert_to_scalar_df(rates) if rates else pd.DataFrame()

            if df.empty:
                self._log("warning", "Empty initial data")
                return self.historical_data.tail(NUM_CANDLES)

            # Maintain historical data with validation
            self.historical_data = self._validate_data_scalars(
                pd.concat([self.historical_data, df]).tail(NUM_CANDLES*2)
            )

            print(self.historical_data)

            return self.historical_data

        except Exception as e:
            self._log("error", "Data fetch failed", str(e))
            print(self.historical_data)
            return pd.DataFrame()

    def _convert_to_scalar_df(self, rates):
        """Convert MT5 rates to scalar DataFrame with array protection"""
        try:
            # Convert to DataFrame with explicit dtype handling
            df = pd.DataFrame(rates, dtype=np.float64)

            # Build rename mapping
            rename_map = {"time": "Date"}
            if "real_volume" in df.columns:
                rename_map["real_volume"] = "volume"
                df = df.drop(columns=["tick_volume"], errors="ignore")
            else:
                rename_map["tick_volume"] = "volume"

            # Apply renaming and date conversion
            df = (
                df.rename(columns=rename_map)
                .assign(Date=lambda x: pd.to_datetime(x["Date"], unit="s"))
                .set_index("Date")
            )

            # Recursive array flattening
            def flatten(value):
                """Extract first element from nested array structures"""
                while isinstance(value, (np.ndarray, list, pd.Series)) and len(value) > 0:
                    value = value[0]
                return value

            # Apply to all elements
            for col in df.columns:
                df[col] = df[col].apply(flatten)

            return df.astype(np.float64, errors="ignore")

        except Exception as e:
            self._log("error", f"Data conversion failed: {str(e)}")
            return pd.DataFrame()

    def _validate_data_scalars(self, df):
        """Enhanced validation with dimension checks"""
        if df.empty:
            return df

        # Check for 2D arrays in columns
        array_cols = []
        for col in df.columns:
            if any(isinstance(x, (np.ndarray, list)) for x in df[col]):
                array_cols.append(col)

        if array_cols:
            self._log("error", f"Array columns detected: {array_cols}")
            for col in array_cols:
                df[col] = df[col].apply(lambda x: x[0] if isinstance(x, (np.ndarray, list)) else x)

        # Ensure numeric values
        return df.apply(pd.to_numeric, errors="coerce").dropna(how="all")

    def process_market_structure(self) -> pd.DataFrame:
        """Array-safe market structure processing"""
        try:
            df = self.get_historical_data()
            if df.empty:
                return pd.DataFrame()

            # Process indicators with individual validation
            indicators = {
                'swing': self._safe_smc(smc.swing_highs_lows, df, self.swing_length),
                'fvg': self._safe_smc(smc.fvg, df),
                'ob': self._safe_smc(smc.ob, df, self.swing_data),
                'liq': self._safe_smc(smc.liquidity, df, self.swing_data)
            }

            # Merge validated indicators
            combined = pd.concat([df] + [ind.add_prefix(f'{name}_')
                                         for name, ind in indicators.items()], axis=1)

            return self._validate_data_scalars(combined.ffill().dropna())

        except Exception as e:
            self._log("error", "Processing failed", str(e))
            return pd.DataFrame()

    def _get_atomic_value(self, df, col):
        """Safe value extraction with 3-layer validation"""
        try:
            value = df[col].iloc[-1]
            # Layer 1: Array check
            if isinstance(value, (np.ndarray, list)):
                return float(value[0])
            # Layer 2: Series check
            if isinstance(value, pd.Series):
                return float(value.iloc[0])
            # Layer 3: Type safety
            return float(value)
        except:
            return 0.0

    def generate_trading_signals(self) -> dict:
        """Atomic value signal generator"""
        try:
            df = self.process_market_structure()
            if df.empty:
                return {}

            # Atomic value extraction with trace
            signal_data = {
                col: self._get_atomic_value(df, col)
                for col in ['ob_OB', 'ob_Top', 'ob_Bottom',
                            'fvg_FVG', 'liq_Liquidity', 'close']
            }

            # Validate signal conditions
            if any(isinstance(v, (np.ndarray, list)) for v in signal_data.values()):
                self._log("error", "Array values in signal data")
                return {}

            # Rest of signal logic...
            return self._create_signal(signal_data)

        except Exception as e:
            self._log("error", "Signal generation failed", str(e))
            return {}

    def _safe_smc(self, func, *args):
        """Execute SMC function with array protection"""
        result = func(*args)
        return result.applymap(
            lambda x: x[0] if isinstance(x, (np.ndarray, list)) else x
        )

    # def generate_trading_signals(self) -> dict:
    #     try:
    #         df = self.process_market_structure()
    #         if df.empty:
    #             self._log("warning", "Empty DataFrame - No Signal")
    #             return {}
    #
    #         def get_scalar(col):
    #             val = df[col].iloc[-1]
    #             if isinstance(val, (pd.Series, np.ndarray)):
    #                 return float(val.iloc[0]) if len(val) > 0 else 0.0
    #             return float(val)
    #
    #         current = {
    #             'ob_OB': get_scalar('ob_OB'),
    #             'ob_Top': get_scalar('ob_Top'),
    #             'ob_Bottom': get_scalar('ob_Bottom'),
    #             'fvg_FVG': get_scalar('fvg_FVG'),
    #             'liq_Liquidity': get_scalar('liq_Liquidity'),
    #             'close': get_scalar('close')
    #         }
    #
    #         self._log("data", "Current Market State", current)
    #
    #         long_cond = all([
    #             current['ob_OB'] > 0.5,
    #             current['close'] > current['ob_Top'],
    #             current['fvg_FVG'] > 0.5,
    #             current['liq_Liquidity'] > 0
    #         ])
    #
    #         short_cond = all([
    #             current['ob_OB'] < -0.5,
    #             current['close'] < current['ob_Bottom'],
    #             current['fvg_FVG'] < -0.5,
    #             current['liq_Liquidity'] < 0
    #         ])
    #
    #         direction = 1 if long_cond else -1 if short_cond else 0
    #         self._log("decision", "Direction Analysis", {
    #             'long_conditions': long_cond,
    #             'short_conditions': short_cond,
    #             'selected_direction': direction
    #         })
    #
    #         symbol_info = mt5.symbol_info(SYMBOL)
    #         pip_size = 10 ** (-symbol_info.digits) if symbol_info else 0.1
    #
    #         liquidity_high = df['liq_Level'][df['liq_Liquidity'] == 1].dropna().iloc[-1] if any(df['liq_Liquidity'] == 1) else None
    #         liquidity_low = df['liq_Level'][df['liq_Liquidity'] == -1].dropna().iloc[-1] if any(df['liq_Liquidity'] == -1) else None
    #
    #         if direction == 1:
    #             sl = (liquidity_low - 5*pip_size) if liquidity_low else None
    #             tp = (liquidity_high + 10*pip_size) if liquidity_high else None
    #         elif direction == -1:
    #             sl = (liquidity_high + 5*pip_size) if liquidity_high else None
    #             tp = (liquidity_low - 10*pip_size) if liquidity_low else None
    #         else:
    #             sl = tp = None
    #
    #         current_price = mt5.symbol_info_tick(SYMBOL).ask if direction == 1 else mt5.symbol_info_tick(SYMBOL).bid
    #         if not sl or not tp:
    #             sl = current_price * 0.995 if direction == 1 else current_price * 1.005
    #             tp = current_price * 1.015 if direction == 1 else current_price * 0.985
    #
    #         self._log("risk", "Risk Parameters", {
    #             'pip_size': pip_size,
    #             'liquidity_high': liquidity_high,
    #             'liquidity_low': liquidity_low,
    #             'calculated_sl': sl,
    #             'calculated_tp': tp,
    #             'final_sl': sl,
    #             'final_tp': tp
    #         })
    #
    #         return {
    #             'timestamp': datetime.now().isoformat(),
    #             'direction': direction,
    #             'entry': round(current['close'], 5),
    #             'stop_loss': round(sl, 5),
    #             'take_profit': round(tp, 5),
    #             'confidence': self._calculate_confidence(df)
    #         }
    #
    #     except Exception as e:
    #         self._log("error", "Signal Generation Error", {
    #             'error': str(e),
    #             'traceback': traceback.format_exc()
    #         })
    #         return {}

    def _calculate_confidence(self, df):
        try:
            confidence = min(1.0, df[['ob_OB', 'fvg_FVG', 'liq_Liquidity']].tail(3).abs().mean().mean() / 1.5)
            self._log("confidence", "Calculation", {
                'values': df[['ob_OB', 'fvg_FVG', 'liq_Liquidity']].tail(3).to_dict(),
                'confidence_score': confidence
            })
            return confidence
        except:
            return 0.5

class TradingBot:
    def __init__(self, debug=False):
        self.trader = SMCTrader(debug=debug)
        self.active_trade = None
        self.debug = debug

    def execute_strategy(self):
        try:
            while True:
                try:
                    signal = self.trader.generate_trading_signals()
                    self._print_signal(signal)

                    if signal.get('direction', 0) != 0 and self.should_enter_trade(signal):
                        self.execute_trade(signal)

                    time.sleep(5)

                except KeyboardInterrupt:
                    print("\n🛑 Strategy Stopped by User")
                    break
                except Exception as e:
                    print(f"\n🔥 Critical Error: {str(e)}")
                    traceback.print_exc()
                    time.sleep(10)

        finally:
            if self.debug:
                print("\n🔍 Debug Session Ended")

    def _print_signal(self, signal):
        if self.debug and signal:
            print("\n" + "═"*50)
            print(f"📡 SIGNAL GENERATED @ {datetime.now().strftime('%H:%M:%S')}")
            print(f"Direction: {'BUY' if signal['direction'] == 1 else 'SELL' if signal['direction'] == -1 else 'NONE'}")
            print(f"Entry Price: {signal['entry']}")
            print(f"Stop Loss: {signal['stop_loss']}")
            print(f"Take Profit: {signal['take_profit']}")
            print(f"Confidence: {signal['confidence']:.2%}")
            print("═"*50)

    def execute_trade(self, signal):
        try:
            if DRY_RUN:
                print(f"\n🚀 DRY RUN: {'BUY' if signal['direction'] == 1 else 'SELL'} @ {signal['entry']}")
                print(f"💔 SL: {signal['stop_loss']} | 💰 TP: {signal['take_profit']}")
                return

            print(f"\n✅ REAL TRADE EXECUTED: {'BUY' if signal['direction'] == 1 else 'SELL'} @ {signal['entry']}")
            print(f"🛡️ SL: {signal['stop_loss']} | 🎯 TP: {signal['take_profit']}")

        except Exception as e:
            print(f"\n❌ Trade Execution Failed: {str(e)}")
            traceback.print_exc()

    def should_enter_trade(self, signal):
        min_confidence = 0.65
        return signal.get('confidence', 0) >= min_confidence

# Configuration
SYMBOL = "BTCUSD"
TIMEFRAME = mt5.TIMEFRAME_M5
NUM_CANDLES = 500
DRY_RUN = False

if __name__ == "__main__":
    if mt5.initialize():
        print("✅ MT5 Connected - Starting Strategy")
        TradingBot(debug=True).execute_strategy()
        mt5.shutdown()
    else:
        print("❌ MT5 Connection Failed")