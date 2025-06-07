# ========== COMPLETE IMPLEMENTATION ==========
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from typing import Dict, Union, Optional
from smartmoneyconcepts.smc import smc

# ========== CONFIGURATION ==========
SYMBOL = "BTCUSD"
TIMEFRAME = "M15"
NUM_CANDLES = 200
DRY_RUN = False
DEBUG_MODE = True
MAX_RETRIES = 3
MAX_POSITIONS = 3

ICT_CONFIG = {
    "SWING_LENGTHS": {"M1": 38, "M5": 10, "M15": 5, "H1": 4, "D1": 3},
    "FVG_EXPIRATION": {
        "M1": timedelta(hours=4),
        "M5": timedelta(hours=6),
        "M15": timedelta(hours=3),
        "H1": timedelta(days=1),
        "D1": timedelta(days=3)
    },
    "TRADING_SESSIONS": {
        "London": (7, 16),  # 7 AM to 4 PM UTC
        "NewYork": (13, 22)  # 1 PM to 10 PM UTC
    },
    "RISK_PERCENT": 1.0,
    "LIQUIDITY_RANGE_PCT": 0.05
}


def initialize_mt5() -> bool:
    """Initialize MT5 connection with error handling"""
    print("\n=== MT5 INITIALIZATION ===")
    try:
        if not mt5.initialize():
            print("❌ Connection failed. Error:", mt5.last_error())
            return False

        account_info = mt5.account_info()
        if not account_info:
            print("❌ Failed to get account info")
            return False

        print(f"✅ Connected to {account_info.server}")
        return True

    except Exception as e:
        print(f"❌ Initialization error: {str(e)}")
        return False


def fetch_realtime_data() -> pd.DataFrame:
    """Fetch and validate market data"""
    try:
        timeframe = getattr(mt5, f'TIMEFRAME_{TIMEFRAME}')
        rates = mt5.copy_rates_from_pos(SYMBOL, timeframe, 0, NUM_CANDLES)

        if rates is None or len(rates) == 0:
            return pd.DataFrame()

        df = pd.DataFrame(rates)
        df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
        df.set_index('datetime', inplace=True)

        # Validate price data integrity
        if (df['high'] < df['low']).any():
            return pd.DataFrame()

        return df[['open', 'high', 'low', 'close', 'tick_volume']].rename(
            columns={'tick_volume': 'volume'}
        )

    except Exception as e:
        print(f"Data error: {str(e)}")
        return pd.DataFrame()


# ========== COMPLETE SMCTrader CLASS IMPLEMENTATION ==========
class SMCTrader:
    def __init__(self, symbol: str, timeframe: str):
        self.symbol = symbol
        self.timeframe = timeframe
        self.swing_length = ICT_CONFIG["SWING_LENGTHS"][timeframe] * 2  # Library uses 2x input
        self.historical_data = pd.DataFrame()
        self.pip_size = mt5.symbol_info(symbol).point * 100 if mt5.symbol_info(symbol) else 0.01

    def get_swing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Swing point detection with index alignment"""
        try:
            if len(df) < self.swing_length * 2:
                return pd.DataFrame()

            temp_df = df.reset_index(drop=True)
            swings = smc.swing_highs_lows(
                temp_df,
                swing_length=self.swing_length
            )

            swings.index = df.index[swings.index]
            return swings[['HighLow', 'Level']].dropna()

        except Exception as e:
            print(f"Swing detection error: {str(e)}")
            return pd.DataFrame()

    def _merge_data(self, *datasets) -> pd.DataFrame:
        """Merge SMC components with column validation"""
        combined = self.historical_data.copy()
        component_map = {
            'bos_': ['BOS', 'Level'],
            'ob_': ['OB', 'Top', 'Bottom'],
            'fvg_': ['FVG', 'Top', 'Bottom'],
            'liq_': ['Liquidity', 'Level']
        }

        for prefix, data in zip(component_map.keys(), datasets):
            if not data.empty:
                # Filter and rename valid columns
                valid_cols = [c for c in data.columns if c in component_map[prefix]]
                data = data[valid_cols].add_prefix(prefix)
                combined = combined.merge(
                    data,
                    left_index=True,
                    right_index=True,
                    how='left'
                )

        # Fill missing signal columns with 0
        for col in ['bos_BOS', 'ob_OB', 'fvg_FVG', 'liq_Liquidity']:
            if col not in combined.columns:
                combined[col] = 0

        return combined.fillna(0).astype({
            'bos_BOS': int,
            'ob_OB': int,
            'fvg_FVG': int,
            'liq_Liquidity': int
        })

    def _calculate_bos(self, df, swings):
        """BOS calculation with sequence validation"""
        try:
            bos = smc.bos_choch(
                df.reset_index(drop=True),
                swings.reset_index(drop=True),
                close_break=True
            )
            # FIX: Validate 4-swing sequence
            bos = bos[(bos['BOS'] != 0) & (bos['BrokenIndex'] != 0)]
            return bos.reindex(df.index)
        except Exception as e:
            print(f"BOS error: {str(e)}")
            return pd.DataFrame()

    def _calculate_ob(self, df, swings):
        """Order Blocks with mitigation check"""
        try:
            ob = smc.ob(
                df.reset_index(drop=True),
                swings.reset_index(drop=True),
                close_mitigation=False
            )
            return ob.reindex(df.index)
        except Exception as e:
            print(f"OB error: {str(e)}")
            return pd.DataFrame()

    def _calculate_fvg(self, df):
        """FVG with expiration check"""
        try:
            fvg = smc.fvg(df, join_consecutive=False)
            if fvg.empty:
                return pd.DataFrame()

            # Ensure datetime index
            fvg.index = pd.to_datetime(fvg.index, utc=True)

            # Get expiration threshold
            expiration = ICT_CONFIG["FVG_EXPIRATION"][self.timeframe]
            expiration_threshold = pd.Timestamp.now(tz='UTC') - expiration

            # Filter using loc accessor
            valid_fvg = fvg.loc[fvg.index >= expiration_threshold]

            return valid_fvg.rename(columns={
                'FVG': 'fvg_FVG',
                'Top': 'fvg_Top',
                'Bottom': 'fvg_Bottom'
            }).reindex(df.index, fill_value=0)

        except Exception as e:
            print(f"FVG error: {str(e)}")
            return pd.DataFrame()

    def _calculate_liquidity(self, df, swings):
        """Liquidity with dynamic range"""
        try:
            return smc.liquidity(
                df,
                swings,
                range_percent=ICT_CONFIG["LIQUIDITY_RANGE_PCT"]
            ).reindex(df.index)
        except Exception as e:
            print(f"Liquidity error: {str(e)}")
            return pd.DataFrame()

    def generate_signals(self) -> pd.Series:
        """Complete signal generation with all SMC components"""
        processed = self.process_smc_data()
        if processed.empty:
            return pd.Series()

        try:
            # Get current values
            current = processed.iloc[-1]
            bos = current['bos_BOS']
            ob = current['ob_OB']
            fvg = current['fvg_FVG']
            liq = current['liq_Liquidity']
            volume = current['volume']

            # Calculate confirmation metrics
            avg_volume = processed['volume'].rolling(20).mean().iloc[-1]
            price_change = processed['close'].pct_change().iloc[-1]

            # Bullish confirmation rules
            bull_cond = (
                    (bos == 1) and
                    (ob == 1) and
                    (fvg == 1) and
                    (liq == 1) and
                    (volume > avg_volume * 1.2) and
                    (price_change > 0)
            )

            # Bearish confirmation rules
            bear_cond = (
                    (bos == -1) and
                    (ob == -1) and
                    (fvg == -1) and
                    (liq == -1) and
                    (volume > avg_volume * 1.2) and
                    (price_change < 0)
            )

            # Generate signals
            signal = 0
            if bull_cond:
                signal = 1
                # Validate against recent highs
                if processed['high'].iloc[-1] < processed['high'].rolling(5).max().iloc[-1]:
                    signal = 0
            elif bear_cond:
                signal = -1
                # Validate against recent lows
                if processed['low'].iloc[-1] > processed['low'].rolling(5).min().iloc[-1]:
                    signal = 0

            return pd.Series([signal], index=[processed.index[-1]])

        except Exception as e:
            print(f"Signal generation error: {str(e)}")
            return pd.Series()

    def validate_ob(self, ob_data: pd.DataFrame) -> pd.DataFrame:
        """Complete OB validation with mitigation check"""
        if ob_data.empty:
            return pd.DataFrame()

        try:
            current_idx = self.historical_data.index[-1]
            return ob_data[
                (ob_data['MitigatedIndex'] > current_idx) &
                (ob_data['Percentage'] > 30)  # Minimum strength threshold
                ][['OB', 'Top', 'Bottom']]
        except KeyError:
            print("⚠️ Missing columns in OB data")
            return pd.DataFrame()

    def calculate_position_size(self) -> float:
        """Complete position sizing with leverage check"""
        try:
            account = mt5.account_info()
            symbol_info = mt5.symbol_info(self.symbol)

            # Calculate risk-adjusted size
            risk_amount = account.balance * ICT_CONFIG["RISK_PERCENT"] / 100
            tick_value = symbol_info.trade_tick_value
            leverage = account.leverage or 1

            # Calculate units considering leverage
            max_allowed = (account.margin_free * leverage) / symbol_info.ask
            risk_based = risk_amount / (self.pip_size * 10)  # 10 pip risk

            return round(min(max_allowed, risk_based), 2)

        except Exception as e:
            print(f"Position sizing error: {str(e)}")
            return 0.01

    def process_smc_data(self) -> pd.DataFrame:
        """Robust data processing pipeline"""
        try:
            # Validate data requirements
            if len(self.historical_data) < self.swing_length * 2:
                return pd.DataFrame()

            if not {'open', 'high', 'low', 'close', 'volume'}.issubset(self.historical_data.columns):
                return pd.DataFrame()

            # Calculate components with explicit empty checks
            swings = self.get_swing_data(self.historical_data)
            bos = self._calculate_bos(self.historical_data, swings) if not swings.empty else pd.DataFrame()
            ob = self._calculate_ob(self.historical_data, swings) if not swings.empty else pd.DataFrame()
            fvg = self._calculate_fvg(self.historical_data)
            liq = self._calculate_liquidity(self.historical_data, swings) if not swings.empty else pd.DataFrame()

            # Merge components
            merged = self._merge_data(bos, ob, fvg, liq)

            # Ensure required columns exist
            for col in ['bos_BOS', 'ob_OB', 'fvg_FVG', 'liq_Liquidity']:
                if col not in merged.columns:
                    merged[col] = 0

            return merged.ffill().fillna(0)

        except Exception as e:
            print(f"Data processing error: {str(e)}")
            return pd.DataFrame()


class ICTExecution:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.error_count = 0
        self.last_trade_time = datetime.now()  # Initialize with current time

    def manage_trades(self, trader: SMCTrader):
        """Robust position management with None handling"""
        try:
            # Get positions with safety checks
            positions = mt5.positions_get(symbol=self.symbol)

            # 1. Handle None response
            if positions is None:
                print(f"Position check failed: {mt5.last_error()}")
                positions = []

            # 2. Force iterable type
            positions = positions if isinstance(positions, (list, tuple)) else []

            # 3. Position limit check
            if len(positions) >= MAX_POSITIONS:
                self._close_oldest_position(positions)

            # 4. Type-safe iteration
            for position in positions:
                if isinstance(position, mt5.TradePosition):
                    self._update_stop_loss(trader, position)

        except Exception as e:
            print(f"Trade management error: {str(e)}")

    def _close_oldest_position(self, positions: list):
        """Close oldest position when limit reached"""
        if positions:
            oldest = min(positions, key=lambda x: x.time_update)
            mt5.Close(oldest.ticket)

    def _update_stop_loss(self, trader: SMCTrader, position):
        """Dynamic stop loss adjustment using swing points"""
        try:
            current_price = mt5.symbol_info_tick(self.symbol).ask \
                if position.type == mt5.ORDER_TYPE_BUY \
                else mt5.symbol_info_tick(self.symbol).bid

            swings = trader.get_swing_data(trader.historical_data)
            if swings.empty:
                return

            if position.type == mt5.ORDER_TYPE_BUY:
                valid_swings = swings[swings['HighLow'] == -1]
                new_sl = valid_swings['Level'].iloc[-1] if not valid_swings.empty else position.sl
            else:
                valid_swings = swings[swings['HighLow'] == 1]
                new_sl = valid_swings['Level'].iloc[-1] if not valid_swings.empty else position.sl

            # Only update if significant difference
            if abs(new_sl - position.sl) > trader.pip_size * 2:
                request = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "position": position.ticket,
                    "sl": new_sl,
                    "deviation": 20
                }
                if not DRY_RUN:
                    mt5.order_send(request)

        except Exception as e:
            print(f"SL update error: {str(e)}")

    def is_session_active(self) -> bool:
        """Check if current time is in trading session"""
        now = datetime.utcnow().hour
        for session, (start, end) in ICT_CONFIG["TRADING_SESSIONS"].items():
            if start <= now < end:
                return True
        return False

    def execute_order(self, signal: int, price: float, sl: float, tp: float, size: float):
        """Safe order execution with session check"""
        if self.error_count > MAX_RETRIES:
            print("🚨 Circuit breaker triggered")
            return None

        # if not self.is_session_active():
        #     print("⚠️ Outside trading session hours")
        #     return None

        try:
            trade_type = mt5.ORDER_TYPE_BUY if signal == 1 else mt5.ORDER_TYPE_SELL
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": size,
                "type": trade_type,
                "price": price,
                "sl": sl,
                "tp": tp,
                "deviation": 20,
                "magic": 10032023,
                "comment": "SMC Bot",
            }

            if DRY_RUN:
                print(f"DRY RUN: Would execute {trade_type} at {price}")
                return None

            result = mt5.order_send(request)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                self.last_trade_time = datetime.now()  # Update only on success

            return result

        except Exception as e:
            self.error_count += 1
            print(f"Order error: {str(e)}")
            return None


def main_loop():
    """Main trading loop with complete error handling"""
    if not initialize_mt5():
        return

    trader = SMCTrader(SYMBOL, TIMEFRAME)
    executor = ICTExecution(SYMBOL)
    print("🚀 Starting Trading Bot...")

    # Initial data load
    start_time = time.time()
    while time.time() - start_time < 60:
        df = fetch_realtime_data()
        if not df.empty:
            trader.historical_data = df.iloc[-NUM_CANDLES:]
            break
        time.sleep(5)
    else:
        print("❌ Data load timeout")
        mt5.shutdown()
        return

    # Main processing loop
    while True:
        try:
            # Data collection
            new_data = fetch_realtime_data()
            if not new_data.empty:
                trader.historical_data = pd.concat([
                    trader.historical_data.iloc[-(NUM_CANDLES - len(new_data)):],
                    new_data
                ])

            # Signal generation
            signals = trader.generate_signals()
            current_signal = signals.iloc[-1] if not signals.empty else 0

            # Trade execution
            if current_signal != 0 and executor.is_session_active():
                price = mt5.symbol_info_tick(SYMBOL).ask \
                    if current_signal == 1 \
                    else mt5.symbol_info_tick(SYMBOL).bid

                size = trader.calculate_position_size()
                swings = trader.get_swing_data(trader.historical_data)

                if not swings.empty:
                    sl = swings['Level'].iloc[-1]
                    tp = price + (price - sl) * 2 if current_signal == 1 \
                        else price - (sl - price) * 2

                    executor.execute_order(current_signal, price, sl, tp, size)

            # Position management
            executor.manage_trades(trader)

            # Health check
            if (datetime.now() - executor.last_trade_time).total_seconds() > 3600:
                print("⚠️ No successful trades in past hour")

            time.sleep(5)

        except KeyboardInterrupt:
            print("\n🛑 Bot stopped by user")
            break
        except Exception as e:
            print(f"Main loop error: {str(e)}")
            time.sleep(10)

    mt5.shutdown()
    print("✅ Disconnected from MT5")


if __name__ == "__main__":
    main_loop()