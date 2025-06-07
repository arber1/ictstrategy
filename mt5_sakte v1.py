import MetaTrader5 as mt5
import pandas as pd
import numpy as np  # Add this at the top with other imports
import time
from smartmoneyconcepts.smc import smc
from datetime import datetime

def initialize_mt5():
    if not mt5.initialize():
        print("Failed to initialize MT5")
        mt5.shutdown()
        return False
    print("✅ Successfully connected to MT5")
    return True

# Configuration
SYMBOL = "BTCUSD"
TIMEFRAME = mt5.TIMEFRAME_M15
NUM_CANDLES = 200
DRY_RUN = False
DEBUG_MODE = True  # Toggle debug prints

class SMCTrader:
    def __init__(self):
        self.last_processed = None
        self.historical_data = pd.DataFrame()  # Store historical context

    def process_smc_data(self, ohlc):
        try:
            # Ensure unique timestamps
            ohlc = ohlc[~ohlc.index.duplicated(keep='last')]

            # Generate swing highs/lows first
            swing_data = smc.swing_highs_lows(ohlc, swing_length=10)

            # Ensure valid swing data before using OB & Liquidity
            if not swing_data.empty:
                ob_data = smc.ob(ohlc, swing_data)
                liquidity_data = smc.liquidity(ohlc, swing_data)
            else:
                ob_data = pd.DataFrame()
                liquidity_data = pd.DataFrame()

            # Ensure we have all required liquidity columns
            for col in ['Liquidity', 'Level', 'End', 'Swept']:
                if f'liq_{col}' not in liquidity_data.columns:
                    liquidity_data[f'liq_{col}'] = np.nan


            # Combine all indicators
            df_combined = pd.concat([
                ohlc,
                smc.fvg(ohlc).add_prefix('fvg_'),
                ob_data.add_prefix('ob_'),
                liquidity_data.add_prefix('liq_')
            ], axis=1)

            # Fill missing values
            df_combined.fillna(method='ffill', inplace=True)
            df_combined.fillna(0, inplace=True)

            # Debugging: Print missing columns
            expected_columns = ['ob_OB', 'ob_Top', 'ob_Bottom', 'fvg_FVG', 'liq_Liquidity']
            missing_columns = [col for col in expected_columns if col not in df_combined.columns]
            if missing_columns:
                print(f"🚨 Missing Columns: {missing_columns}")

            # ✅ Debugging: Print Last 5 Rows of Each Indicator Separately
            if DEBUG_MODE:
                print("\n=== DEBUG: ORDER BLOCKS ===")
                print(df_combined[['ob_OB', 'ob_Top', 'ob_Bottom']].tail(5))

                print("\n=== DEBUG: FAIR VALUE GAP ===")
                print(df_combined[['fvg_FVG']].tail(5))

                # print("\n=== DEBUG: BREAK OF STRUCTURE (BOS) ===")
                # print(df_combined[['bos']].tail(5))
                #
                # print("\n=== DEBUG: CHANGE OF CHARACTER (CHOCH) ===")
                # print(df_combined[['CHOCH']].tail(5))

                print("\n=== DEBUG: LIQUIDITY ===")
                print(df_combined[['liq_Liquidity', 'liq_Level']].tail(5))


            return df_combined
        except Exception as e:
            print(f"Processing Error: {str(e)}")
            return ohlc

    def generate_signals(self, df):
        try:
            df['signal'] = 0

            # Convert all SMC columns to floats explicitly
            smc_cols = ['ob_OB', 'ob_Top', 'ob_Bottom', 'fvg_FVG', 'liq_Liquidity', 'liq_Level']
            df[smc_cols] = df[smc_cols].astype(float)
            df['bos'] = smc.bos_choch(df, smc.swing_highs_lows(df, swing_length=50))['BOS']

            # Bullish conditions
            bull_mask = (
                    df['ob_OB'].eq(1.0) &
                    df['close'].gt(df['ob_Top']) &
                    df['fvg_FVG'].eq(1.0) &
                    df['liq_Liquidity'].eq(1.0) &
                    df['close'].gt(df['liq_Level']) &  # Changed from liq_Liquidity to liq_Level
                    (df['bos'].eq(1.0) | df['bos'].isna())
            )

            # Bearish conditions (fixed comparison)
            bear_mask = (
                    df['ob_OB'].eq(-1.0) &
                    df['close'].lt(df['ob_Bottom']) &
                    df['fvg_FVG'].eq(-1.0) &
                    df['liq_Liquidity'].eq(-1.0) &
                    df['close'].lt(df['liq_Level']) &  # Changed from liq_Liquidity to liq_Level
                    (df['bos'].eq(-1.0) | df['bos'].isna())
            )

            df.loc[bull_mask, 'signal'] = 1
            df.loc[bear_mask, 'signal'] = -1

            if DEBUG_MODE:
                print("\n=== DEBUG CONDITIONS ===")
                print("Liquidity Levels:", df[['liq_Liquidity', 'liq_Level']].tail(5))
                print("Bear Mask Values:", bear_mask.tail(5).values)
                print("\n=== DEBUG: BEARISH COMPONENTS ===")
                last_row = df.iloc[-1]
                print(f"OB: {last_row['ob_OB']} | Close: {last_row['close']} | OB Bottom: {last_row['ob_Bottom']}")
                print(f"FVG: {last_row['fvg_FVG']} | Liq Direction: {last_row['liq_Liquidity']} | Liq Level: {last_row['liq_Level']}")
                print(f"BOS: {last_row['bos']} | Signal: {last_row['signal']}")


            return df
        except Exception as e:
            print(f"Signal Generation Error: {str(e)}")
            return df

def fetch_realtime_data():
    """ Fetch live M5 data from MT5 and process OHLCV for Smart Money Concepts (SMC). """
    rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, NUM_CANDLES)

    if rates is None:
        print("❌ No data received from MT5")
        return None

    # Convert to DataFrame
    df = pd.DataFrame(rates)

    # 🔹 Debug: Print Raw Data
    print("\n=== 🔍 RAW MT5 DATA ===")
    print(df.tail(5))  # Print last 5 rows to check values

    # Rename columns to match SMC requirements
    df.rename(columns={"time": "Date", "open": "open", "high": "high", "low": "low",
                       "close": "close", "tick_volume": "tick_volume", "real_volume": "real_volume"}, inplace=True)

    # Ensure volume is correctly assigned
    if "real_volume" in df.columns and df["real_volume"].sum() > 0:
        df["volume"] = df["real_volume"]
    elif "tick_volume" in df.columns:
        df["volume"] = df["tick_volume"]
    else:
        df["volume"] = 0  # Default if no volume exists

    # 🔹 Debug: Print Volume Data
    print("\n=== 🔍 VOLUME DATA ===")
    print(df[["Date", "volume", "tick_volume", "real_volume"]].tail(5))

    # Convert Date to Timestamp
    df["Date"] = pd.to_datetime(df["Date"], unit="s", utc=True)
    df.set_index("Date", inplace=True)

    return df.tail(NUM_CANDLES)  # Keep last N candles

def get_open_trade_count():
    """ Check how many trades are currently open for the given symbol. """
    open_orders = mt5.positions_get(symbol=SYMBOL)
    if open_orders is None:
        return 0  # No open trades
    return len(open_orders)  # Return number of open position

def execute_trade(signal, historical_data):
    """ Execute a trade based on the signal with corrected SL/TP logic """

    # Take pair info
    symbol_info = mt5.symbol_info(SYMBOL)
    if not symbol_info:
        print(f"❌ {SYMBOL} not found")
        return

    # CHECK CURRENT OPEN TRADES
    open_trade_count = get_open_trade_count()
    if open_trade_count >= 5:
        print(f"⚠️ Trade limit reached! ({open_trade_count}/5 open trades)")
        return

    # Trade symbol if it visible
    if not symbol_info.visible:
        mt5.symbol_select(SYMBOL, True)
        print(f"✅ {SYMBOL} enabled")

    # Configuration
    lot_size = 0.01
    price = mt5.symbol_info_tick(SYMBOL).ask if signal == 1 else mt5.symbol_info_tick(SYMBOL).bid
    trade_type = mt5.ORDER_TYPE_BUY if signal == 1 else mt5.ORDER_TYPE_SELL
    pip_size = symbol_info.point * 10  # 1 pip = 0.1 for BTCUSD

    try:
        # Take last 50 candle for swing high/low
        swing_data = smc.swing_highs_lows(historical_data, swing_length=50)
        last_swing_high = swing_data[swing_data['HighLow'] == 1]['Level'].iloc[-1]
        last_swing_low = swing_data[swing_data['HighLow'] == -2]['Level'].iloc[-2]
    except IndexError:
        print("❌ Swing not found")
        return

    # Calculate TP and SL
    if signal == 1:  # BULLISH
        tp = last_swing_high - (10 * pip_size)  # 10 pips below swing high
        sl = last_swing_low + (5 * pip_size)    # 5 pips above swing low
    else:           # BEARISH
        tp = last_swing_low + (10 * pip_size)   # 10 pips above swing low
        sl = last_swing_high - (5 * pip_size)   # 5 pips below swing high

    # Validate position
    if (signal == 1 and price <= sl) or (signal == -1 and price >= sl):
        print(f"⚠️ Invalid Stop Loss | Price: {price} | SL: {sl}")
        return

    # Create request
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": lot_size,
        "type": trade_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 20,
        "magic": 10032023,
        "comment": "SMC Strategy",
        "type_time": mt5.ORDER_TIME_GTC,
    }

    # Debug info
    if DEBUG_MODE:
        print(f"\n{'BULLISH' if signal == 1 else 'BEARISH'} TRADE SETUP")
        print(f"Entry: {price} | SL: {sl} | TP: {tp}")
        print(f"Swing High: {last_swing_high} | Swing Low: {last_swing_low}")

    # Send order
    result = mt5.order_send(request)

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"❌ Trade failed: {result.comment}")
    else:
        print(f"✅ Trade executed: Ticket={result.order}")

def valid_tick(tick):
    return (
            tick is not None and
            tick.time > 0 and
            tick.bid > 0 and
            tick.ask > 0 and
            tick.time_msc > 0
    )

def main_loop():
    trader = SMCTrader()
    if DEBUG_MODE:
        print("🚀 Starting SMC Trading Bot with DEBUG MODE...")
    else:
        print("🚀 Starting SMC Trading Bot...")

    while True:
        try:
            tick = mt5.symbol_info_tick(SYMBOL)
            if not valid_tick(tick):
                print("Invalid tick - retrying...")
                time.sleep(1)
                continue

            # Get fresh data
            new_data = fetch_realtime_data()
            print(new_data)
            if new_data is None:
                time.sleep(1)
                continue

            # Store historical context
            trader.historical_data = pd.concat([
                trader.historical_data,
                new_data
            ]).drop_duplicates().tail(500)  # Keep last 500 candles

            # Process and generate signals
            processed = trader.process_smc_data(trader.historical_data)
            trader.last_processed = new_data.index[-1]
            signals = trader.generate_signals(processed)

            current_signal = signals['signal'].iloc[-1]
            print(current_signal)
            if current_signal != 0:
                if DEBUG_MODE:
                    print(f"\n=== EXECUTION TRIGGERED [{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] ===")
                execute_trade(current_signal, trader.historical_data)

            time.sleep(1)

        except KeyboardInterrupt:
            if DEBUG_MODE:
                print("\n🛑 Bot stopped by user")
            break
        except Exception as e:
            if DEBUG_MODE:
                print(f"⚠️ Unexpected error: {str(e)}")
            time.sleep(1)

if __name__ == "__main__":
    if initialize_mt5():
        main_loop()
    mt5.shutdown()
    if DEBUG_MODE:
        print("✅ MT5 connection closed")