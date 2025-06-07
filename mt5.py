# import MetaTrader5 as mt5
# import pandas as pd
# pd.set_option('future.no_silent_downcasting', True)
# import numpy as np
# import time
# from smartmoneyconcepts.smc import smc
# from datetime import datetime, timedelta
#
# # ========== Configuration ==========
# SYMBOL = "EURUSD"
# TIMEFRAME = "M15"
# NUM_CANDLES = 500
# DRY_RUN = False
# DEBUG_MODE = True
#
# # ========== ICT Configuration ==========
# ICT_CONFIG = {
#     "SWING_LENGTHS": {
#         "M1": 75,
#         "M5": 20,
#         "M15": 14,
#         "H1": 7,
#         "D1": 5
#     },
#     "FVG_EXPIRATION": {
#         "M5": timedelta(hours=6),
#         "M15": timedelta(hours=12),
#         "H1": timedelta(days=1),
#         "D1": timedelta(days=3)
#     },
#     "TRADING_SESSIONS": {
#         "London": (7, 16),    # UTC hours
#         "NewYork": (13, 22)   # UTC hours
#     },
#     "RISK_PERCENT": 1.0
# }
#
# def initialize_mt5():
#     if not mt5.initialize():
#         print("Failed to initialize MT5")
#         return False
#     print("✅ Successfully connected to MT5")
#     return True
#
# def fetch_realtime_data():
#     """Fetch data without timezone information"""
#     try:
#         rates = mt5.copy_rates_from_pos(SYMBOL, getattr(mt5, f'TIMEFRAME_{TIMEFRAME}'), 0, NUM_CANDLES)
#         if rates is None or len(rates) == 0:
#             return pd.DataFrame()
#         df = pd.DataFrame(rates)
#         df.rename(columns={
#             "time": "datetime",
#             "open": "open",
#             "high": "high",
#             "low": "low",
#             "close": "close",
#             "tick_volume": "volume"
#         }, inplace=True)
#         df["datetime"] = pd.to_datetime(df["datetime"], unit="s")
#         df.set_index("datetime", inplace=True)
#         return df[["open", "high", "low", "close", "volume"]]
#     except Exception as e:
#         print(f"Data error: {str(e)}")
#         return pd.DataFrame()
#
# class SMCTrader:
#     def __init__(self, symbol, timeframe):
#         self.symbol = symbol
#         self.timeframe = timeframe
#         self.swing_length = ICT_CONFIG["SWING_LENGTHS"][timeframe]
#         self.fvg_expiration = ICT_CONFIG["FVG_EXPIRATION"][timeframe]
#         self.historical_data = pd.DataFrame()
#         self.active_fvgs = pd.DataFrame()
#         self.pip_size = mt5.symbol_info(symbol).point * 10
#
#     def get_swing_data(self, df):
#         """Swing detection with fallback values"""
#         try:
#             swings = smc.swing_highs_lows(df, swing_length=self.swing_length)
#             if swings.empty:
#                 if DEBUG_MODE:
#                     print("No swing data detected; applying fallback values.")
#                 return pd.DataFrame({
#                     'HighLow': [1, -1],
#                     'Level': [df['high'].max(), df['low'].min()]
#                 })
#             # Ensure both swing types exist
#             for stype in [1, -1]:
#                 if swings[swings['HighLow'] == stype].empty:
#                     level = df['high'].max() if stype == 1 else df['low'].min()
#                     swings = swings.append({
#                         'HighLow': stype,
#                         'Level': level
#                     }, ignore_index=True)
#             return swings
#         except Exception as e:
#             print(f"Swing error: {str(e)}")
#             return pd.DataFrame({
#                 'HighLow': [1, -1],
#                 'Level': [df['high'].max(), df['low'].min()]
#             })
#
#     def validate_fvg(self, fvg_data):
#         """FVG validation without timezones"""
#         if fvg_data.empty:
#             if DEBUG_MODE:
#                 print("FVG data is empty.")
#             return fvg_data
#
#         try:
#             if not isinstance(fvg_data.index, pd.DatetimeIndex):
#                 # Align index with historical data timestamps
#                 fvg_data.index = self.historical_data.index[-len(fvg_data):]
#             # Get most recent datetime from historical data
#             current_time = self.historical_data.index[-1]
#             # Calculate FVG age
#             fvg_age = current_time - fvg_data.index
#             valid_fvg = fvg_data[fvg_age < self.fvg_expiration].copy()
#             # Check mitigation using the last close price
#             last_close = self.historical_data['close'].iloc[-1]
#             valid_fvg['Mitigated'] = np.where(
#                 (valid_fvg['FVG'] == 1) & (last_close < valid_fvg['Bottom']), True,
#                 np.where((valid_fvg['FVG'] == -1) & (last_close > valid_fvg['Top']), True, False)
#             )
#             if DEBUG_MODE:
#                 print("Validated FVGs:", valid_fvg.dropna())
#             return valid_fvg[~valid_fvg['Mitigated']]
#         except Exception as e:
#             print(f"FVG error: {str(e)}")
#             return pd.DataFrame()
#
#     def process_smc_data(self):
#         """Process all SMC components"""
#         try:
#             if self.historical_data.empty:
#                 return pd.DataFrame()
#
#             swings = self.get_swing_data(self.historical_data)
#             if DEBUG_MODE:
#                 print("Swings data:", swings.dropna())
#
#             ob_data = smc.ob(self.historical_data, swings)
#             if DEBUG_MODE:
#                 print("Order Block data:", ob_data.dropna())
#
#             fvg_data = smc.fvg(self.historical_data)
#
#             liquidity_data = smc.liquidity(self.historical_data, swings)
#             if DEBUG_MODE:
#                 print("Liquidity data:", liquidity_data.dropna())
#
#             bos_data = smc.bos_choch(self.historical_data, swings, close_break=True)
#             if DEBUG_MODE:
#                 print("BOS/CHOCH data:", bos_data.dropna())
#
#             self.active_fvgs = self.validate_fvg(fvg_data)
#             if DEBUG_MODE:
#                 print("FVG data:", self.active_fvgs.dropna())
#
#             # Combine data
#             combined = self.historical_data.copy()
#             for data, prefix in zip([ob_data, liquidity_data, bos_data, self.active_fvgs],
#                                     ['ob_', 'liq_', 'bos_', 'fvg_']):
#                 if not data.empty:
#                     combined = combined.join(data.add_prefix(prefix), how='left')
#
#             # Fill missing values and convert types
#             combined = combined.ffill()
#             combined = combined.infer_objects(copy=False)
#             if DEBUG_MODE:
#                 print("Combined SMC data columns:", combined.columns)
#             return combined
#
#         except Exception as e:
#             print(f"Processing error: {str(e)}")
#             return pd.DataFrame()
#
#     def generate_signals(self):
#         """Generate trading signals using OB, FVG, Liquidity, BOS/CHOCH and swing pattern"""
#         processed = self.process_smc_data()
#         if processed.empty:
#             if DEBUG_MODE:
#                 print("Processed data is empty; cannot generate signals.")
#             return pd.Series()
#
#         try:
#             # Get swing data and derive the swing signal from the last two swings (excluding the very last row)
#             swings = self.get_swing_data(self.historical_data)
#             swing_signal = 0  # default if pattern not met
#             if not swings.empty and len(swings) > 2:
#                 swings_filtered = swings.iloc[:-1]  # Exclude the last swing
#                 if len(swings_filtered) >= 2:
#                     last_two = swings_filtered.iloc[-2:]
#                     if last_two.iloc[0]['HighLow'] == -1 and last_two.iloc[1]['HighLow'] == 1:
#                         swing_signal = 1
#                     elif last_two.iloc[0]['HighLow'] == 1 and last_two.iloc[1]['HighLow'] == -1:
#                         swing_signal = -1
#             if DEBUG_MODE:
#                 print("Derived swing signal:", swing_signal)
#
#             # Ensure the required columns exist
#             required_columns = ['ob_OB', 'fvg_FVG', 'bos_BOS', 'bos_CHOCH', 'liq_Liquidity']
#             for col in required_columns:
#                 if col not in processed.columns:
#                     if DEBUG_MODE:
#                         print(f"Warning: {col} column missing from processed data.")
#                     processed[col] = np.nan
#
#             # ICT signal conditions:
#             # For a bullish signal, we require:
#             #   OB = 1, FVG = 1, Liquidity = 1, derived swing signal = 1, and either BOS = 1 or CHoCH = 1.
#             # For a bearish signal, we require:
#             #   OB = -1, FVG = -1, Liquidity = -1, derived swing signal = -1, and either BOS = -1 or CHoCH = -1.
#             bull_cond = (
#                     (processed['ob_OB'] == 1) &
#                     (processed['fvg_FVG'] == 1) &
#                     (processed['liq_Liquidity'] == 1) &
#                     (swing_signal == 1) &
#                     ((processed['bos_BOS'] == 1) | (processed['bos_CHOCH'] == 1))
#             )
#
#             bear_cond = (
#                     (processed['ob_OB'] == -1) &
#                     (processed['fvg_FVG'] == -1) &
#                     (processed['liq_Liquidity'] == -1) &
#                     (swing_signal == -1) &
#                     ((processed['bos_BOS'] == -1) | (processed['bos_CHOCH'] == -1))
#             )
#
#             signals = np.select([bull_cond, bear_cond], [1, -1], 0)
#             if DEBUG_MODE:
#                 print("Latest signal:", signals[-1])
#             return pd.Series(signals, index=processed.index)
#
#         except Exception as e:
#             print(f"Signal error: {str(e)}")
#             return pd.Series()
#
#     def calculate_position_size(self):
#         """Calculate position size with 1% risk using a fixed risk measure"""
#         try:
#             balance = mt5.account_info().balance
#             risk_amount = balance * 0.01
#             # Assume a fixed stop loss of 20 pips as risk measure
#             risk_pips = 20
#             pip_value = self.pip_size
#             size = round(risk_amount / (risk_pips * pip_value), 2)
#             return max(size, 0.01)
#         except Exception as e:
#             print(f"Position sizing error: {str(e)}")
#             return 0.01
#
#     def is_trading_session(self):
#         """Check UTC trading hours"""
#         try:
#             now = datetime.utcnow().hour
#             london = ICT_CONFIG["TRADING_SESSIONS"]["London"]
#             ny = ICT_CONFIG["TRADING_SESSIONS"]["NewYork"]
#             return (london[0] <= now <= london[1]) or (ny[0] <= now <= ny[1])
#         except Exception as e:
#             print(f"Trading session error: {str(e)}")
#             return True
#
# class ICTExecution:
#     def __init__(self, symbol):
#         self.symbol = symbol
#         self.max_positions = 3
#
#     def manage_trades(self):
#         """Manage open positions"""
#         try:
#             for position in mt5.positions_get(symbol=self.symbol) or []:
#                 self.update_stop_loss(position)
#         except Exception as e:
#             print(f"Trade mgmt error: {str(e)}")
#
#     def update_stop_loss(self, position):
#         """Trailing stop logic"""
#         try:
#             pip = mt5.symbol_info(self.symbol).point * 10
#             if position.type == 0:
#                 price = mt5.symbol_info_tick(self.symbol).ask
#                 new_sl = max(position.sl, price - 20 * pip)
#             else:
#                 price = mt5.symbol_info_tick(self.symbol).bid
#                 new_sl = min(position.sl, price + 20 * pip)
#             if new_sl != position.sl:
#                 mt5.order_send({
#                     "action": mt5.TRADE_ACTION_SLTP,
#                     "position": position.ticket,
#                     "sl": new_sl,
#                     "deviation": 20
#                 })
#         except Exception as e:
#             print(f"SL error: {str(e)}")
#
#     def execute_order(self, signal, price, sl, tp, size):
#         """Execute trade order"""
#         if self.get_open_trades() >= self.max_positions:
#             if DEBUG_MODE:
#                 print("Maximum positions reached.")
#             return None
#         try:
#             order_type = mt5.ORDER_TYPE_BUY if signal == 1 else mt5.ORDER_TYPE_SELL
#             request = {
#                 "action": mt5.TRADE_ACTION_DEAL,
#                 "symbol": self.symbol,
#                 "volume": size,
#                 "type": order_type,
#                 "price": price,
#                 "sl": sl,
#                 "tp": tp,
#                 "deviation": 20,
#                 "magic": 10032023,
#                 "comment": "ICT Strategy"
#             }
#             result = mt5.order_send(request)
#             if DEBUG_MODE:
#                 print("Order sent result:", result)
#             return result
#         except Exception as e:
#             print(f"Order error: {str(e)}")
#             return None
#
#     def get_open_trades(self):
#         """Count open positions"""
#         try:
#             return len(mt5.positions_get(symbol=self.symbol) or [])
#         except Exception as e:
#             print(f"Get open trades error: {str(e)}")
#             return 0
#
# def main_loop():
#     if not initialize_mt5():
#         return
#
#     trader = SMCTrader(SYMBOL, TIMEFRAME)
#     executor = ICTExecution(SYMBOL)
#     print("🚀 Starting Trading Bot...")
#
#     # Data collection
#     print("🕒 Loading historical data...")
#     start_time = time.time()
#     while time.time() - start_time < 60:  # 1 minute timeout
#         new_data = fetch_realtime_data()
#         if not new_data.empty:
#             trader.historical_data = pd.concat([
#                 trader.historical_data,
#                 new_data
#             ]).drop_duplicates().tail(NUM_CANDLES)
#             if len(trader.historical_data) >= NUM_CANDLES:
#                 print(f"✅ Collected {NUM_CANDLES} candles")
#                 break
#         time.sleep(5)
#     else:
#         print("❌ Data collection failed")
#         mt5.shutdown()
#         return
#
#     # Main loop
#     while True:
#         try:
#             # Check market hours
#             if not trader.is_trading_session():
#                 time.sleep(60)
#                 continue
#
#             # Get new data
#             new_data = fetch_realtime_data()
#             if new_data.empty:
#                 time.sleep(5)
#                 continue
#
#             # Update historical data
#             trader.historical_data = pd.concat([
#                 trader.historical_data,
#                 new_data
#             ]).drop_duplicates().tail(NUM_CANDLES)
#
#             # Generate signals
#             signals = trader.generate_signals()
#             if signals.empty:
#                 if DEBUG_MODE:
#                     print("No signals generated.")
#                 time.sleep(1)
#                 continue
#
#             signal = signals.iloc[-1]
#             if DEBUG_MODE:
#                 print("Latest generated signal:", signal)
#
#             # Execute trade if signal exists
#             if signal != 0:
#                 price = (mt5.symbol_info_tick(SYMBOL).ask if signal == 1
#                          else mt5.symbol_info_tick(SYMBOL).bid)
#
#                 # Get swing data from the most recent 100 candles
#                 swings = trader.get_swing_data(trader.historical_data.tail(100))
#                 if DEBUG_MODE:
#                     print("Recent swings:", swings)
#
#                 if signal == 1:
#                     filtered_swings = swings[swings['HighLow'] == -1]
#                 else:
#                     filtered_swings = swings[swings['HighLow'] == 1]
#
#                 if not filtered_swings.empty:
#                     sl = filtered_swings['Level'].values[-1]
#                     tp = price + 2 * (price - sl) if signal == 1 else price - 2 * (sl - price)
#                 else:
#                     # Fallback SL/TP if swing data is missing
#                     sl = price * 0.995 if signal == 1 else price * 1.005
#                     tp = price * 1.015 if signal == 1 else price * 0.985
#
#                 if abs(price - sl) > 10 * trader.pip_size:
#                     size = trader.calculate_position_size()
#                     executor.execute_order(signal, price, sl, tp, size)
#
#             # Manage open trades
#             executor.manage_trades()
#             time.sleep(1)
#
#         except KeyboardInterrupt:
#             print("\n🛑 Bot stopped by user")
#             break
#         except Exception as e:
#             print(f"Main error: {str(e)}")
#             time.sleep(1)
#
#     mt5.shutdown()
#     print("✅ Connection closed")
#
# if __name__ == "__main__":
#     main_loop()

# import MetaTrader5 as mt5
# import pandas as pd
# pd.set_option('future.no_silent_downcasting', True)
# import numpy as np
# import time
# from smartmoneyconcepts.smc import smc
# from datetime import datetime, timedelta
#
# # ========== Configuration ==========
# SYMBOL = "BTCUSD"
# TIMEFRAME = "M15"
# NUM_CANDLES = 500
# DRY_RUN = False
# DEBUG_MODE = True
#
# # ========== ICT Configuration ==========
# ICT_CONFIG = {
#     "SWING_LENGTHS": {
#         "M1": 75,
#         "M5": 20,
#         "M15": 14,
#         "H1": 7,
#         "D1": 5
#     },
#     "FVG_EXPIRATION": {
#         "M1": pd.Timedelta(hours=2),
#         "M5": pd.Timedelta(hours=6),
#         "M15": pd.Timedelta(hours=12),
#         "H1": pd.Timedelta(days=1),
#         "D1": pd.Timedelta(days=3)
#     },
#     "TRADING_SESSIONS": {
#         "London": (7, 16),
#         "NewYork": (13, 22)
#     },
#     "RISK_PERCENT": 1.0
# }
#
# def initialize_mt5():
#     if not mt5.initialize():
#         print("Failed to initialize MT5")
#         return False
#     print("✅ Successfully connected to MT5")
#     return True
#
# def fetch_realtime_data():
#     """Fetch and format market data with proper type handling"""
#     try:
#         timeframe = getattr(mt5, f'TIMEFRAME_{TIMEFRAME}', mt5.TIMEFRAME_M15)
#         rates = mt5.copy_rates_from_pos(SYMBOL, timeframe, 0, NUM_CANDLES)
#
#         if rates is None or len(rates) == 0:
#             return pd.DataFrame()
#
#         df = pd.DataFrame(rates)
#         df['datetime'] = pd.to_datetime(df['time'], unit='s', errors='coerce')
#         df = df.dropna(subset=['datetime']).set_index('datetime')
#         return df[['open', 'high', 'low', 'close', 'tick_volume']] \
#             .rename(columns={'tick_volume': 'volume'}) \
#             .sort_index(ascending=True)
#     except Exception as e:
#         print(f"Data error: {str(e)}")
#         return pd.DataFrame()
#
# def is_trading_session():
#     """Session check with proper time comparisons"""
#     try:
#         now = datetime.utcnow()
#         l_start, l_end = ICT_CONFIG["TRADING_SESSIONS"]["London"]
#         ny_start, ny_end = ICT_CONFIG["TRADING_SESSIONS"]["NewYork"]
#
#         london = (now >= now.replace(hour=l_start, minute=0, second=0)) & \
#                  (now <= now.replace(hour=l_end, minute=0, second=0))
#         newyork = (now >= now.replace(hour=ny_start, minute=0, second=0)) & \
#                   (now <= now.replace(hour=ny_end, minute=0, second=0))
#         return london or newyork
#     except Exception as e:
#         print(f"Session error: {str(e)}")
#         return True
#
# class SMCTrader:
#     def __init__(self, symbol, timeframe):
#         self.symbol = symbol
#         self.timeframe = timeframe
#         self.swing_length = ICT_CONFIG["SWING_LENGTHS"][timeframe]
#         self.fvg_expiration = ICT_CONFIG["FVG_EXPIRATION"][timeframe]
#         self.historical_data = pd.DataFrame()
#         self.pip_size = mt5.symbol_info(symbol).point * 10
#
#     # ========== Enhanced Validation Methods ==========
#     def validate_swings(self, swings_data):
#         """Swing validation me trajtim të duhur të datave"""
#         try:
#             if swings_data.empty:
#                 return self.generate_fallback_swings()
#
#             # Konverto indeksin në DatetimeIndex nëse është e nevojshme
#             swings_data.index = pd.to_datetime(swings_data.index)
#
#             current_time = self.historical_data.index[-1]
#             start_date = current_time - self.fvg_expiration
#
#             # Krijo maskën duke përdorur krahasime të drejtpërdrejta të datave
#             valid_swings = swings_data.loc[swings_data.index >= start_date].copy()
#             valid_swings['Valid'] = valid_swings['HighLow'].diff().abs() >= 1
#             valid_swings = valid_swings[valid_swings['Valid']]
#
#             return valid_swings.iloc[-3:] if not valid_swings.empty else self.generate_fallback_swings()
#         except Exception as e:
#             print(f"Gabim në validimin e swinɡve: {str(e)}")
#             return self.generate_fallback_swings()
#
#     def validate_ob(self, ob_data):
#         """Validimi i OB me trajtim të saktë të datave"""
#         try:
#             if ob_data.empty or not {'OB', 'Top', 'Bottom'}.issubset(ob_data.columns):
#                 return self.generate_fallback_ob()
#
#             # Konverto indeksin në DatetimeIndex
#             ob_data.index = pd.to_datetime(ob_data.index)
#
#             current_time = self.historical_data.index[-1]
#             start_date = current_time - self.fvg_expiration
#
#             valid_ob = ob_data.loc[ob_data.index >= start_date].copy()
#             valid_ob['Valid'] = np.where(
#                 valid_ob['OB'] == 1,
#                 (self.historical_data['close'].values > valid_ob['Bottom'].values) &
#                 (self.historical_data['close'].values < valid_ob['Top'].values),
#                 (self.historical_data['close'].values < valid_ob['Top'].values) &
#                 (self.historical_data['close'].values > valid_ob['Bottom'].values)
#             )
#             return valid_ob[valid_ob['Valid']].iloc[-2:]
#         except Exception as e:
#             print(f"Gabim në OB: {str(e)}")
#             return self.generate_fallback_ob()
#
#     def validate_fvg(self, fvg_data):
#         """Validimi i FVG me trajtim të saktë të datave"""
#         try:
#             if fvg_data.empty:
#                 return self.generate_fallback_fvg()
#
#             # Konverto indeksin në DatetimeIndex
#             fvg_data.index = pd.to_datetime(fvg_data.index)
#
#             current_time = self.historical_data.index[-1]
#             start_date = current_time - self.fvg_expiration
#
#             valid_fvg = fvg_data.loc[fvg_data.index >= start_date].copy()
#             valid_fvg['Mitigated'] = np.where(
#                 valid_fvg['FVG'] == 1,
#                 self.historical_data['close'].values >= valid_fvg['Bottom'].values,
#                 self.historical_data['close'].values <= valid_fvg['Top'].values
#             )
#             return valid_fvg[~valid_fvg['Mitigated']]
#         except Exception as e:
#             print(f"Gabim në FVG: {str(e)}")
#             return self.generate_fallback_fvg()
#
#     # ========== Improved Fallback Generators ==========
#     def generate_fallback_swings(self):
#         """Generate fallback swings with proper index"""
#         try:
#             df = self.historical_data.tail(50)
#             return pd.DataFrame({
#                 'HighLow': [1, -1],
#                 'Level': [df['high'].max(), df['low'].min()],
#                 'Valid': [True, True]
#             }, index=pd.DatetimeIndex([
#                 df.index[-1] - pd.Timedelta(minutes=15),
#                 df.index[-1]
#             ]))
#         except:
#             return pd.DataFrame(columns=['HighLow', 'Level', 'Valid'],
#                                 index=pd.DatetimeIndex([]))
#
#     def generate_fallback_ob(self):
#         """Generate fallback OB with valid structure"""
#         try:
#             df = self.historical_data.tail(20)
#             return pd.DataFrame({
#                 'OB': [1 if df['close'].iloc[-1] > df['close'].mean() else -1],
#                 'Top': [df['high'].max()],
#                 'Bottom': [df['low'].min()],
#                 'Valid': [True]
#             }, index=pd.DatetimeIndex([df.index[-1]]))
#         except:
#             return pd.DataFrame(columns=['OB', 'Top', 'Bottom', 'Valid'],
#                                 index=pd.DatetimeIndex([]))
#
#     def generate_fallback_fvg(self):
#         """Generate fallback FVG with valid index"""
#         try:
#             df = self.historical_data.tail(10)
#             return pd.DataFrame({
#                 'FVG': [1 if df['close'].iloc[-3] > df['open'].iloc[-3] else -1],
#                 'Top': df['high'].iloc[-3],
#                 'Bottom': df['low'].iloc[-3],
#                 'Mitigated': [False]
#             }, index=pd.DatetimeIndex([df.index[-3]]))
#         except:
#             return pd.DataFrame(columns=['FVG', 'Top', 'Bottom', 'Mitigated'],
#                                 index=pd.DatetimeIndex([]))
#
#     # ========== Core Strategy Logic ==========
#     def process_smc_data(self):
#         """Process market structure data"""
#         try:
#             self.historical_data = self.historical_data.dropna().sort_index()
#
#             swings = smc.swing_highs_lows(self.historical_data, self.swing_length)
#             ob_data = smc.ob(self.historical_data, swings)
#             fvg_data = smc.fvg(self.historical_data)
#
#             processed = self.historical_data.copy()
#             processed = processed.join([
#                 self.validate_swings(swings).add_prefix('swing_'),
#                 self.validate_ob(ob_data).add_prefix('ob_'),
#                 self.validate_fvg(fvg_data).add_prefix('fvg_')
#             ], how='left')
#
#             return processed.ffill().dropna(how='all')
#         except Exception as e:
#             print(f"Processing error: {str(e)}")
#             return self.historical_data
#
#     def get_swing_data(self, df):
#         """Get swing points with proper error handling"""
#         try:
#             swings = smc.swing_highs_lows(df, self.swing_length)
#             return swings if not swings.empty else pd.DataFrame({
#                 'HighLow': [1, -1],
#                 'Level': [df['high'].max(), df['low'].min()]
#             }, index=df.index[-2:])
#         except Exception as e:
#             print(f"Swing detection error: {str(e)}")
#             return pd.DataFrame({
#                 'HighLow': [1, -1],
#                 'Level': [df['high'].max(), df['low'].min()]
#             }, index=df.index[-2:])
#
#     def generate_signals(self):
#         """Generate trading signals with confluence check"""
#         processed = self.process_smc_data()
#         if processed.empty:
#             return 0
#
#         try:
#             ob = processed['ob_OB'].ffill().iloc[-1]
#             fvg = processed['fvg_FVG'].ffill().iloc[-1]
#             swing = processed['swing_HighLow'].ffill().iloc[-1]
#
#             bull = sum([1 if ob == 1 else 0, 1 if fvg == 1 else 0, 1 if swing == 1 else 0])
#             bear = sum([1 if ob == -1 else 0, 1 if fvg == -1 else 0, 1 if swing == -1 else 0])
#
#             signal = 1 if bull >= 2 else -1 if bear >= 2 else 0
#
#             if DEBUG_MODE:
#                 print(f"\nSignal Report:")
#                 print(f"OB: {ob} | FVG: {fvg} | Swing: {swing}")
#                 print(f"Bullish: {bull}/3 | Bearish: {bear}/3")
#                 print(f"Decision: {'Buy' if signal == 1 else 'Sell' if signal == -1 else 'Wait'}")
#
#             return signal
#         except Exception as e:
#             print(f"Signal error: {str(e)}")
#             return 0
#
#     def calculate_position_size(self):
#         """Calculate position size with volatility adjustment"""
#         try:
#             balance = mt5.account_info().balance
#             risk_amount = balance * (ICT_CONFIG["RISK_PERCENT"] / 100)
#             atr = (self.historical_data['high'].tail(14) - self.historical_data['low'].tail(14)).mean()
#             atr = max(atr * self.pip_size, 10 * self.pip_size)
#             return round(risk_amount / atr, 2)
#         except Exception as e:
#             print(f"Position error: {str(e)}")
#             return 0.01
#
# class ICTExecution:
#     def __init__(self, symbol):
#         self.symbol = symbol
#         self.max_positions = 3
#         self.trader = None
#
#     def manage_trades(self):
#         """Manage open positions with proper error handling"""
#         try:
#             for position in mt5.positions_get(symbol=self.symbol) or []:
#                 self.update_stop_loss(position)
#                 self.move_to_breakeven(position)
#         except Exception as e:
#             print(f"Trade error: {str(e)}")
#
#     def calculate_stop_levels(self, signal, trader):
#         """Calculate stop levels with validation"""
#         try:
#             swings = trader.get_swing_data(trader.historical_data.tail(100))
#             current_price = trader.historical_data['close'].iloc[-1]
#             pip = trader.pip_size
#
#             if signal == 1:  # Buy
#                 sl = (swings[swings['HighLow'] == -1]['Level'].iloc[-1]
#                       if not swings.empty else current_price - 20 * pip)
#                 sl = min(sl, current_price - pip)  # Ensure SL below price
#                 tp = current_price + 3 * (current_price - sl)
#             else:  # Sell
#                 sl = (swings[swings['HighLow'] == 1]['Level'].iloc[-1]
#                       if not swings.empty else current_price + 20 * pip)
#                 sl = max(sl, current_price + pip)  # Ensure SL above price
#                 tp = current_price - 3 * (sl - current_price)
#
#             return (round(sl, 5), round(tp, 5))
#         except Exception as e:
#             print(f"Stop calculation error: {str(e)}")
#             current_price = trader.historical_data['close'].iloc[-1]
#             pip = trader.pip_size
#             if signal == 1:
#                 return (round(current_price - 20 * pip, 5),
#                         round(current_price + 40 * pip, 5))
#             else:
#                 return (round(current_price + 20 * pip, 5),
#                         round(current_price - 40 * pip, 5))
#
#     def update_stop_loss(self, position):
#         """Update stops with volatility adjustment"""
#         try:
#             pip = mt5.symbol_info(self.symbol).point * 10
#             tick = mt5.symbol_info_tick(self.symbol)
#             df = self.trader.historical_data.tail(20)
#
#             volatility = (df['high'] - df['low']).mean() * pip
#             trail_dist = max(volatility * 0.5, 15)
#
#             if position.type == 0:  # Buy
#                 new_sl = tick.bid - trail_dist
#                 new_sl = max(new_sl, position.sl)
#             else:  # Sell
#                 new_sl = tick.ask + trail_dist
#                 new_sl = min(new_sl, position.sl)
#
#             if abs(new_sl - position.sl) > 5 * pip:
#                 self.modify_position(position, new_sl)
#         except Exception as e:
#             print(f"SL error: {str(e)}")
#
#     def modify_position(self, position, new_sl):
#         """Modify position with validation"""
#         request = {
#             "action": mt5.TRADE_ACTION_SLTP,
#             "position": position.ticket,
#             "sl": new_sl,
#             "deviation": 20
#         }
#         result = mt5.order_send(request)
#         if result.retcode != mt5.TRADE_RETCODE_DONE:
#             print(f"Modify position failed: {result.comment}")
#
#     def execute_order(self, signal, price, sl, tp, size):
#         """Execute order with proper validation"""
#         try:
#             if self.get_open_trades() >= self.max_positions:
#                 return None
#
#             # Validate stop levels
#             if signal == 1 and sl >= price:
#                 sl = price - 10 * self.trader.pip_size
#                 tp = price + 30 * self.trader.pip_size
#             elif signal == -1 and sl <= price:
#                 sl = price + 10 * self.trader.pip_size
#                 tp = price - 30 * self.trader.pip_size
#
#             request = {
#                 "action": mt5.TRADE_ACTION_DEAL,
#                 "symbol": self.symbol,
#                 "volume": round(size, 2),
#                 "type": mt5.ORDER_TYPE_BUY if signal == 1 else mt5.ORDER_TYPE_SELL,
#                 "price": price,
#                 "sl": sl,
#                 "tp": tp,
#                 "deviation": 20,
#                 "magic": 10032023,
#                 "comment": "ICT Strategy"
#             }
#
#             result = mt5.order_send(request)
#             if result.retcode != mt5.TRADE_RETCODE_DONE:
#                 print(f"Order failed: {result.comment}")
#             return result
#         except Exception as e:
#             print(f"Order error: {str(e)}")
#             return None
#
#     def get_open_trades(self):
#         """Count open positions"""
#         return len(mt5.positions_get(symbol=self.symbol) or [])
#
# def main_loop():
#     if not initialize_mt5():
#         return
#
#     trader = SMCTrader(SYMBOL, TIMEFRAME)
#     executor = ICTExecution(SYMBOL)
#     executor.trader = trader
#
#     print("🚀 Starting Trading Bot...")
#
#     # Data initialization
#     print("🕒 Loading market data...")
#     start = time.time()
#     while time.time() - start < 120:
#         df = fetch_realtime_data()
#         if not df.empty:
#             trader.historical_data = pd.concat([trader.historical_data, df]) \
#                 .tail(NUM_CANDLES) \
#                 .drop_duplicates() \
#                 .sort_index()
#             if len(trader.historical_data) >= 500:
#                 print(f"✅ Data loaded ({len(trader.historical_data)} candles)")
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
#             if not is_trading_session():
#                 time.sleep(300)
#                 continue
#
#             # Update market data
#             new_data = fetch_realtime_data()
#             if not new_data.empty:
#                 trader.historical_data = pd.concat([trader.historical_data, new_data]) \
#                     .tail(NUM_CANDLES) \
#                     .drop_duplicates() \
#                     .sort_index()
#
#             # Generate and execute signals
#             signal = trader.generate_signals()
#             if signal != 0:
#                 tick = mt5.symbol_info_tick(SYMBOL)
#                 price = round(tick.ask if signal == 1 else tick.bid, 5)
#                 sl, tp = executor.calculate_stop_levels(signal, trader)
#
#                 # Ensure valid stop levels
#                 if (signal == 1 and sl < price) or (signal == -1 and sl > price):
#                     size = trader.calculate_position_size()
#                     if not DRY_RUN:
#                         executor.execute_order(signal, price, sl, tp, size)
#                 else:
#                     print(f"Invalid stops: Price {price} | SL {sl}")
#
#             # Manage trades
#             executor.manage_trades()
#             time.sleep(30)
#
#         except KeyboardInterrupt:
#             print("\n🛑 Bot stopped by user")
#             break
#         except Exception as e:
#             print(f"Main error: {str(e)}")
#             time.sleep(60)
#
# if __name__ == "__main__":
#     main_loop()

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
from smartmoneyconcepts.smc import smc
from datetime import datetime, timedelta

# ========== Configuration ==========
SYMBOL = "BTCUSD"
TIMEFRAME = "M5"
NUM_CANDLES = 500
DRY_RUN = False
DEBUG_MODE = True

# ========== ICT Configuration ==========
ICT_CONFIG = {
    "SWING_LENGTHS": {
        "M1": 75,
        "M5": 20,
        "M15": 35,
        "H1": 7,
        "D1": 5
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

def fetch_realtime_data():
    """Fetch live data from MT5 with UTC timezone."""
    try:
        tf = getattr(mt5, f'TIMEFRAME_{TIMEFRAME}')
        rates = mt5.copy_rates_from_pos(SYMBOL, tf, 0, NUM_CANDLES)
        if rates is None or len(rates) == 0:
            return pd.DataFrame()

        df = pd.DataFrame(rates)
        df.rename(columns={
            "time": "datetime",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "tick_volume": "tick_volume",
            "real_volume": "real_volume"
        }, inplace=True)

        if "real_volume" in df.columns and df["real_volume"].sum() > 0:
            df["volume"] = df["real_volume"]
        elif "tick_volume" in df.columns and df["tick_volume"].sum() > 0:
            df["volume"] = df["tick_volume"]
        else:
            df["volume"] = 1

        df["datetime"] = pd.to_datetime(df["datetime"], unit="s", utc=True)
        df.set_index("datetime", inplace=True)
        print(df[["open", "high", "low", "close", "volume"]])
        return df[["open", "high", "low", "close", "volume"]]

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
        self.active_fvgs = pd.DataFrame()
        self.pip_size = mt5.symbol_info(symbol).point * 10

    def is_trading_session(self):
        """Check active trading sessions"""
        try:
            now = datetime.utcnow().hour
            london = ICT_CONFIG["TRADING_SESSIONS"]["London"]
            ny = ICT_CONFIG["TRADING_SESSIONS"]["NewYork"]
            return (london[0] <= now <= london[1]) or (ny[0] <= now <= ny[1])
        except Exception as e:
            print(f"Session check error: {e}")
            return True

    def get_swing_data(self, df):
        """Detect swing points with fallback and correct index types"""
        try:
            # reset index for SMC, then map back to datetime index
            temp = df.reset_index(drop=True)
            swings = smc.swing_highs_lows(temp, swing_length=self.swing_length)
            if not isinstance(swings, pd.DataFrame):
                print("⚠️ Swing data is not a DataFrame. Using fallback.")
                swings = pd.DataFrame()

                if swings.empty or len(swings) < 3:  # Need at least 3 swings to skip the last one
                    if DEBUG_MODE:
                        print("Using fallback swings")
                    return pd.DataFrame({
                        'HighLow': [1, -1],
                        'Level': [df['high'].max(), df['low'].min()]
                    }, index=df.index[-3:-1])  # Use second and third last points instead of last two
    
                # ensure both types present
                for stype in [1, -1]:
                    if swings[swings['HighLow'] == stype].empty:
                        level = df['high'].max() if stype == 1 else df['low'].min()
                        swings = pd.concat([swings, pd.DataFrame({
                            'HighLow': [stype],
                            'Level': [level]
                        }, index=[len(temp)-2])])  # Add at second-to-last position instead of last
    
                # map integer index back to datetime
                swings.index = df.index[swings.index]
                
                # Return swings excluding the last valid one
                if len(swings) > 2:
                    return swings.iloc[-3:-1]  # Return third-last and second-last instead of last two
                else:
                    return swings.iloc[:-1]  # If we have only 2 swings, return just the first one
            except Exception as e:
                print(f"Swing error: {e}")
                return pd.DataFrame({
                    'HighLow': [1, -1],
                    'Level': [df['high'].max(), df['low'].min()]
                }, index=df.index[-3:-1])  # Use second and third last points in fallback too

    def validate_fvg(self, fvg_data):
        """Validate FVGs"""
        if not isinstance(fvg_data, pd.DataFrame):
            print("⚠️ Invalid FVG data type")
            return pd.DataFrame()

        if fvg_data.empty:
            return fvg_data
        try:
            fvg_data = fvg_data.reindex(self.historical_data.index, method='ffill')
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
        """Validate Order Blocks"""
        if not isinstance(ob_data, pd.DataFrame):
            return pd.DataFrame()
        if ob_data.empty:
            return ob_data

        try:
            # Konverto kolonat kritike në float
            ob_data['Top'] = pd.to_numeric(ob_data['Top'], errors='coerce')
            ob_data['Bottom'] = pd.to_numeric(ob_data['Bottom'], errors='coerce')

            # Hiq rreshtat me vlera NaN pas konvertimit
            ob_data.dropna(subset=['Top', 'Bottom'], inplace=True)

            # Rindekso dhe krahaso me çmimin aktual
            ob_data = ob_data.reindex(self.historical_data.index, method='ffill')
            current_price = self.historical_data['close'].iloc[-1]

            # Krijo kushtet e mitigimit
            ob_data['Mitigated'] = np.where(
                (ob_data['OB'] == 1) & (current_price > ob_data['Top']),
                True,
                np.where((ob_data['OB'] == -1) & (current_price < ob_data['Bottom']),
                         True, False)
            )
            return ob_data[~ob_data['Mitigated']]
        except Exception as e:
            print(f"OB validation error: {e}")
            return pd.DataFrame()

    def validate_bos(self, bos_data):
        """Validate BOS/CHOCH"""
        if not isinstance(bos_data, pd.DataFrame):
            print("⚠️ Invalid BOS data type")
            return pd.DataFrame()

        if bos_data.empty:
            return bos_data
        try:
            bos_data = bos_data.reindex(self.historical_data.index, method='ffill')
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
            liq_data = liq_data.reindex(self.historical_data.index, method='ffill')
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
            if self.historical_data.empty:
                return pd.DataFrame()

            # prepare integer-indexed temp for SMC
            temp = self.historical_data.reset_index(drop=True)

            # swings
            swings = self.get_swing_data(self.historical_data)

            bos_data = pd.DataFrame()
            ob_data = pd.DataFrame()
            fvg_data = pd.DataFrame()
            liq_data = pd.DataFrame()

            # BOS
            try:
                bos_raw = smc.bos_choch(temp, swings, close_break=True)
                if isinstance(bos_raw, pd.DataFrame):
                    # map index back
                    bos_raw.index = self.historical_data.index[bos_raw.index]
                    bos_data = self.validate_bos(bos_raw)
                else:
                    print(f"⚠️ BOS returned non-DataFrame: {type(bos_raw)}")
            except Exception as e:
                print(f"⚠️ BOS Error: {str(e)}")

            # Order Blocks
            try:
                ob_raw = smc.ob(temp, swings)
                if isinstance(ob_raw, pd.DataFrame):
                    ob_raw.index = self.historical_data.index[ob_raw.index]
                    ob_data = self.validate_ob(ob_raw)
            except Exception as e:
                print(f"⚠️ OB Error: {str(e)}")

            # Fair Value Gaps
            try:
                fvg_raw = smc.fvg(temp)
                if isinstance(fvg_raw, pd.DataFrame):
                    fvg_raw.index = self.historical_data.index[fvg_raw.index]
                    fvg_data = self.validate_fvg(fvg_raw)
            except Exception as e:
                print(f"⚠️ FVG Error: {str(e)}")

            # Liquidity
            try:
                liq_raw = smc.liquidity(temp, swings)
                if isinstance(liq_raw, pd.DataFrame):
                    liq_raw.index = self.historical_data.index[liq_raw.index]
                    liq_data = self.validate_liquidity(liq_raw)
            except Exception as e:
                print(f"⚠️ Liquidity Error: {str(e)}")

            # combine
            combined = pd.concat([
                self.historical_data,
                bos_data.add_prefix('bos_'),
                ob_data.add_prefix('ob_'),
                fvg_data.add_prefix('fvg_'),
                liq_data.add_prefix('liq_')
            ], axis=1).ffill().fillna(0)

            return combined
        except Exception as e:
            print(f"⛔ Processing error: {str(e)}")
            import traceback; traceback.print_exc()
            return pd.DataFrame()

    def calculate_position_size(self):
        """Calculate position size with 1% risk"""
        try:
            balance = mt5.account_info().balance
            if balance <= 0:
                raise ValueError("Balance cannot be zero or negative!")

            risk_amount = balance * ICT_CONFIG["RISK_PERCENT"] / 100
            risk_pips = 20  # 20 pips
            pip_value = 1 / self.pip_size  # Për BTCUSD 1 pip = 0.1
            size = risk_amount / (risk_pips * pip_value)
            return round(max(size, 0.01), 2)
        except Exception as e:
            print(f"Position size error: {e}")
            return 0.01

    def generate_signals(self):
        """Generate trading signals"""
        processed = self.process_smc_data()
        if processed.empty:
            return pd.Series()

        processed['signal'] = 0
        swing_signal = 0

        try:
            swings = self.get_swing_data(processed)
            if len(swings) >= 2:
                last_two = swings.iloc[-2:]
                if last_two.iloc[0]['HighLow'] == -1 and last_two.iloc[1]['HighLow'] == 1:
                    swing_signal = 1
                elif last_two.iloc[0]['HighLow'] == 1 and last_two.iloc[1]['HighLow'] == -1:
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
            'liq_Liquidity': 0
        }
        for col, default in required_columns.items():
            if col not in processed.columns:
                processed[col] = default

        bull_cond = (
                            (processed['ob_OB'] == 1) &
                            (processed['close'] > processed['ob_Top']) &
                            (processed['fvg_FVG'] == 1) &
                            (processed['liq_Liquidity'] == 1) &
                            (swing_signal == 1) &
                            ((processed['bos_BOS'] == 1) | (processed['bos_CHOCH'] == 1)))

        bear_cond = (
                    (processed['ob_OB'] == -1) &
                    (processed['close'] < processed['ob_Bottom']) &
                    (processed['fvg_FVG'] == -1) &
                    (processed['liq_Liquidity'] == -1) &
                    (swing_signal == -1) &
                    ((processed['bos_BOS'] == -1) | (processed['bos_CHOCH'] == -1)))

        processed.loc[bull_cond, 'signal'] = 1
        processed.loc[bear_cond, 'signal'] = -1

        if DEBUG_MODE:
            print("\n=== DEBUG SIGNALS ===")
        print(processed[['signal', 'ob_OB', 'fvg_FVG', 'bos_BOS']].tail())
        print(f"Current signal: {processed['signal'].iloc[-1]}")

        return processed['signal']


class ICTExecution:
    def __init__(self, symbol):
        self.symbol = symbol
        self.max_positions = 3

    def manage_trades(self):
        """Manage open positions"""
        try:
            positions = mt5.positions_get(symbol=self.symbol)
            if positions is None:
                return

            for position in positions:
                pip = mt5.symbol_info(self.symbol).point * 10
                current_price = mt5.symbol_info_tick(self.symbol).ask if position.type == 0 else mt5.symbol_info_tick(
                    self.symbol).bid

                new_sl = position.sl
                if position.type == 0:  # Long
                    new_sl = max(position.sl, current_price - 20 * pip)
                else:  # Short
                    new_sl = min(position.sl, current_price + 20 * pip)

                if new_sl != position.sl:
                    mt5.order_send({
                        "action": mt5.TRADE_ACTION_SLTP,
                        "position": position.ticket,
                        "sl": new_sl,
                        "deviation": 20
                    })
        except Exception as e:
            print(f"Trade management error: {e}")

    def execute_order(self, signal, price, sl, tp, size):
        """Execute trade order"""
        if self.get_open_trades() >= self.max_positions:
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
            "magic": 10032023,
            "comment": "ICT Strategy"
        }

        if not DRY_RUN:
            result = mt5.order_send(request)
            if DEBUG_MODE:
                print(f"Order executed: {result}")
            return result
        else:
            print(f"DRY RUN: {order_type} {self.symbol} {size} lots @ {price}")
            return None

    def get_open_trades(self):
        """Count open positions"""
        try:
            return len(mt5.positions_get(symbol=self.symbol) or [])
        except Exception as e:
            print(f"Position count error: {e}")
            return 0


def main_loop():
    if not initialize_mt5():
        return

    trader = SMCTrader(SYMBOL, TIMEFRAME)
    executor = ICTExecution(SYMBOL)
    print("🚀 Starting Trading Bot...")

    # Load initial data
    start_time = time.time()
    while time.time() - start_time < 60:
        df = fetch_realtime_data()
        if not df.empty:
            trader.historical_data = df.iloc[-NUM_CANDLES:]
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
            if not trader.is_trading_session():
                if DEBUG_MODE:
                    print("💤 Outside trading session")
                time.sleep(60)
                continue

            new_data = fetch_realtime_data()
            if new_data.empty:
                time.sleep(1)
                continue

            # Update historical data
            trader.historical_data = pd.concat([
                trader.historical_data,
                new_data
            ]).drop_duplicates().iloc[-NUM_CANDLES:]

            # Generate and execute signals
            signals = trader.generate_signals()
            current_signal = signals.iloc[-1] if not signals.empty else 0

            if current_signal != 0:
                price = mt5.symbol_info_tick(SYMBOL).ask if current_signal == 1 else mt5.symbol_info_tick(SYMBOL).bid
                swings = trader.get_swing_data(trader.historical_data)

                if current_signal == 1:
                    sl = swings[swings['HighLow'] == -1]['Level'].values[-1]
                    tp = price + 2 * (price - sl)
                else:
                    sl = swings[swings['HighLow'] == 1]['Level'].values[-1]
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