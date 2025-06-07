import time
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from talib import RSI, STOCH, ATR

# Initialize MT5 connection
if not mt5.initialize():
    print("MT5 initialization failed. Error code:", mt5.last_error())
    quit()

# Strategy Parameters
SYMBOL = "BTCUSD"
TIMEFRAME = mt5.TIMEFRAME_H1
EMA_PERIOD = 50
STOCH_RSI_PERIOD = 14
ATR_PERIOD = 14
ATR_MULTIPLIER = 3  # For trailing stop distance
RISK_PERCENT = 2  # Risk per trade (2% of equity)


def get_ema(data, period=50):
    ema = data['close'].ewm(span=period, adjust=False).mean()
    print(f"[DEBUG] EMA calculation - Period: {period}, Latest EMA: {ema.iloc[-1]}")
    return ema


def get_stoch_rsi(data, period=14):
    rsi = RSI(data['close'], timeperiod=period)
    stoch_k, stoch_d = STOCH(rsi, rsi, rsi,
                             fastk_period=period,
                             slowk_period=3,
                             slowd_period=3)
    # Convert to pandas Series to make it compatible with the rest of the code
    stoch_k = pd.Series(stoch_k)
    stoch_d = pd.Series(stoch_d)
    latest_k = stoch_k.iloc[-1] if not stoch_k.empty else None
    latest_d = stoch_d.iloc[-1] if not stoch_d.empty else None
    print(f"[DEBUG] Stochastic RSI - Period: {period}, Latest K: {latest_k}, Latest D: {latest_d}")
    return stoch_k, stoch_d


def get_atr(data, period=14):
    try:
        atr = pd.Series(ATR(data['high'], data['low'], data['close'], timeperiod=period))
        latest_atr = atr.iloc[-1] if not atr.empty else None
        print(f"[DEBUG] ATR calculation - Period: {period}, Latest ATR: {latest_atr}")
        return atr
    except Exception as e:
        print(f"[ERROR] ATR calculation failed: {str(e)}")
        return pd.Series([None] * len(data))


def calculate_position_size(symbol, risk_percent, stop_loss_pips):
    account_info = mt5.account_info()
    balance = account_info.balance
    risk_amount = balance * (risk_percent / 100)
    point = mt5.symbol_info(symbol).point
    position_size = risk_amount / (stop_loss_pips * point * 10)  # Adjusted for FX
    print(
        f"[DEBUG] Position Size - Balance: {balance}, Risk Amount: {risk_amount}, Stop Loss Pips: {stop_loss_pips}, Size: {round(position_size, 2)}")
    return round(position_size, 2)


def execute_trade(symbol, direction, volume, sl, tp):
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": direction,
        "price": mt5.symbol_info_tick(symbol).ask if direction == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(
            symbol).bid,
        "sl": sl,
        "tp": tp,
        "deviation": 20,
        "magic": 12345,
        "comment": "HybridStrategyBot",
        "type_time": mt5.ORDER_TIME_GTC,
    }
    result = mt5.order_send(request)
    print(
        f"[DEBUG] Trade Execution - Symbol: {symbol}, Direction: {direction}, Volume: {volume}, SL: {sl}, TP: {tp}, Result: {result}")
    return result


def trailing_stop(symbol, ticket, new_sl):
    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "position": ticket,
        "symbol": symbol,
        "sl": new_sl,
        "tp": 0,  # Keep TP unchanged
        "magic": 12345,
    }
    result = mt5.order_send(request)
    print(f"[DEBUG] Trailing Stop Update - Symbol: {symbol}, Ticket: {ticket}, New SL: {new_sl}, Result: {result}")
    return result  # Added return statement


def main():
    open_positions = {}
    while True:
        try:
            # Fetch latest 100 candles
            rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, 100)
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df['ema'] = get_ema(df, EMA_PERIOD)
            df['stoch_k'], df['stoch_d'] = get_stoch_rsi(df, STOCH_RSI_PERIOD)
            df['atr'] = get_atr(df, ATR_PERIOD)

            # Latest candle data
            latest = df.iloc[-1]
            prev = df.iloc[-2]

            # Check for BUY signal (EMA trend + Stochastic RSI oversold)
            if (latest['close'] > latest['ema'] and
                    latest['stoch_k'] < 20 and
                    prev['stoch_k'] < latest['stoch_k']):

                # Calculate ATR-based stop loss
                atr_value = latest['atr']
                if atr_value is not None:
                    sl_price = latest['close'] - (ATR_MULTIPLIER * atr_value)
                    sl_pips = (latest['close'] - sl_price) * 1e4  # For FX pairs

                    # Position sizing
                    size = calculate_position_size(SYMBOL, RISK_PERCENT, sl_pips)

                    # Execute trade
                    result = execute_trade(SYMBOL, mt5.ORDER_TYPE_BUY, size, sl_price, 0)
                    if result.retcode == mt5.TRADE_RETCODE_DONE:
                        open_positions[result.order] = {
                            'sl': sl_price,
                            'highest_price': latest['close']
                        }

            # Update trailing stops for open positions
            for ticket, pos in list(open_positions.items()):
                current_price = mt5.symbol_info_tick(SYMBOL).ask
                if latest['atr'] is not None:
                    new_sl = current_price - (ATR_MULTIPLIER * latest['atr'])

                    # Move SL only if price rises
                    if current_price > pos['highest_price'] and new_sl > pos['sl']:
                        result = trailing_stop(SYMBOL, ticket, new_sl)
                        if result.retcode == mt5.TRADE_RETCODE_DONE:
                            open_positions[ticket]['sl'] = new_sl
                            open_positions[ticket]['highest_price'] = current_price

        except Exception as e:
            print(f"[ERROR] Main loop error: {str(e)}")

        time.sleep(1)  # Check every sec


if __name__ == "__main__":
    main()