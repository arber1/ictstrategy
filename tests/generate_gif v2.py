import MetaTrader5 as mt5
import pandas as pd
import time
import imageio
import plotly.graph_objects as go
import plotly.io as pio
from io import BytesIO
from PIL import Image
import numpy as np

# Initialize MT5 connection
if not mt5.initialize():
    print("❌ Failed to connect to MT5. Is MT5 running?")
    exit()

# ✅ Get account info
account_info = mt5.account_info()
if account_info:
    print(f"✅ Connected to MT5 - Account ID: {account_info.login}")
else:
    print("❌ Failed to retrieve account info. Are you logged into a broker?")
    mt5.shutdown()
    exit()


# Define a function to retrieve data from MT5
def import_data(symbol, start_str, timeframe):

    # Convert start date to UTC
    utc_from = pd.to_datetime(start_str)
    utc_to = pd.to_datetime("now")

    print(f"📅 Retrieving data from {utc_from} to {utc_to}...")

    # 🔄 Get market data
    symbol = "XRPUSD"  # Change as needed
    timeframe = mt5.TIMEFRAME_M5  # 5-minute timeframe
    bars = 500  # Number of candles

    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)

    print(rates)


    if rates is None:
        error_code = mt5.last_error()
        print(f"❌ Failed to retrieve data for {symbol}. Error code: {error_code[0]}, {error_code[1]}")
        mt5.shutdown()
        exit()

    # Convert to DataFrame
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)

    print(f"✅ Successfully retrieved {len(df)} data points for {symbol}.")
    return df

# Test with a symbol from available symbols list and smaller date range
symbol = "XRPUSD"  # Change this if needed, ensure the exact symbol name (e.g., XRPUSD. or XRPUSD=X)
start_date = "2025-03-09"  # Use a smaller date range
timeframe = mt5.TIMEFRAME_M5  # 5-minute timeframe (you can change to M1 or M15)
df = import_data(symbol, start_date, timeframe)

# Trim the data to the last 500 rows
df = df.iloc[-500:]

# Debug: Show retrieved data
print("✅ Retrieved Data:")
print(df.head())

# Frame generation for the GIF
gif = []
window = 100
resolution = (800, 700)
fps = 15  # Adjust FPS as needed

# Ensure we're getting frames
print("📸 Generating frames...")

for pos in range(window, len(df)):
    print(f"🚀 Generating frame {pos - window + 1}/{len(df) - window}")

    window_df = df.iloc[pos - window: pos]

    fig = go.Figure(data=[
        go.Candlestick(
            x=window_df.index,
            open=window_df["open"],
            high=window_df["high"],
            low=window_df["low"],
            close=window_df["close"],
            increasing_line_color="#77dd76",
            decreasing_line_color="#ff6962",
        )
    ])

    fig.update_layout(
        width=resolution[0], height=resolution[1],
        xaxis_rangeslider_visible=False, showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(12, 14, 18, 1)",
        font=dict(color="white"),
        xaxis=dict(visible=False), yaxis=dict(visible=False)
    )

    try:
        img_buffer = pio.to_image(fig, format="png", width=resolution[0], height=resolution[1])
        img = Image.open(BytesIO(img_buffer))
        gif.append(np.array(img))  # Add the frame to the gif list

        # Debug: Track progress
        print(f"✅ Frame {pos - window + 1} generated.")
    except Exception as e:
        print(f"❌ Error while generating frame {pos - window + 1}: {str(e)}")

# Check if we have any frames
if not gif:
    print("❌ No frames were generated.")
else:
    # Save GIF
    print(f"📦 Saving GIF with {len(gif)} frames...")
    imageio.mimsave("xrpusd.gif", gif, duration=1 / fps)
    print("✅ GIF saved successfully as 'xrpusd.gif'.")

# Shutdown MT5
mt5.shutdown()
