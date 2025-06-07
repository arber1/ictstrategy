import pandas as pd
import plotly.graph_objects as go
import sys
import os
from binance.client import Client
import MetaTrader5 as mt5
from datetime import datetime
import numpy as np
import time
import imageio
from io import BytesIO
from PIL import Image
import plotly.io as pio
from io import BytesIO
from PIL import Image
import numpy as np

sys.path.append(os.path.abspath("../"))
from smartmoneyconcepts.smc import smc

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

def add_FVG(fig, df, fvg_data):
    for i in range(len(fvg_data["FVG"])):
        if not np.isnan(fvg_data["FVG"][i]):
            x1 = int(
                fvg_data["MitigatedIndex"][i]
                if fvg_data["MitigatedIndex"][i] != 0
                else len(df) - 1
            )
            fig.add_shape(
                # filled Rectangle
                type="rect",
                x0=df.index[i],
                y0=fvg_data["Top"][i],
                x1=df.index[x1],
                y1=fvg_data["Bottom"][i],
                line=dict(
                    width=0,
                ),
                fillcolor="yellow",
                opacity=0.2,
            )
            mid_x = round((i + x1) / 2)
            mid_y = (fvg_data["Top"][i] + fvg_data["Bottom"][i]) / 2
            fig.add_trace(
                go.Scatter(
                    x=[df.index[mid_x]],
                    y=[mid_y],
                    mode="text",
                    text="FVG",
                    textposition="middle center",
                    textfont=dict(color='rgba(255, 255, 255, 0.4)', size=8),
                )
            )
    return fig


def add_swing_highs_lows(fig, df, swing_highs_lows_data):
    indexs = []
    level = []
    for i in range(len(swing_highs_lows_data)):
        if not np.isnan(swing_highs_lows_data["HighLow"][i]):
            indexs.append(i)
            level.append(swing_highs_lows_data["Level"][i])

    # plot these lines on a graph
    for i in range(len(indexs) - 1):
        fig.add_trace(
            go.Scatter(
                x=[df.index[indexs[i]], df.index[indexs[i + 1]]],
                y=[level[i], level[i + 1]],
                mode="lines",
                line=dict(
                    color=(
                        "rgba(0, 128, 0, 0.2)"
                        if swing_highs_lows_data["HighLow"][indexs[i]] == -1
                        else "rgba(255, 0, 0, 0.2)"
                    ),
                ),
            )
        )

    return fig


def add_bos_choch(fig, df, bos_choch_data):
    for i in range(len(bos_choch_data["BOS"])):
        if not np.isnan(bos_choch_data["BOS"][i]):
            # add a label to this line
            mid_x = round((i + int(bos_choch_data["BrokenIndex"][i])) / 2)
            mid_y = bos_choch_data["Level"][i]
            fig.add_trace(
                go.Scatter(
                    x=[df.index[i], df.index[int(bos_choch_data["BrokenIndex"][i])]],
                    y=[bos_choch_data["Level"][i], bos_choch_data["Level"][i]],
                    mode="lines",
                    line=dict(
                        color="rgba(255, 165, 0, 0.2)",
                    ),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[df.index[mid_x]],
                    y=[mid_y],
                    mode="text",
                    text="BOS",
                    textposition="top center" if bos_choch_data["BOS"][i] == 1 else "bottom center",
                    textfont=dict(color="rgba(255, 165, 0, 0.4)", size=8),
                )
            )
        if not np.isnan(bos_choch_data["CHOCH"][i]):
            # add a label to this line
            mid_x = round((i + int(bos_choch_data["BrokenIndex"][i])) / 2)
            mid_y = bos_choch_data["Level"][i]
            fig.add_trace(
                go.Scatter(
                    x=[df.index[i], df.index[int(bos_choch_data["BrokenIndex"][i])]],
                    y=[bos_choch_data["Level"][i], bos_choch_data["Level"][i]],
                    mode="lines",
                    line=dict(
                        color="rgba(0, 0, 255, 0.2)",
                    ),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[df.index[mid_x]],
                    y=[mid_y],
                    mode="text",
                    text="CHOCH",
                    textposition="top center" if bos_choch_data["CHOCH"][i] == 1 else "bottom center",
                    textfont=dict(color="rgba(0, 0, 255, 0.4)", size=8),
                )
            )

    return fig


def add_OB(fig, df, ob_data):
    def format_volume(volume):
        if volume >= 1e12:
            return f"{volume / 1e12:.3f}T"
        elif volume >= 1e9:
            return f"{volume / 1e9:.3f}B"
        elif volume >= 1e6:
            return f"{volume / 1e6:.3f}M"
        elif volume >= 1e3:
            return f"{volume / 1e3:.3f}k"
        else:
            return f"{volume:.2f}"

    for i in range(len(ob_data["OB"])):
        if ob_data["OB"][i] == 1:
            x1 = int(
                ob_data["MitigatedIndex"][i]
                if ob_data["MitigatedIndex"][i] != 0
                else len(df) - 1
            )
            fig.add_shape(
                type="rect",
                x0=df.index[i],
                y0=ob_data["Bottom"][i],
                x1=df.index[x1],
                y1=ob_data["Top"][i],
                line=dict(color="Purple"),
                fillcolor="Purple",
                opacity=0.2,
                name="Bullish OB",
                legendgroup="bullish ob",
                showlegend=True,
            )

            if ob_data["MitigatedIndex"][i] > 0:
                x_center = df.index[int(i + (ob_data["MitigatedIndex"][i] - i) / 2)]
            else:
                x_center = df.index[int(i + (len(df) - i) / 2)]

            y_center = (ob_data["Bottom"][i] + ob_data["Top"][i]) / 2
            volume_text = format_volume(ob_data["OBVolume"][i])
            # Add annotation text
            annotation_text = f'OB: {volume_text} ({ob_data["Percentage"][i]}%)'

            fig.add_annotation(
                x=x_center,
                y=y_center,
                xref="x",
                yref="y",
                align="center",
                text=annotation_text,
                font=dict(color="rgba(255, 255, 255, 0.4)", size=8),
                showarrow=False,
            )

    for i in range(len(ob_data["OB"])):
        if ob_data["OB"][i] == -1:
            x1 = int(
                ob_data["MitigatedIndex"][i]
                if ob_data["MitigatedIndex"][i] != 0
                else len(df) - 1
            )
            fig.add_shape(
                type="rect",
                x0=df.index[i],
                y0=ob_data["Bottom"][i],
                x1=df.index[x1],
                y1=ob_data["Top"][i],
                line=dict(color="Purple"),
                fillcolor="Purple",
                opacity=0.2,
                name="Bearish OB",
                legendgroup="bearish ob",
                showlegend=True,
            )

            if ob_data["MitigatedIndex"][i] > 0:
                x_center = df.index[int(i + (ob_data["MitigatedIndex"][i] - i) / 2)]
            else:
                x_center = df.index[int(i + (len(df) - i) / 2)]

            y_center = (ob_data["Bottom"][i] + ob_data["Top"][i]) / 2
            volume_text = format_volume(ob_data["OBVolume"][i])
            # Add annotation text
            annotation_text = f'OB: {volume_text} ({ob_data["Percentage"][i]}%)'

            fig.add_annotation(
                x=x_center,
                y=y_center,
                xref="x",
                yref="y",
                align="center",
                text=annotation_text,
                font=dict(color="rgba(255, 255, 255, 0.4)", size=8),
                showarrow=False,
            )
    return fig


def add_liquidity(fig, df, liquidity_data):
    # draw a line horizontally for each liquidity level
    for i in range(len(liquidity_data["Liquidity"])):
        if not np.isnan(liquidity_data["Liquidity"][i]):
            fig.add_trace(
                go.Scatter(
                    x=[df.index[i], df.index[int(liquidity_data["End"][i])]],
                    y=[liquidity_data["Level"][i], liquidity_data["Level"][i]],
                    mode="lines",
                    line=dict(
                        color="rgba(255, 165, 0, 0.2)",
                    ),
                )
            )
            mid_x = round((i + int(liquidity_data["End"][i])) / 2)
            fig.add_trace(
                go.Scatter(
                    x=[df.index[mid_x]],
                    y=[liquidity_data["Level"][i]],
                    mode="text",
                    text="Liquidity",
                    textposition="top center" if liquidity_data["Liquidity"][i] == 1 else "bottom center",
                    textfont=dict(color="rgba(255, 165, 0, 0.4)", size=8),
                )
            )
        if liquidity_data["Swept"][i] != 0 and not np.isnan(liquidity_data["Swept"][i]):
            # draw a red line between the end and the swept point
            fig.add_trace(
                go.Scatter(
                    x=[
                        df.index[int(liquidity_data["End"][i])],
                        df.index[int(liquidity_data["Swept"][i])],
                    ],
                    y=[
                        liquidity_data["Level"][i],
                        (
                            df["high"].iloc[int(liquidity_data["Swept"][i])]
                            if liquidity_data["Liquidity"][i] == 1
                            else df["low"].iloc[int(liquidity_data["Swept"][i])]
                        ),
                    ],
                    mode="lines",
                    line=dict(
                        color="rgba(255, 0, 0, 0.2)",
                    ),
                )
            )
            mid_x = round((i + int(liquidity_data["Swept"][i])) / 2)
            mid_y = (
                            liquidity_data["Level"][i]
                            + (
                                df["high"].iloc[int(liquidity_data["Swept"][i])]
                                if liquidity_data["Liquidity"][i] == 1
                                else df["low"].iloc[int(liquidity_data["Swept"][i])]
                            )
                    ) / 2
            fig.add_trace(
                go.Scatter(
                    x=[df.index[mid_x]],
                    y=[mid_y],
                    mode="text",
                    text="Liquidity Swept",
                    textposition="top center" if liquidity_data["Liquidity"][i] == 1 else "bottom center",
                    textfont=dict(color="rgba(255, 0, 0, 0.4)", size=8),
                )
            )
    return fig


def add_previous_high_low(fig, df, previous_high_low_data):
    high = previous_high_low_data["PreviousHigh"]
    low = previous_high_low_data["PreviousLow"]

    # create a list of all the different high levels and their indexes
    high_levels = []
    high_indexes = []
    for i in range(len(high)):
        if not np.isnan(high[i]) and high[i] != (high_levels[-1] if len(high_levels) > 0 else None):
            high_levels.append(high[i])
            high_indexes.append(i)

    low_levels = []
    low_indexes = []
    for i in range(len(low)):
        if not np.isnan(low[i]) and low[i] != (low_levels[-1] if len(low_levels) > 0 else None):
            low_levels.append(low[i])
            low_indexes.append(i)

    # plot these lines on a graph
    for i in range(len(high_indexes)-1):
        fig.add_trace(
            go.Scatter(
                x=[df.index[high_indexes[i]], df.index[high_indexes[i+1]]],
                y=[high_levels[i], high_levels[i]],
                mode="lines",
                line=dict(
                    color="rgba(255, 255, 255, 0.2)",
                ),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[df.index[high_indexes[i+1]]],
                y=[high_levels[i]],
                mode="text",
                text="PH",
                textposition="top center",
                textfont=dict(color="rgba(255, 255, 255, 0.4)", size=8),
            )
        )

    for i in range(len(low_indexes)-1):
        fig.add_trace(
            go.Scatter(
                x=[df.index[low_indexes[i]], df.index[low_indexes[i+1]]],
                y=[low_levels[i], low_levels[i]],
                mode="lines",
                line=dict(
                    color="rgba(255, 255, 255, 0.2)",
                ),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[df.index[low_indexes[i+1]]],
                y=[low_levels[i]],
                mode="text",
                text="PL",
                textposition="bottom center",
                textfont=dict(color="rgba(255, 255, 255, 0.4)", size=8),
            )
        )

    return fig


def add_sessions(fig, df, sessions):
    for i in range(len(sessions["Active"])-1):
        if sessions["Active"][i] == 1:
            fig.add_shape(
                type="rect",
                x0=df.index[i],
                y0=sessions["Low"][i],
                x1=df.index[i + 1],
                y1=sessions["High"][i],
                line=dict(
                    width=0,
                ),
                fillcolor="#16866E",
                opacity=0.2,
            )
    return fig


def add_retracements(fig, df, retracements):
    for i in range(len(retracements)):
        if (
                (
                        (
                                retracements["Direction"].iloc[i + 1]
                                if i < len(retracements) - 1
                                else 0
                        )
                        != retracements["Direction"].iloc[i]
                        or i == len(retracements) - 1
                )
                and retracements["Direction"].iloc[i] != 0
                and (
                retracements["Direction"].iloc[i + 1]
                if i < len(retracements) - 1
                else retracements["Direction"].iloc[i]
        )
                != 0
        ):
            fig.add_annotation(
                x=df.index[i],
                y=(
                    df["high"].iloc[i]
                    if retracements["Direction"].iloc[i] == -1
                    else df["low"].iloc[i]
                ),
                xref="x",
                yref="y",
                text=f"C:{retracements['CurrentRetracement%'].iloc[i]}%<br>D:{retracements['DeepestRetracement%'].iloc[i]}%",
                font=dict(color="rgba(255, 255, 255, 0.4)", size=8),
                showarrow=False,
            )
    return fig


# # get the data
# def import_data(symbol, start_str, timeframe):
#     client = Client()
#     start_str = str(start_str)
#     end_str = f"{datetime.now()}"
#     df = pd.DataFrame(
#         client.get_historical_klines(
#             symbol=symbol, interval=timeframe, start_str=start_str, end_str=end_str
#         )
#     ).astype(float)
#     df = df.iloc[:, :6]
#     df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
#     df = df.set_index("timestamp")
#     df.index = pd.to_datetime(df.index, unit="ms").strftime("%Y-%m-%d %H:%M:%S")
#     return df
#
#
# df = import_data("XRPUSDT", "2024-03-01", "5m")
# df = df.iloc[-500:]
#
# def fig_to_buffer(fig):
#     fig_bytes = fig.to_image(format="png")
#     fig_buffer = BytesIO(fig_bytes)
#     fig_image = Image.open(fig_buffer)
#     return np.array(fig_image)
#
#
# gif = []
#
# window = 100
# for pos in range(window, len(df)):
#     window_df = df.iloc[pos - window : pos]
#
#     fig = go.Figure(
#         data=[
#             go.Candlestick(
#                 x=window_df.index,
#                 open=window_df["open"],
#                 high=window_df["high"],
#                 low=window_df["low"],
#                 close=window_df["close"],
#                 increasing_line_color="#77dd76",
#                 decreasing_line_color="#ff6962",
#             )
#         ]
#     )
#
#     fvg_data = smc.fvg(window_df, join_consecutive=True)
#     swing_highs_lows_data = smc.swing_highs_lows(window_df, swing_length=5)
#     bos_choch_data = smc.bos_choch(window_df, swing_highs_lows_data)
#     ob_data = smc.ob(window_df, swing_highs_lows_data)
#     liquidity_data = smc.liquidity(window_df, swing_highs_lows_data)
#     previous_high_low_data = smc.previous_high_low(window_df, time_frame="4h")
#     sessions = smc.sessions(window_df, session="London")
#     retracements = smc.retracements(window_df, swing_highs_lows_data)
#     fig = add_FVG(fig, window_df, fvg_data)
#     fig = add_swing_highs_lows(fig, window_df, swing_highs_lows_data)
#     fig = add_bos_choch(fig, window_df, bos_choch_data)
#     fig = add_OB(fig, window_df, ob_data)
#     fig = add_liquidity(fig, window_df, liquidity_data)
#     fig = add_previous_high_low(fig, window_df, previous_high_low_data)
#     fig = add_sessions(fig, window_df, sessions)
#     fig = add_retracements(fig, window_df, retracements)
#
#     fig.update_layout(xaxis_rangeslider_visible=False)
#     fig.update_layout(showlegend=False)
#     fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
#     fig.update_xaxes(visible=False, showticklabels=False)
#     fig.update_yaxes(visible=False, showticklabels=False)
#     fig.update_layout(plot_bgcolor="rgba(0,0,0,0)")
#     fig.update_layout(paper_bgcolor="rgba(12, 14, 18, 1)")
#     fig.update_layout(font=dict(color="white"))
#
#     # reduce the size of the image
#     fig.update_layout(width=500, height=300)
#
#     gif.append(fig_to_buffer(fig))
#
# # save the gif
# imageio.mimsave("xrpusd.gif", gif, duration=1)
#
#
#
#
#


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
start_date = "2025-03-07"  # Use a smaller date range
timeframe = mt5.TIMEFRAME_M5  # 5-minute timeframe (you can change to M1 or M15)
df = import_data(symbol, start_date, timeframe)

# Trim the data to the last 500 rows
df = df.iloc[-500:]

# Debug: Show retrieved data
print("✅ Retrieved Data:")
print(df.head())

# Shutdown MT5
mt5.shutdown()

# Function to convert figure to buffer (PNG)
def fig_to_buffer(fig):
    fig_bytes = fig.to_image(format="png")
    fig_buffer = BytesIO(fig_bytes)
    fig_image = Image.open(fig_buffer)
    return np.array(fig_image)

# Set up GIF parameters
gif = []
window = 100  # Size of the window for candlestick charting
frame_dir = "frames"
os.makedirs(frame_dir, exist_ok=True)  # Create a folder to store frames
resolution = (500, 300)  # Resolution of the generated frames
fps = 1  # 1 second per frame

# Debug: Track the number of frames generated
frames_generated = 0
total_frames = len(df) - window

# Generate frames for GIF
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