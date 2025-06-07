from functools import wraps, lru_cache
import numpy as np
import pandas as pd
from numba import njit
from pandas import DataFrame
import dask.dataframe as dd
import warnings

# Helper function for empty DataFrame creation
def _empty_df(columns, dtype=np.float32):
    return pd.DataFrame(columns=columns, dtype=dtype)

def inputvalidator(input_="ohlc"):
    def dfcheck(func):
        @wraps(func)
        def wrap(*args, **kwargs):
            args = list(args)
            if not args:
                return func(*args, **kwargs)

            # Find the DataFrame argument
            df_index = 0 if isinstance(args[0], pd.DataFrame) else 1
            if df_index >= len(args):
                return func(*args, **kwargs)

            # Only process pandas DataFrames
            if isinstance(args[df_index], pd.DataFrame):
                # Lowercase column names
                args[df_index] = args[df_index].rename(columns=str.lower)

                # Validate required columns
                inputs = {
                    "o": "open", "h": "high", "l": "low",
                    "c": kwargs.get("column", "close").lower(),
                    "v": "volume"
                }

                missing = [inputs[l] for l in input_ if inputs[l] not in args[df_index].columns]
                if missing:
                    raise LookupError(f'Missing columns: {missing}')

            return func(*args, **kwargs)
        return wrap
    return dfcheck

def apply(decorator):
    def decorate(cls):
        for attr in cls.__dict__:
            if callable(getattr(cls, attr)):
                setattr(cls, attr, decorator(getattr(cls, attr)))
        return cls
    return decorate

@apply(inputvalidator(input_="ohlc"))
class smc:
    __version__ = "0.4.0"
    __author__ = "Arbri"
    ohlc = None

    # Numba-optimized mitigation calculation
    @staticmethod
    @njit(nogil=True)
    def _numba_mitigation(close, fvg, top, bottom):
        mitigated = np.full(len(close), np.nan)
        for i in range(len(fvg)):
            if np.isnan(fvg[i]):
                continue
            start = i + 2
            if start >= len(close):
                continue
            if fvg[i] == 1:
                for j in range(start, len(close)):
                    if close[j] < bottom[i]:
                        mitigated[i] = j
                        break
            else:
                for j in range(start, len(close)):
                    if close[j] > top[i]:
                        mitigated[i] = j
                        break
        return mitigated

    @classmethod
    @lru_cache(maxsize=8)
    def _resampled_data(cls, tf: str):
        return cls.ohlc.resample(tf).agg({'high':'max', 'low':'min'}).shift(1)

    @classmethod
    def fvg(cls, ohlc: pd.DataFrame, join_consecutive: bool = False) -> pd.DataFrame:
        """
        FIXED Fair Value Gap detection with proper type handling
        """
        # Fixed conditions with complete parentheses
        bull_cond = ((ohlc['high'].shift(1) < ohlc['low'].shift(-1)) &
                    (ohlc['close'] > ohlc['open']))
        bear_cond = ((ohlc['low'].shift(1) > ohlc['high'].shift(-1)) &
        (ohlc['close'] < ohlc['open']))

        fvg = np.select([bull_cond, bear_cond], [1, -1], np.nan)

        # Calculate boundaries using vectorized operations
        top = np.where(
            fvg == 1,
            ohlc['low'].shift(-1),
            np.where(fvg == -1, ohlc['low'].shift(1), np.nan)
        )
        bottom = np.where(
            fvg == 1,
            ohlc['high'].shift(1),
            np.where(fvg == -1, ohlc['high'].shift(-1), np.nan)
        )

        if join_consecutive:
            valid_mask = ~pd.isna(fvg)
        streak_groups = (valid_mask.astype(int).diff() != 0).cumsum()

        fvg = pd.Series(fvg).groupby(streak_groups).transform('first')
        top = pd.Series(top).groupby(streak_groups).transform('max')
        bottom = pd.Series(bottom).groupby(streak_groups).transform('min')

        # Use float64 for Numba compatibility
        mitigated = cls._numba_mitigation(
            ohlc['close'].values.astype(np.float64),
            fvg.astype(np.float64),
            top.astype(np.float64),
            bottom.astype(np.float64)
        )

        return pd.DataFrame({
            'FVG': pd.Categorical(fvg),
            'Top': top,
            'Bottom': bottom,
            'MitigatedIndex': pd.Series(mitigated).astype('Int64')
        }, index=ohlc.index)

    @classmethod
    def swing_highs_lows(cls, ohlc: DataFrame, swing_length: int = 50) -> DataFrame:
        # Optimized swing detection with consecutive swing cleaning
        roll_window = swing_length * 2
        roll_high = ohlc['high'].rolling(roll_window, center=True, min_periods=1).max()
        roll_low = ohlc['low'].rolling(roll_window, center=True, min_periods=1).min()

        is_high = (ohlc['high'] == roll_high) & (ohlc['high'] > ohlc['high'].shift(1)) & (ohlc['high'] > ohlc['high'].shift(-1))
        is_low = (ohlc['low'] == roll_low) & (ohlc['low'] < ohlc['low'].shift(1)) & (ohlc['low'] < ohlc['low'].shift(-1))

        combined = np.select([is_high, is_low], [1, -1], np.nan)
        positions = np.where(~np.isnan(combined))[0]

        # Clean consecutive swings
        i = 0
        while i < len(positions)-1:
            current = positions[i]
            next_pos = positions[i+1]

            if combined[current] == combined[next_pos]:
                if combined[current] == 1 and ohlc['high'].iloc[current] < ohlc['high'].iloc[next_pos]:
                    combined[current] = np.nan
                elif combined[current] == -1 and ohlc['low'].iloc[current] > ohlc['low'].iloc[next_pos]:
                    combined[current] = np.nan
            i += 1

        level = np.where(combined == 1, ohlc['high'],
                         np.where(combined == -1, ohlc['low'], np.nan))

        return pd.DataFrame({
            'HighLow': pd.Categorical(combined),
            'Level': level.astype(np.float32)
        }, index=ohlc.index)

    @classmethod
    def bos_choch(cls, ohlc: DataFrame, swing_points: DataFrame, close_break: bool = True) -> DataFrame:
        swings = swing_points.dropna()
        if len(swings) < 3:
            return _empty_df(["BOS", "CHOCH", "Level", "BrokenIndex"])

        bos = np.zeros(len(ohlc), dtype=np.int8)
        choch = np.zeros(len(ohlc), dtype=np.int8)
        level = np.full(len(ohlc), np.nan, dtype=np.float32)
        broken = np.full(len(ohlc), np.nan, dtype=np.float32)

        swing_type = swings.HighLow.values.astype(np.int8)
        swing_level = swings.Level.values.astype(np.float32)
        swing_idx = swings.index.values

        # Enhanced pattern recognition
        for i in range(2, len(swing_type)):
            # BOS detection
            if (swing_type[i-2] == 1 and swing_type[i-1] == -1 and swing_type[i] == 1 and
                    swing_level[i] > swing_level[i-2]):
                bos[swing_idx[i-1]] = 1
                level[swing_idx[i-1]] = swing_level[i]
            elif (swing_type[i-2] == -1 and swing_type[i-1] == 1 and swing_type[i] == -1 and
                  swing_level[i] < swing_level[i-2]):
                bos[swing_idx[i-1]] = -1
                level[swing_idx[i-1]] = swing_level[i]

            # CHOCH detection
            if (swing_type[i-1] == swing_type[i] and
                    ((swing_type[i] == 1 and swing_level[i] < swing_level[i-1]) or
                     (swing_type[i] == -1 and swing_level[i] > swing_level[i-1]))):
                choch[swing_idx[i]] = -1 if swing_type[i] == 1 else 1
                level[swing_idx[i]] = swing_level[i]

        # Mitigation checking
        price = ohlc.close.values if close_break else ohlc.high.values
        for idx in np.where((bos != 0) | (choch != 0))[0]:
            lvl = level[idx]
            direction = bos[idx] if bos[idx] != 0 else choch[idx]
            start = idx + 2

            if direction == 1:
                mask = price[start:] > lvl
            else:
                mask = price[start:] < lvl

            if np.any(mask):
                broken[idx] = start + np.argmax(mask)

        return pd.DataFrame({
            'BOS': pd.Categorical(bos),
            'CHOCH': pd.Categorical(choch),
            'Level': level,
            'BrokenIndex': pd.Series(broken).astype('Int64')
        }).replace(0, np.nan)

    @classmethod
    def ob(cls, ohlc: DataFrame, swing_points: DataFrame, close_mitigation: bool = False) -> DataFrame:
        swh = swing_points[swing_points.HighLow == 1]
        swl = swing_points[swing_points.HighLow == -1]

        swh_level = swh['Level'].reindex(ohlc.index, method='ffill').shift()
        swl_level = swl['Level'].reindex(ohlc.index, method='ffill').shift()

        # Enhanced OB detection with volume metrics
        bull_ob = (
                (swh_level.index.to_series().diff() > 1) &
                (ohlc.close.shift() < ohlc.open.shift()) &
                (ohlc.close > ohlc.open) &
                (ohlc.high > swh_level)
        )
        bear_ob = (
                (swl_level.index.to_series().diff() > 1) &
                (ohlc.close.shift() > ohlc.open.shift()) &
                (ohlc.close < ohlc.open) &
                (ohlc.low < swl_level)
        )

        ob = np.select([bull_ob, bear_ob], [1, -1], np.nan)
        top = np.where(bull_ob, ohlc.high, np.where(bear_ob, ohlc.high.shift(), np.nan))
        bottom = np.where(bull_ob, ohlc.low.shift(), np.where(bear_ob, ohlc.low, np.nan))
        ob_volume = np.where(~np.isnan(ob),
                             ohlc.volume + ohlc.volume.shift(1) + ohlc.volume.shift(2),
                             np.nan)

        # Mitigation tracking
        close = ohlc.close.values.astype(np.float32)
        mitigated = np.full(len(ohlc), np.nan, dtype=np.float32)

        for idx in np.where(~np.isnan(ob))[0]:
            if ob[idx] == 1:
                mask = close[idx+1:] < bottom[idx]
            else:
                mask = close[idx+1:] > top[idx]

            if np.any(mask):
                mitigated[idx] = idx + 1 + np.argmax(mask)

        return pd.DataFrame({
            'OB': pd.Categorical(ob),
            'Top': top.astype(np.float32),
            'Bottom': bottom.astype(np.float32),
            'OBVolume': ob_volume.astype(np.float32),
            'MitigatedIndex': pd.Series(mitigated).astype('Int64')
        })

    @classmethod
    def liquidity(cls, ohlc: DataFrame, swing_points: DataFrame, range_percent: float = 0.01) -> DataFrame:
        valid_swings = swing_points.dropna()
        if len(valid_swings) < 2:
            return _empty_df(["Liquidity", "Level", "Swept"])

        # Cluster detection logic
        swing_levels = valid_swings['Level'].values
        swing_types = valid_swings['HighLow'].values
        pip_range = (ohlc['high'].max() - ohlc['low'].min()) * range_percent

        clusters = []
        current_cluster = []
        for i, (stype, level) in enumerate(zip(swing_types, swing_levels)):
            if not current_cluster:
                current_cluster.append((i, stype, level))
            else:
                if (abs(level - current_cluster[-1][2]) <= pip_range and
                        stype == current_cluster[-1][1]):
                    current_cluster.append((i, stype, level))
                else:
                    if len(current_cluster) >= 2:
                        clusters.append(current_cluster)
                    current_cluster = [(i, stype, level)]

        # Process clusters
        liquidity = np.full(len(ohlc), np.nan)
        levels = np.full(len(ohlc), np.nan)
        swept = np.full(len(ohlc), np.nan)

        for cluster in clusters:
            cluster_indices = [c[0] for c in cluster]
            avg_level = np.mean([c[2] for c in cluster])
            first_idx = valid_swings.index[cluster_indices[0]]

            # Find first sweep after cluster
            if cluster[0][1] == 1:  # Bullish cluster
                sweep_mask = ohlc['high'][first_idx:] > avg_level + pip_range
            else:  # Bearish cluster
                sweep_mask = ohlc['low'][first_idx:] < avg_level - pip_range

            if np.any(sweep_mask):
                sweep_idx = sweep_mask.idxmax()
                liquidity[first_idx] = cluster[0][1]
                levels[first_idx] = avg_level
                swept[first_idx] = sweep_idx

        return pd.DataFrame({
            'Liquidity': liquidity,
            'Level': levels.astype(np.float32),
            'Swept': pd.to_datetime(swept)
        }, index=ohlc.index)

    @classmethod
    def previous_high_low(cls, ohlc: DataFrame, time_frame: str = "1D") -> DataFrame:
        """
        Calculate previous session's high/low levels.

        Parameters:
        ohlc (DataFrame): OHLC price data
        time_frame (str): Resampling period (15m,1H,4H,1D,1W,1M)

        Returns:
        DataFrame: Contains columns:
            - PreviousHigh/Low: Session reference levels
            - BrokenHigh/Low: Cumulative break status

        Example:
        phl = smc.previous_high_low(data, '1D')
        data['close'].plot()
        phl.PreviousHigh.plot(color='red', alpha=0.5)
        """
        valid_tf = ["15m", "1H", "4H", "1D", "1W", "1M"]
        if time_frame not in valid_tf:
            raise ValueError(f"Invalid timeframe. Use one of {valid_tf}")

        resampled = cls._resampled_data(time_frame)
        ph = resampled['high'].reindex(ohlc.index, method='ffill').astype(np.float32)
        pl = resampled['low'].reindex(ohlc.index, method='ffill').astype(np.float32)

        return pd.DataFrame({
            'PreviousHigh': ph,
            'PreviousLow': pl,
            'BrokenHigh': (ohlc.close > ph).astype(float).cummax(),
            'BrokenLow': (ohlc.close < pl).astype(float).cummax()
        })

    @classmethod
    def sessions(cls, ohlc: DataFrame, session: str,
                 start_time: str = "", end_time: str = "",
                 time_zone: str = "UTC") -> DataFrame:
        """
        Analyze trading session activity (Asian/London/NY).

        Parameters:
        ohlc (DataFrame): OHLC price data
        session (str): Sydney, Tokyo, London, New York, Custom
        start_time/end_time (str): Custom session times (HH:MM)
        time_zone (str): Timezone for custom session

        Returns:
        DataFrame: Contains columns:
            - Active: Session active flag
            - High/Low: Session price range

        Example:
        session = smc.sessions(data, 'New York')
        data['close'].plot()
        session[session.Active].High.plot.area(alpha=0.1)
        """
        session_map = {
            "Sydney": ("21:00", "06:00", "Australia/Sydney"),
            "Tokyo": ("00:00", "09:00", "Asia/Tokyo"),
            "London": ("07:00", "16:00", "Europe/London"),
            "New York": ("13:00", "22:00", "America/New_York"),
            "Custom": (start_time, end_time, time_zone)
        }


        start, end, tz = session_map[session]
        idx = ohlc.index.tz_convert(tz)
        start_time = pd.to_datetime(start).time()
        end_time = pd.to_datetime(end).time()

        if session == "Sydney":
            mask = ((idx.time >= start_time) | (idx.time <= end_time))
        else:
            mask = idx.indexer_between_time(start_time, end_time)

        # # Add timezone localization safety (line 423)
        # if ohlc.index.tz is None:
        #     idx = ohlc.index.tz_localize('UTC').tz_convert(tz)
        # else:
        #     idx = ohlc.index.tz_convert(tz)

        session_changes = mask.ne(mask.shift()).cumsum()
        high = ohlc['high'].groupby(session_changes).cummax().astype(np.float32)
        low = ohlc['low'].groupby(session_changes).cummin().astype(np.float32)

        return pd.DataFrame({
            'Active': mask.astype('category'),
            'High': high,
            'Low': low
        })

    @classmethod
    def retracements(cls, ohlc: DataFrame, swing_points: DataFrame) -> DataFrame:
        """
        Calculate price retracement percentages.

        Parameters:
        ohlc (DataFrame): OHLC price data
        swing_points (DataFrame): From swing_highs_lows()

        Returns:
        DataFrame: Contains columns:
            - Direction: 1=uptrend, -1=downtrend
            - CurrentRetracement%: Current pullback level
            - DeepestRetracement%: Maximum pullback level

        Example:
        ret = smc.retracements(data, swings)
        ret['CurrentRetracement%'].plot()
        plt.yticks([23.6, 38.2, 50, 61.8, 78.6])
        """
        valid_swings = swing_points.dropna()
        if len(valid_swings) < 2:
            return _empty_df(["Direction", "CurrentRetracement%", "DeepestRetracement%"])

        direction = valid_swings['HighLow'].reindex(ohlc.index).ffill().astype('category')
        swing_levels = valid_swings['Level'].reindex(ohlc.index).ffill().astype(np.float32)

        ranges = np.where(
            direction == 1,
            swing_levels - ohlc.low.astype(np.float32),
            ohlc.high.astype(np.float32) - swing_levels
        )

        total_ranges = valid_swings['Level'].diff().abs().reindex(ohlc.index).ffill().astype(np.float32)
        current_ret = (ranges / np.where(total_ranges == 0, 1, total_ranges)) * 100
        current_ret = np.nan_to_num(current_ret, nan=0.0)

        deepest_ret = current_ret.copy()
        for group in direction.ne(direction.shift()).cumsum().unique():
            mask = direction.ne(direction.shift()).cumsum() == group
            deepest_ret[mask] = np.maximum.accumulate(current_ret[mask])

        return pd.DataFrame({
            'Direction': direction,
            'CurrentRetracement%': current_ret.round(1).astype(np.float32),
            'DeepestRetracement%': deepest_ret.round(1).astype(np.float32)
        })

    @classmethod
    def dask_swing_highs_lows(cls, ddf: dd.DataFrame, swing_length: int) -> dd.DataFrame:
        """
       Distributed swing point detection for large datasets.

       Parameters:
       ddf (dask.DataFrame): Distributed OHLC data
       swing_length (int): Same as swing_highs_lows()

       Returns:
       dask.DataFrame: Same schema as swing_highs_lows()

       Example:
        dask_df = dd.from_pandas(data, npartitions=4)
        swings = smc.dask_swing_highs_lows(dask_df, 5).compute()
       """
        meta = pd.DataFrame({
            'HighLow': pd.Series([], dtype='category'),
            'Level': pd.Series([], dtype=np.float32)
        })
        return ddf.map_partitions(
            cls.swing_highs_lows,
            swing_length=swing_length,
            meta=meta
        )

    @classmethod
    def dask_retracements(cls, ddf: dd.DataFrame, swing_points: dd.DataFrame) -> dd.DataFrame:
        """
       Distributed retracement calculation for large datasets.

       Parameters:
       ddf (dask.DataFrame): Distributed OHLC data
       swing_points (dask.DataFrame): From dask_swing_highs_lows()

       Returns:
       dask.DataFrame: Same schema as retracements()

       Example:
       dask_swings = smc.dask_swing_highs_lows(dask_df, 5)
       ret = smc.dask_retracements(dask_df, dask_swings).compute()
       """
        meta = _empty_df(["Direction", "CurrentRetracement%", "DeepestRetracement%"], dtype=np.float32)
        return ddf.map_partitions(
            cls.retracements,
            swing_points=swing_points,
            meta=meta
        )

    @classmethod
    def _empty_df(cls, columns):
        return pd.DataFrame(columns=columns, index=pd.RangeIndex(0))