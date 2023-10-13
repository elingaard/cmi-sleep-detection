import datetime as dt

import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter1d
from tqdm import tqdm


def preprocess_timeseries(ts_df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess timeseries data to correct timestamp format and more memory-efficient
    data types"""
    ts_df["timestamp"] = ts_df["timestamp"].str[:-5]  # remove tzinfo
    ts_df["timestamp"] = pd.to_datetime(ts_df["timestamp"], format="ISO8601")
    ts_df["hour"] = ts_df["timestamp"].dt.hour.astype(np.uint8)
    ts_df["weekday"] = ts_df["timestamp"].dt.weekday.astype(np.uint8)
    return ts_df


def preprocess_events(events_df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess events data to correct timestamp format and more memory-efficient
    data types"""
    events_df = events_df.dropna().reset_index(drop=True)
    events_df["timestamp"] = events_df["timestamp"].str[:-5]  # remove tzinfo
    events_df["timestamp"] = pd.to_datetime(events_df["timestamp"], format="ISO8601")
    events_df["step"] = events_df["step"].astype(int)
    return events_df


def single_series_1min_resampling(ts_df: pd.DataFrame) -> pd.DataFrame:
    """Resample a single raw time series (5sec) data to 1min intervals"""
    ts_5sec_df = ts_df.copy()
    ts_5sec_df = ts_5sec_df.set_index("timestamp")
    ts_1min_df = ts_5sec_df.resample("1min").agg(
        step=("step", "min"),
        anglez_1min_mean=("anglez", "mean"),
        anglez_1min_std=("anglez", "std"),
        enmo_1min_mean=("enmo", "mean"),
        enmo_1min_std=("enmo", "std"),
    )
    ts_1min_df = ts_1min_df.dropna().reset_index()
    ts_1min_df["step"] = ts_1min_df["step"].astype(np.uint32)
    ts_1min_df["series_id"] = ts_5sec_df.series_id.iloc[0]
    ts_1min_df["hour"] = ts_1min_df["timestamp"].dt.hour.astype(np.uint8)
    ts_1min_df["weekday"] = ts_1min_df["timestamp"].dt.weekday.astype(np.uint8)
    return ts_1min_df


def resample_timeseries_data_to_1min(series_df: pd.DataFrame) -> pd.DataFrame:
    """Iterate over each series in the data and resample to 1min interval"""
    series_grps = series_df.groupby(by="series_id")
    one_min_dfs = []
    for series_id, ts_df in tqdm(series_grps):
        ts_5sec_df = ts_df.copy()
        ts_5sec_df = preprocess_timeseries(ts_5sec_df)
        ts_1min_df = single_series_1min_resampling(ts_5sec_df)
        one_min_dfs.append(ts_1min_df)

    full_1min_df = pd.concat(one_min_dfs).reset_index(drop=True)

    return full_1min_df


def create_asleep_column(ts_df: pd.DataFrame, events_df: pd.DataFrame) -> pd.DataFrame:
    """Create a new binary column in the timeseries data indicating whether the person
    is asleep at the specified timestamp or not. This column can be derived from the
    onset and wakeup events in the events dataframe"""
    ts_df = ts_df.copy()
    ts_df["asleep"] = 0
    ts_df = ts_df.reset_index(drop=True)
    for night_id, night_df in events_df.groupby(by="night"):
        if len(night_df) != 2:
            continue
        onset_step = night_df[night_df["event"] == "onset"].step.item()
        wakeup_step = night_df[night_df["event"] == "wakeup"].step.item()
        step_list = ts_df["step"].tolist()
        onset_1min_idx = step_list.index(onset_step)
        wake_1min_idx = step_list.index(wakeup_step)
        ts_df.loc[onset_1min_idx:wake_1min_idx, "asleep"] = 1
    ts_df["asleep"] = ts_df["asleep"].astype(np.uint8)
    return ts_df


def event_minmax_prune_timeseries(
    ts_df: pd.DataFrame, events_df: pd.DataFrame, hour_buffer: int
) -> pd.DataFrame:
    """Prune timeseries dataframe to only include timestamps in the min and max
    intervals of the event dataframe (some timeseries have long stretches of unannotated
    data)"""
    time_buffer = dt.timedelta(hours=hour_buffer)
    min_event_dt = events_df["timestamp"].min() - time_buffer
    max_event_dt = events_df["timestamp"].max() + time_buffer
    prune_idx = (ts_df["timestamp"] >= min_event_dt) & (
        ts_df["timestamp"] <= max_event_dt
    )
    ts_df = ts_df[prune_idx].reset_index(drop=True)
    return ts_df


def remove_no_sleep_periods(ts_df: pd.DataFrame, min_hours: int = 24) -> pd.DataFrame:
    """Remove any sequences in the time series where no sleep has been detected for
    'min_hours'. This most likely indicates that data was not annotated in this
     period."""
    no_sleep_24h = (1 - maximum_filter1d(ts_df["asleep"], size=60 * min_hours)).astype(
        bool
    )
    ts_df = ts_df[~no_sleep_24h]
    return ts_df


def add_asleep_target_to_timeseries_data(
    series_df: pd.DataFrame, events_df: pd.DataFrame
):
    """For each series in the data add a column which indicates whether the person is
    asleep or not. This is based on the onset and wakeup times in the events data."""
    series_grps = series_df.groupby(by="series_id")
    event_grps = events_df.groupby(by="series_id")
    series_grp_keys = set(series_grps.groups.keys())
    event_grp_keys = set(event_grps.groups.keys())
    # remove keys that are not present in both events and series data
    grp_keys = series_grp_keys.intersection(event_grp_keys)

    ts_asleep_dataframes = []
    for series_id in tqdm(grp_keys):
        event_df = event_grps.get_group(series_id)
        ts_df = series_grps.get_group(series_id)
        ts_df = create_asleep_column(ts_df, event_df)
        ts_df = event_minmax_prune_timeseries(ts_df, event_df, hour_buffer=3)
        ts_df = remove_no_sleep_periods(ts_df)
        ts_asleep_dataframes.append(ts_df)
    ts_asleep_df = pd.concat(ts_asleep_dataframes).reset_index(drop=True)
    return ts_asleep_df
