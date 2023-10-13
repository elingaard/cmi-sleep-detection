from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm


def rolled_timeseries_features(
    ts_df: pd.DataFrame, roll_freqs: List[str] = ["5min", "15min", "30min", "1H"]
) -> pd.DataFrame:
    """Generate time series features based on rolling windows. In this case the
    mean and standard deviation of each 'roll_freqs' frequency will be calculated"""
    rolled_dfs = []
    for freq in roll_freqs:
        ts_roll_df = ts_df.rolling(freq).agg(
            {"enmo_1min_mean": ["mean", "std"], "anglez_1min_mean": ["mean", "std"]}
        )
        ts_roll_df.columns = [
            f"enmo_{freq}_mean",
            f"enmo_{freq}_std",
            f"anglez_{freq}_mean",
            f"anglez_{freq}_std",
        ]
        rolled_dfs.append(ts_roll_df)
    roll_df = pd.concat(rolled_dfs, axis=1)
    return roll_df


def extract_rolling_features(ts_1min_df: pd.DataFrame) -> pd.DataFrame:
    series_grps = ts_1min_df.groupby(by="series_id")
    feature_dataframes = []
    for series_id, ts_df in tqdm(series_grps):
        ts_df = ts_df.set_index("timestamp")
        roll_feat_df = rolled_timeseries_features(ts_df)
        feat_df = pd.concat([ts_df, roll_feat_df], axis=1)
        feat_df = feat_df.dropna()
        feature_dataframes.append(feat_df)

    full_feat_df = pd.concat(feature_dataframes).reset_index(names="timestamp")
    full_feat_df["step"] = full_feat_df["step"].astype(np.uint32)

    return full_feat_df
