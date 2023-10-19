from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm


def iqr90(x):
    return np.quantile(x, q=0.95) - np.quantile(x, q=0.05)


def rolled_timeseries_features(
    ts_df: pd.DataFrame,
    agg_dict: dict,
    roll_freqs: List[str],
) -> pd.DataFrame:
    """Generate time series features based on rolling windows. In this case the
    mean and standard deviation of each 'roll_freqs' frequency will be calculated"""
    rolled_dfs = []
    for freq in roll_freqs:
        ts_roll_df = ts_df.rolling(freq).agg(agg_dict)
        feat_cols = [
            "_".join(name_parts) + f"_{freq}"
            for name_parts in ts_roll_df.columns.to_flat_index()
        ]
        ts_roll_df.columns = feat_cols
        ts_roll_df[feat_cols] = ts_roll_df[feat_cols].astype(np.float32)
        rolled_dfs.append(ts_roll_df)
    roll_df = pd.concat(rolled_dfs, axis=1)
    return roll_df


def extract_rolling_features(
    ts_1min_df: pd.DataFrame, agg_dict, roll_freqs
) -> pd.DataFrame:
    series_grps = ts_1min_df.groupby(by="series_id")
    feature_dataframes = []
    for series_id, ts_df in tqdm(series_grps):
        ts_df = ts_df.set_index("timestamp")
        roll_feat_df = rolled_timeseries_features(ts_df, agg_dict, roll_freqs)
        feat_df = pd.concat([ts_df, roll_feat_df], axis=1)
        feat_df = feat_df.dropna()
        feature_dataframes.append(feat_df)

    full_feat_df = pd.concat(feature_dataframes).reset_index(names="timestamp")
    full_feat_df["step"] = full_feat_df["step"].astype(np.uint32)

    return full_feat_df
