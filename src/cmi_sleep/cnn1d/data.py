from typing import Tuple

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import lightning.pytorch as pl


def time_encoding(ts_df: pd.DataFrame):
    """Use most common awake and onset times as a prior for encoding the timestamp to
    a numerical value. Encoding is done at minute resolution"""
    n_mins_day = 60 * 24
    awake_prior_vals = np.sin(np.linspace(0, np.pi, n_mins_day) + 0.208 * np.pi) ** 24
    awake_prior_dict = dict(zip(range(1440), awake_prior_vals))
    onset_prior_vals = np.sin(np.linspace(0, np.pi, n_mins_day) + 0.555 * np.pi) ** 24
    onset_prior_dict = dict(zip(range(1440), onset_prior_vals))
    time_df = pd.DataFrame()
    time_df["onset_prior"] = (
        (ts_df.timestamp.dt.hour * 60 + ts_df.timestamp.dt.minute)
        .map(onset_prior_dict)
        .astype(np.float32)
    )
    time_df["awake_prior"] = (
        (ts_df.timestamp.dt.hour * 60 + ts_df.timestamp.dt.minute)
        .map(awake_prior_dict)
        .astype(np.float32)
    )
    return time_df


class CMITimeSeriesSampler(Dataset):
    """Dataset class for sampling time-series from the CMI Sleep Detection dataset. 
    A random time-series of size 'sample_size' is sampled from the specified series
    index"""

    def __init__(
        self, series_df, sample_size: int, feat_cols: list, target_col: str
    ) -> None:
        super().__init__()
        self.feat_cols = feat_cols
        self.target_col = target_col
        self.series_df = series_df
        self.series_grps = series_df.groupby(by="series_id")
        self.series_ids = list(self.series_grps.groups.keys())
        self.sample_size = sample_size

    def check_timeseries_continuity(self, ts_df):
        step_sizes = ts_df["step"].diff()[1:].astype(int)
        is_cont = (step_sizes == step_sizes.iloc[0]).all()
        return is_cont

    def __len__(self):
        return len(self.series_ids)

    def __getitem__(self, index):
        sid = self.series_ids[index]
        sample_df = self.series_grps.get_group(sid)
        sample_df = sample_df.reset_index(drop=True)
        assert len(sample_df) > self.sample_size

        is_cont_ts = False
        while ~is_cont_ts:
            start_idx = np.random.randint(0, len(sample_df) - self.sample_size)
            end_idx = start_idx + self.sample_size
            ts_df = sample_df[start_idx:end_idx]
            is_cont_ts = self.check_timeseries_continuity(ts_df)

        X_data = ts_df[self.feat_cols].T.to_numpy()
        y_data = ts_df[self.target_col].to_numpy().astype(np.float32)

        return X_data, y_data


class CMIDataModule(pl.LightningDataModule):
    def __init__(self, datapath: str, batch_size: int, sample_size: int):
        super().__init__()
        self.datapath = datapath
        self.batch_size = batch_size
        self.sample_size = sample_size

    def setup(self, stage: str):
        series_df = pd.read_parquet(self.datapath)
        series_df["anglez_1min_mean"] = series_df["anglez_1min_mean"] / 90.0
        time_enc_df = time_encoding(series_df)
        self.series_df = pd.concat([series_df, time_enc_df], axis=1)

        feat_cols = ["anglez_1min_mean", "enmo_1min_mean", "onset_prior", "awake_prior"]
        target_col = "asleep"
        dset = CMITimeSeriesSampler(
            series_df,
            sample_size=self.sample_size,
            feat_cols=feat_cols,
            target_col=target_col,
        )
        self.train_dset, self.val_dset = random_split(
            dset, [0.8, 0.2], generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return DataLoader(self.train_dset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dset, batch_size=self.batch_size, shuffle=False)
