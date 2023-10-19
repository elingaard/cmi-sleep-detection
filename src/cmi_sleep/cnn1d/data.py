from typing import Tuple

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import lightning.pytorch as pl


def time_prior_encoding(ts_df: pd.DataFrame):
    """Use most common awake and onset time as a prior for encoding the timestamp to
    a numerical value."""
    awake_prior = dict(
        zip(range(1440), np.sin(np.linspace(0, np.pi, 1440) + 0.208 * np.pi) ** 24)
    )
    onset_prior = dict(
        zip(range(1440), np.sin(np.linspace(0, np.pi, 1440) + 0.555 * np.pi) ** 24)
    )
    ts_df["onset_prior"] = (
        (ts_df.timestamp.dt.hour * 60 + ts_df.timestamp.dt.minute)
        .map(onset_prior)
        .astype(np.float32)
    )
    ts_df["awake_prior"] = (
        (ts_df.timestamp.dt.hour * 60 + ts_df.timestamp.dt.minute)
        .map(awake_prior)
        .astype(np.float32)
    )
    return ts_df


class CMITimeSeriesSampler(Dataset):
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
        std_scaler = StandardScaler()
        series_df[["anglez_1min_mean", "enmo_1min_mean"]] = std_scaler.fit_transform(
            series_df[["anglez_1min_mean", "enmo_1min_mean"]]
        )
        series_df = time_prior_encoding(series_df)

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
