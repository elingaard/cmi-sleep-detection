from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl


class FPN1D(nn.Module):
    """Feature pyramid extraction for 1D data. Creates features at different scales
    using dilated convolutions"""

    def __init__(self, in_chs: int, out_chs: int, kernel_size: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_chs, out_chs, kernel_size=kernel_size, padding="same")
        self.conv2 = nn.Conv1d(
            in_chs,
            out_chs,
            kernel_size=kernel_size,
            padding="same",
            dilation=2,
        )
        self.conv3 = nn.Conv1d(
            in_chs,
            out_chs,
            kernel_size=kernel_size,
            padding="same",
            dilation=4,
        )
        self.batch_norm = nn.BatchNorm1d(out_chs * 3)
        self.activation = nn.LeakyReLU()

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.batch_norm(x)
        x = self.activation(x)
        return x


class ResNetBlock1D(nn.Module):
    def __init__(self, n_chs: int, kernel_size: int) -> None:
        super().__init__()
        self.n_chs = n_chs
        self.kernel_size = kernel_size

        self.block = nn.Sequential(
            nn.Conv1d(n_chs, n_chs, kernel_size=kernel_size, padding="same"),
            nn.BatchNorm1d(n_chs),
            nn.LeakyReLU(),
            nn.Conv1d(n_chs, n_chs, kernel_size=kernel_size, padding="same"),
            nn.BatchNorm1d(n_chs),
        )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        res = self.block(x)
        x_out = x + res
        x_out = F.leaky_relu(x_out)
        return x_out


class ClassificationHead(nn.Module):
    def __init__(self, in_chs) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_chs, in_chs // 4, kernel_size=5, padding="same"
            ),
            nn.BatchNorm1d(in_chs // 4),
            nn.LeakyReLU(),
            nn.Dropout1d(p=0.4)
        )
        self.conv2 = nn.Conv1d(
            in_chs // 4, 1, kernel_size=61, padding="same"
        )

    def forward(
        self, x: torch.FloatTensor
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        x = self.conv1(x)
        y_logits = self.conv2(x).squeeze()
        y_proba = F.sigmoid(y_logits)
        return y_logits, y_proba


class EventDetectionCNN(nn.Module):
    def __init__(self, in_chs: int, feat_chs, n_resnet_blocks: int) -> None:
        super().__init__()
        resnet_feat_chs = feat_chs * 3
        self.input_layer = FPN1D(in_chs, feat_chs, kernel_size=31)
        self.resnet_blocks = nn.Sequential(
            *[ResNetBlock1D(resnet_feat_chs, kernel_size=11)] * n_resnet_blocks
        )
        self.cls_head = ClassificationHead(resnet_feat_chs)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.resnet_blocks(x)
        y_logits, y_proba = self.cls_head(x)
        return y_logits, y_proba

    def predict_from_df(self, df, feat_cols):
        x = df[feat_cols].T.to_numpy(dtype=np.float32)
        x = torch.tensor(x).float().unsqueeze(0)
        with torch.no_grad():
            _, y_proba = self.forward(x)
        return y_proba.numpy()


class CMISleepDetectionCNN(pl.LightningModule):
    def __init__(self, in_chs: int, feat_chs: int, n_resnet_blocks: int):
        super().__init__()
        self.model = EventDetectionCNN(in_chs, feat_chs, n_resnet_blocks)
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x_in, y_target = batch
        y_logits, _ = self(x_in)
        loss = self.bce_loss(y_logits, y_target)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x_in, y_target = batch
        y_logits, y_proba = self(x_in)
        val_loss = self.bce_loss(y_logits, y_target)
        self.log("val/loss", val_loss)

        # calculate accuracy
        y_pred = y_proba > 0.5
        y_target = y_target > 0.5
        val_acc = torch.sum(y_pred == y_target) / y_target.nelement()
        self.log("val/acc", val_acc)

        return val_loss, val_acc

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=2e-4)
