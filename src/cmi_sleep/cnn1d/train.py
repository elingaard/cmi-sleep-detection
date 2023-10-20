import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from cmi_sleep.cnn1d.data import CMIDataModule
from cmi_sleep.cnn1d.model import CMISleepDetectionCNN

if __name__ == "__main__":
    datapath = "/Users/martinelingaard/repos/cmi-sleep-detection/data/train_series_1min.parquet"
    datamodule = CMIDataModule(datapath, batch_size=128, sample_size=60 * 12)
    model = CMISleepDetectionCNN(in_chs=4, feat_chs=32, n_resnet_blocks=3)

    model_ckpt_callback = ModelCheckpoint(save_top_k=1, monitor="val/acc", mode="max")
    early_stop_callback = EarlyStopping(monitor="val/acc", mode="max", patience=10)
    trainer = pl.Trainer(
        log_every_n_steps=1,
        check_val_every_n_epoch=10,
        max_epochs=1000,
        accelerator="mps",
        callbacks=[early_stop_callback, model_ckpt_callback],
    )
    trainer.fit(model, datamodule)
