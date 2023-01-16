from typing import Type
import pytorch_lightning as pl
from .dataset import DataModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


def train_model(
    model_cls: Type[pl.LightningModule],
    hparams,
    datamodule: DataModule,
    accelerator="gpu",
):
    pl.seed_everything(42)

    model = model_cls(hparams)

    model_chkpt = ModelCheckpoint(
        dirpath=f"./data/checkpoints/{hparams['name']}/",
        filename="model",
        monitor="val/loss",
        mode="min",
        verbose=True,
    )
    trainer = pl.Trainer(
        logger=TensorBoardLogger(
            save_dir="./data/logs",
            name=hparams["name"],
            default_hp_metric=False,
        ),
        callbacks=[model_chkpt],
        num_sanity_val_steps=0,
        log_every_n_steps=1,
        max_epochs=hparams["num_epochs"],
        accelerator=accelerator,
    )
    trainer.fit(model=model, datamodule=datamodule)