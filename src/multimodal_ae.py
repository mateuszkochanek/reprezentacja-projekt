from typing import List, Dict, Type
import torch
from torch import nn
from torch.nn.functional import mse_loss
import pytorch_lightning as pl
from abc import abstractmethod
import numpy as np


class MultimodalEncoder(nn.Module):

    def __init__(
        self,
        modality_names: List[str],
        in_dims: Dict[str, int],
        hidden_dims: List[int],
        out_dim: int,
        last_activation: Type[nn.Module],
    ):
        super().__init__()

        self.modality_names = modality_names
        
        self.modality_to_encoder_map = nn.ModuleDict()
        for modality_name in modality_names:
            self.modality_to_encoder_map[modality_name] = nn.Sequential(
                nn.Linear(in_dims[modality_name], hidden_dims[0]),
                nn.ReLU(inplace=True),
                *[
                    layer
                    for idx in range(len(hidden_dims) - 1)
                    for layer in (nn.Linear(hidden_dims[idx], hidden_dims[idx + 1]), nn.ReLU(inplace=True))
                ],
                nn.Linear(hidden_dims[-1], out_dim),
                last_activation(),
            )
            

    def forward(self, x: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        return [
            self.modality_to_encoder_map[modality_name](x[modality_name])
            for modality_name in self.modality_names
        ]
        
    @staticmethod
    def from_hparams(hparams):
        return MultimodalEncoder(
            modality_names=hparams["modality_names"],
            in_dims=hparams["data_dims"],
            hidden_dims=hparams["hidden_dims"],
            out_dim=hparams["emb_dim"],
            last_activation=nn.Tanh,
        )


class AvgFusion(nn.Module):
    
    def forward(self, h: List[torch.Tensor]) -> torch.Tensor:
        return sum(h) / len(h)
    
    
class MLPFusion(nn.Module):
    
    def __init__(
        self,
        modality_dim: int,
        num_modalities: int,
        hidden_dims: List[int],
        out_dim: int,
        last_activation: Type[nn.Module],
    ):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(modality_dim * num_modalities, hidden_dims[0]),
            nn.ReLU(inplace=True),
            *[
                layer
                for idx in range(len(hidden_dims) - 1)
                for layer in (nn.Linear(hidden_dims[idx], hidden_dims[idx + 1]), nn.ReLU(inplace=True))
            ],
            nn.Linear(hidden_dims[-1], out_dim),
            last_activation(),
        )
        
        
    def forward(self, h: List[torch.Tensor]) -> torch.Tensor:
        mlp_input = torch.cat(h, dim=1)
        return self.mlp(mlp_input)


class MultimodalDecoder(nn.Module):

    def __init__(
        self,
        modality_names: List[str],
        in_dim: int,
        hidden_dims: List[int],
        out_dims: Dict[str, int],
        last_activation: Type[nn.Module],
    ):
        super().__init__()

        self.modality_names = modality_names
        
        self.modality_to_encoder_map = nn.ModuleDict()
        for modality_name in modality_names:
            self.modality_to_encoder_map[modality_name] = nn.Sequential(
                nn.Linear(in_dim, hidden_dims[0]),
                nn.ReLU(inplace=True),
                *[
                    layer
                    for idx in range(len(hidden_dims) - 1)
                    for layer in (nn.Linear(hidden_dims[idx], hidden_dims[idx + 1]), nn.ReLU(inplace=True))
                ],
                nn.Linear(hidden_dims[-1], out_dims[modality_name]),
                last_activation(),
            )

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            modality_name: self.modality_to_encoder_map[modality_name](z)
            for modality_name in self.modality_names
        }


class BaseAE(pl.LightningModule):

    def __init__(self, hparams, encoder: nn.Module, decoder: nn.Module):
        super().__init__()

        self.save_hyperparameters(hparams)

        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, batch_idx: int):
        return {"loss": self._common_step(batch)}

    def training_epoch_end(self, outputs):
        avg_loss = self._summarize_outputs(outputs)

        self.log("step", self.trainer.current_epoch)
        self.log("train/loss", avg_loss, on_epoch=True, on_step=False)

    def validation_step(self, batch, batch_idx: int):
        return {"loss": self._common_step(batch)}

    def validation_epoch_end(self, outputs):
        avg_loss = self._summarize_outputs(outputs)

        self.log("step", self.trainer.current_epoch)
        self.log("val/loss", avg_loss, on_epoch=True, on_step=False)

    @abstractmethod
    def _common_step(self, batch) -> torch.Tensor:
        pass

    @staticmethod
    def _summarize_outputs(outputs):
        losses = [out["loss"] for out in outputs]

        avg_loss = np.mean([loss.cpu() for loss in losses])

        return avg_loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            params=self.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )


class MultimodalAE(BaseAE):

    def __init__(self, hparams):
        encoder_cls = hparams["encoder_cls"]
        
        super().__init__(
            hparams=hparams,
            encoder=encoder_cls.from_hparams(hparams),
            decoder=MultimodalDecoder(
                modality_names=hparams["modality_names"],
                in_dim=hparams["emb_dim"],
                hidden_dims=hparams["hidden_dims"][::-1],
                out_dims=hparams["data_dims"],
                last_activation=nn.Identity,
            ),
        )
        
        if hparams["fusion"] == "Avg":
            self.fusion = AvgFusion()
        elif hparams["fusion"] == "MLP":
            self.fusion = MLPFusion(
                modality_dim=hparams["emb_dim"],
                num_modalities=len(hparams["modality_names"]),
                hidden_dims=[hparams["emb_dim"], hparams["emb_dim"]],
                out_dim=hparams["emb_dim"],
                last_activation=nn.Tanh,
            )
        else:
            raise ValueError(f"Unknown fusion module: \"{hparams['fusion']}\"")

    def forward(self, batch) -> torch.Tensor:
        encoded = self.encoder(batch)
        return self.fusion(encoded)
    
    def encode(self, batch):
        return self.forward(batch)

    def _common_step(self, batch) -> torch.Tensor:
        z = self.forward(batch)
        x_rec = self.decoder(z)
        mse = 0
        for modality_name in x_rec:
            mse += mse_loss(batch[modality_name], x_rec[modality_name])
        return mse / len(batch)
