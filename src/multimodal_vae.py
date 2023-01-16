import os
import sys
import shutil
from tqdm import tqdm
from typing import Dict, List, Type, Tuple
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from abc import abstractmethod

class Swish(nn.Module):
    """https://arxiv.org/abs/1710.05941"""

    def forward(self, x):
        return x * torch.sigmoid(x)

class Encoder(nn.Module):
    """Parametrizes q(z|x)."""

    def __init__(self,
                 in_dims: int,
                 hidden_dims: int,
                 out_dim: int,
                 activation: Type[nn.Module]
                 ):
        super(Encoder, self).__init__()
        self.activation = activation
        self.out_dim = out_dim
        self.encode = nn.Sequential(
            nn.Linear(in_dims, hidden_dims[0]),
            self.activation(),
            *[
                layer
                for idx in range(len(hidden_dims) - 1)
                for layer in (nn.Linear(hidden_dims[idx], hidden_dims[idx + 1]), self.activation())
            ],
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dims[-1], out_dim * 2)
        )

    def forward(self, x):
        out_dim = self.out_dim
        x = self.encode(x)
        return x[:, :out_dim], x[:, out_dim:]
    
    @staticmethod
    def from_hparams(hparams, type_name):
        return Encoder(
            in_dims=hparams["data_dims"][type_name],
            hidden_dims=hparams["hidden_dims"][type_name],
            out_dim=hparams["emb_dim"],
            activation=hparams["activation"],
        )


class Decoder(nn.Module):
    """Parametrizes p(x|z)."""

    def __init__(self,
                 in_dims: int,
                 hidden_dims: int,
                 out_dim: int,
                 activation: Type[nn.Module]
                 ):
        super(Decoder, self).__init__()
        self.out_dim = out_dim
        self.activation = activation
        self.decode = nn.Sequential(
            nn.Linear(in_dims, hidden_dims[0]),
            self.activation(),
            *[
                layer
                for idx in range(len(hidden_dims) - 1)
                for layer in (nn.Linear(hidden_dims[idx], hidden_dims[idx + 1]), self.activation())
            ],
            nn.Linear(hidden_dims[-1], out_dim)
        )

    def forward(self, z):
        # the input will be a vector of size |n_latents|
        z = self.decode(z)
        # returns reconstructed image/text embedding
        return z  # NOTE: no sigmoid here. See in training
    
    @staticmethod
    def from_hparams(hparams, type_name):
        return Decoder(
            in_dims=hparams["emb_dim"],
            hidden_dims=hparams["hidden_dims"][type_name][::-1],
            out_dim=hparams["data_dims"][type_name],
            activation=hparams["activation"],
        )
    
class ProductOfExperts(nn.Module):
    """Return parameters for product of independent experts.
    See https://arxiv.org/pdf/1410.7827.pdf for equations.

    @param mu: M x D for M experts
    @param logvar: M x D for M experts
    """

    def forward(self, mu, logvar, eps=1e-8):
        var = torch.exp(logvar) + eps
        # precision of i-th Gaussian expert at point x
        T = 1. / var
        pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var = 1. / torch.sum(T, dim=0)
        pd_logvar = torch.log(pd_var)
        return pd_mu, pd_logvar
    
def prior_expert(size, use_cuda=False):
    """Universal prior expert. Here we use a spherical
    Gaussian: N(0, 1).

    @param size: integer
                 dimensionality of Gaussian
    @param use_cuda: boolean [default: False]
                     cast CUDA on variables
    """
    mu = torch.zeros(size)
    logvar = torch.log(torch.ones(size))
    if use_cuda:
        mu, logvar = mu.cuda(), logvar.cuda()
    return mu, logvar
    
def elbo_loss(recon_img_emb, img_emb, recon_text_emb, text_emb, mu, logvar,
              lambda_image=1.0, lambda_text=1.0, annealing_factor=1):
    
    image_bce, text_bce = 0, 0  # default params

    if recon_img_emb is not None and img_emb is not None:
        image_bce = F.mse_loss(recon_img_emb, img_emb) #torch.sum(binary_cross_entropy_with_logits(recon_img_emb, img_emb))

    if recon_text_emb is not None and text_emb is not None:
        text_bce = F.mse_loss(recon_text_emb, text_emb) #torch.sum(binary_cross_entropy_with_logits(recon_text_emb, text_emb))
    
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    KLD = torch.mean(
        -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1),
        dim=0,
    )

    ELBO = lambda_image * image_bce + lambda_text * text_bce + annealing_factor * KLD
    return ELBO

class BaseVAE(pl.LightningModule):

    def __init__(self,
                 hparams,
                 image_encoder,
                 image_decoder,
                 text_encoder,
                 text_decoder,):
        super().__init__()

        self.save_hyperparameters(hparams)

        self.image_encoder = image_encoder
        self.image_decoder = image_decoder
        self.text_encoder = text_encoder
        self.text_decoder = text_decoder

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

class MVAE(BaseVAE):
    def __init__(self, hparams):
        super().__init__(
            hparams=hparams,
            image_encoder=Encoder.from_hparams(hparams, 'image'),
            image_decoder=Decoder.from_hparams(hparams, 'image'),
            text_encoder=Encoder.from_hparams(hparams, 'text'),
            text_decoder=Decoder.from_hparams(hparams, 'text')
        )
        self.product_of_experts = ProductOfExperts()
        self.n_latents = hparams["emb_dim"]
        self.lambda_image=hparams["lambda_image"]
        self.lambda_text=hparams["lambda_text"]
        self.annealing_factor=hparams["annealing_factor"]

    def get_representation(self, img_emb, text_emb):
        mu, logvar = self.forward_encoder(img_emb, text_emb)
        z = self.reparametrize(mu, logvar)
        return z
        
    def reparametrize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:  # return mean during inference
            return mu

    def forward(self, img_emb=None, text_emb=None):
        mu, logvar = self.forward_encoder(img_emb, text_emb)
        # reparametrization trick to sample
        z = self.reparametrize(mu, logvar)
        # reconstruct inputs based on that gaussian
        image_recon, text_recon = self.forward_decoder(z)
        return image_recon, text_recon, mu, logvar

    def forward_encoder(self, img_emb=None, text_emb=None):
        if img_emb is not None:
            batch_size = img_emb.size(0)
        else:
            batch_size = text_emb.size(0)

        use_cuda = next(self.parameters()).is_cuda  # check if CUDA
        mu, logvar = prior_expert((1, batch_size, self.n_latents),
                                  use_cuda=use_cuda)
        if img_emb is not None:
            image_mu, image_logvar = self.image_encoder(img_emb)
            mu = torch.cat((mu, image_mu.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, image_logvar.unsqueeze(0)), dim=0)

        if text_emb is not None:
            text_mu, text_logvar = self.text_encoder(text_emb)
            mu = torch.cat((mu, text_mu.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, text_logvar.unsqueeze(0)), dim=0)

        # product of experts to combine gaussians
        mu, logvar = self.product_of_experts(mu, logvar)
        return mu, logvar

    def forward_decoder(self, z):
        image_recon = self.image_decoder(z)
        text_recon = self.text_decoder(z)
        return image_recon, text_recon
    
    def _common_step(self, batch) -> torch.Tensor:
        img_emb = batch['img_emb']
        text_emb = batch['text_emb']
        train_loss = 0  # accumulate train loss here so we don't store a lot of things.

        # compute ELBO using all data (``complete")
        recon_img_emb, recon_text_emb, mu, logvar = self.forward(img_emb, text_emb)
        train_loss += elbo_loss(recon_img_emb, img_emb, recon_text_emb, text_emb,
                                mu=mu,
                                logvar=logvar,
                                lambda_image=self.lambda_image,
                                lambda_text=self.lambda_text,
                                annealing_factor=self.annealing_factor
                                )

        # compute ELBO using only img_emb data
        recon_img_emb, _, mu, logvar = self.forward(img_emb=img_emb)
        train_loss += elbo_loss(recon_img_emb, img_emb, None, None,
                                mu=mu,
                                logvar=logvar,
                                lambda_image=self.lambda_image,
                                lambda_text=self.lambda_text,
                                annealing_factor=self.annealing_factor
                                )

        # compute ELBO using only text data
        _, recon_text_emb, mu, logvar = self.forward(text_emb=text_emb)
        train_loss += elbo_loss(None, None, recon_text_emb, text_emb,
                                mu=mu,
                                logvar=logvar,
                                lambda_image=self.lambda_image,
                                lambda_text=self.lambda_text,
                                annealing_factor=self.annealing_factor
                                )
        return train_loss
