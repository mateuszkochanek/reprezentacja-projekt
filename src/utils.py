from typing import Tuple, Type
import numpy as np
from datasets import load_dataset
from PIL import Image
import torch
from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt
import pytorch_lightning as pl

from .dataset import DataModule


CUBE_DATASET = load_dataset("alkzar90/CC6204-Hackaton-Cub-Dataset")


def get_most_similar(
    x: torch.Tensor,
    anchor: torch.Tensor,
    metric: str = "cosine",
    num_neighbors: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:

    nn = NearestNeighbors(
        n_neighbors=num_neighbors + 1,
        metric=metric
    ).fit(x)

    similarities, indices = nn.kneighbors(anchor.reshape(1, -1))
    return similarities[0][1:], indices[0][1:]


def get_image(image_index: int, split="train", dataset="cub") -> np.ndarray:
    if dataset == "cub":
        return np.array(CUBE_DATASET[split][image_index]['image'])
    elif dataset == "hatefull":
        with Image.open(f"data/heatfull_meme/data/img/{image_index}.png") as img:
            return np.array(img)
    else:
        raise Exception(f"Dataset {dataset} is invalid")


@torch.no_grad()
def extract_multimodal_ae_embeddings(
    model_cls: Type[pl.LightningModule],
    name: str,
    datamodule: DataModule,
):
    best_model = model_cls.load_from_checkpoint(
        checkpoint_path=f"./data/checkpoints/{name}/model.ckpt",
        use_cuda=True,
    )
    best_model = best_model.cuda()
    best_model.eval()

    z = []

    for batch in datamodule.all_dataloader():
        z.append(best_model(batch))

    return torch.cat(z, dim=0)

@torch.no_grad()
def extract_multimodal_vae_embeddings(
    model_cls: Type[pl.LightningModule],
    name: str,
    datamodule: DataModule,
):
    best_model = model_cls.load_from_checkpoint(
        checkpoint_path=f"./data/checkpoints/{name}/model.ckpt",
        use_cuda=True,
    )
    best_model = best_model.cuda()
    best_model.eval()

    z = []
    for batch in datamodule.all_dataloader():
        text_emb = batch["text_emb"].cuda()
        img_emb = batch["img_emb"].cuda()
        z.append(best_model.get_representation(img_emb, text_emb))
    return torch.cat(z, dim=0).cpu()


def visualize_most_similar(
    df,
    title: str,
    anchor_index: int,
    z: torch.Tensor,
    metric: str = "l2",
    num_similar: int = 5,
    split="train",
    dataset="cub",
) -> plt.Figure:
    fig, axs = plt.subplots(figsize=(15, 5), ncols=num_similar + 1)

    similarities, indices = get_most_similar(
        x=z,
        anchor=z[anchor_index],
        metric=metric,
        num_neighbors=num_similar,
    )
    
    image = get_image(
        int(df.loc[anchor_index, "image_index"]),
        split=split,
        dataset=dataset,
    )
    desc = df.loc[anchor_index, "text"]
    print(f"Anchor: {desc}")
    axs[0].imshow(image)
    axs[0].set(title="Anchor", xticks=[], yticks=[])

    for sim, index, ax in zip(similarities, indices, axs[1:]):
        image = get_image(
            int(df.loc[index, "image_index"]),
            split=split,
            dataset=dataset,
        )
        ax.imshow(image)
        desc = df.loc[index, "text"]
        print(f"{index}: {desc}")
        ax.set(title=f"Dist: {sim:.10f}, ID: {index}", xticks=[], yticks=[])
    fig.suptitle(title)
    fig.tight_layout()

    return fig