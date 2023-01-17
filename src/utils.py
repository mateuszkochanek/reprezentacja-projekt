import base64
import io
from typing import Tuple, Type
import numpy as np
import pandas as pd
from datasets import load_dataset
from PIL import Image
import torch
from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt
import pytorch_lightning as pl
from dash import dcc, html, Input, Output, no_update
from jupyter_dash import JupyterDash
import plotly.express as px
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
        image_name = str(image_index)
        if len(image_name) < 5:
            image_name = "0" + image_name
        with Image.open(f"data/heatfull_meme/data/img/{image_name}.png") as img:
            return np.array(img)
    else:
        raise Exception(f"Dataset {dataset} is invalid")
        
        
def np_image_to_base64(im_matrix):
    im = Image.fromarray(im_matrix)
    buffer = io.BytesIO()
    im.save(buffer, format="png")
    encoded_image = base64.b64encode(buffer.getvalue()).decode()
    im_url = "data:image/jpeg;base64, " + encoded_image
    return im_url


def make_interactive_scatter_plot(
    title: str,
    z_2d: torch.Tensor,
    df: pd.DataFrame,
    port: int,
    dataset_name = 'hatefull',
    color_map = {"1": "red", "0": "blue"}
):
    data = df['label'].values.tolist()
    labels = [str(x) for x in data]
    fig = px.scatter(
        x=z_2d[:, 0],
        y=z_2d[:, 1],
        title=title,
        color=labels,
        color_discrete_map = color_map,
    )

    app = JupyterDash(__name__)

    app.layout = html.Div(
        className="container",
        children=[
            dcc.Graph(id="graph-5", figure=fig, clear_on_unhover=True),
            dcc.Tooltip(id="graph-tooltip-5", direction="bottom"),
        ],
    )

    @app.callback(
        Output("graph-tooltip-5", "show"),
        Output("graph-tooltip-5", "bbox"),
        Output("graph-tooltip-5", "children"),
        Input("graph-5", "hoverData"),
    )
    def display_hover(hoverData):
        if hoverData is None:
            return False, no_update, no_update

        hover_data = hoverData["points"][0]
        bbox = hover_data["bbox"]
        num = hover_data["pointNumber"]

        image = get_image(df["image_index"].iloc[num], dataset=dataset_name)
        #im_matrix = df["image"].iloc[num].permute(1, 2, 0).numpy().astype("uint8")
        im_url = np_image_to_base64(image)
        children = [
            html.Div([
                html.P(
                    f"Text: {df['text'].iloc[num]}",
                    style={
                        "width": "224px",
                        "fontSize": "10px",
                        "whiteSpace": "pre-wrap",
                    },
                ),
                html.Img(
                    src=im_url,
                    style={
                        "width": "224px",
                        "display": "block",
                        "margin": "0 auto",
                    },
                ),
            ])
        ]

        return True, bbox, children

    app.run_server(port=port, mode="inline", debug=True)



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