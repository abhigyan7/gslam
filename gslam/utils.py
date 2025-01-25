from multiprocessing import Queue
import pdb
import sys

from matplotlib import colormaps
import numpy as np
import torch
from PIL import Image
from sklearn.neighbors import NearestNeighbors

import functools
from typing import Callable, TypeVar, List

T1 = TypeVar("T1")


def create_batch(
    things: List[T1], getter: Callable[[T1], torch.Tensor]
) -> torch.Tensor:
    things = [getter(thing) for thing in things]
    return torch.stack(things, dim=0)


def knn(x: torch.Tensor, K: int = 4) -> torch.Tensor:
    x_np = x.cpu().numpy()
    model = NearestNeighbors(n_neighbors=K, metric="euclidean").fit(x_np)
    distances, _ = model.kneighbors(x_np)
    return torch.from_numpy(distances).to(x)


def torch_image_to_np(torch_img: torch.Tensor, minmax_norm: bool = False) -> np.ndarray:
    img = torch_img.detach().cpu().numpy()
    if minmax_norm:
        img = (img - img.min()) / (img.max() - img.min())
    img = np.uint8(img.clip(0.0, 1.0) * 255.0)
    return img


def torch_to_pil(torch_img: torch.Tensor, minmax_norm: bool = False) -> Image:
    img = torch_image_to_np(torch_img)
    return Image.fromarray(img)


def get_projection_matrix():
    Ks = (
        torch.FloatTensor(
            [
                [525.0, 0.0, 319.5],
                [0.0, 525.5, 239.5],
                [0.0, 0.0, 0.0],
            ]
        )
        .unsqueeze(0)
        .cuda()
    )
    return Ks


def q_get(queue: Queue):
    if queue.empty():
        return None
    return queue.get()


def unvmap(func: Callable[[torch.Tensor], torch.Tensor]):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        args = [arg.unsqueeze(0) for arg in args]
        ret = func(*args, **kwargs)
        return ret.squeeze(0)

    return wrapper


@torch.no_grad()
def false_colormap(
    image: torch.Tensor,
    near: float = None,
    far: float = None,
    mask: torch.Tensor = None,
) -> Image:
    '''image in (H,W)'''
    if mask is None:
        valid_pixels = image
    else:
        valid_pixels = image[mask]
    if near is None:
        near = valid_pixels.min()
    if far is None:
        far = valid_pixels.max()
    image = (image - near) / (far - near + 1e-10)
    image = torch.nan_to_num(image, 0.0)
    image = image.clip(0.0, 1.0)
    image = (image * 255.0).long()
    image = torch.tensor(colormaps['turbo'].colors, device=image.device)[image]
    image = image * 255.0
    if mask is not None:
        # make invalid regions black
        image[~mask] = 0.0
    image = image.detach().cpu().numpy().astype(np.uint8)
    return Image.fromarray(image)


class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    from https://stackoverflow.com/a/23654936
    """

    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin
