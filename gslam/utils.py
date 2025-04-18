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
    things: List[T1],
    getter: Callable[[T1], torch.Tensor] = None,
) -> torch.Tensor:
    if getter is not None:
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
    colormap: str = 'turbo',
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
    image = torch.tensor(colormaps[colormap].colors, device=image.device)[image]
    image = image * 255.0
    if mask is not None:
        # make invalid regions black
        image[~mask] = 0.0
    image = image.detach().cpu().numpy().astype(np.uint8)
    return Image.fromarray(image)


class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that works in multiprocessing child processes."""

    def __init__(self, global_pause_event, *args, **kwargs):
        self.global_pause_event = global_pause_event
        super().__init__(*args, **kwargs)  # Properly initialize Pdb

    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            self.global_pause_event.set()  # Signal all other processes to pause
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin
            self.global_pause_event.clear()  # Resume other processes after debugging


def total_variation_loss(img: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    v_h = img[..., 1:, :] - img[..., :-1, :]
    v_w = img[..., :, 1:] - img[..., :, :-1]
    if mask is not None:
        v_h = v_h * mask[..., 1:, :]
        v_w = v_w * mask[..., :, 1:]
    tv_h = (v_h).pow(2).mean()
    tv_w = (v_w).pow(2).mean()
    return tv_h + tv_w


def edge_aware_tv(
    depth: torch.Tensor, rgb: torch.Tensor, mask: torch.Tensor = None
) -> torch.Tensor:
    """
    Args:
        depth: [batch, H, W]
        rgb: [batch, H, W, 3]
        mask: [mask, H, W, 1]
    """
    grad_depth_x = torch.abs(depth[..., :, :-1, None] - depth[..., :, 1:, None])
    grad_depth_y = torch.abs(depth[..., :-1, :, None] - depth[..., 1:, :, None])

    grad_img_x = torch.mean(
        torch.abs(rgb[..., :, :-1, :] - rgb[..., :, 1:, :]), -1, keepdim=True
    )
    grad_img_y = torch.mean(
        torch.abs(rgb[..., :-1, :, :] - rgb[..., 1:, :, :]), -1, keepdim=True
    )

    grad_depth_x *= torch.exp(-grad_img_x)
    grad_depth_y *= torch.exp(-grad_img_y)

    return (
        grad_depth_x[mask[..., :, :-1, None]].sum()
        + grad_depth_y[mask[..., :-1, :, None]].sum()
    )


class StopOnPlateau:
    '''Stop optimization if loss doesn't decrease appreciably for a bit'''

    def __init__(self, patience, min_loss):
        self.patience = patience
        self.counter = 0
        self.min_loss = min_loss
        self.last_loss = None

    def stop(self, loss):
        if self.last_loss is None:
            self.last_loss = loss
            return False
        if loss > self.min_loss:
            return False
        elif self.last_loss > loss:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.counter = 0
        self.last_loss = loss
        return False


# A class that does absolutely nothing
# Useful to turn rerun logs into no-ops
class BlackHole:
    def __init__(self, *args, **kwargs):
        pass

    def __get_attr__(self, name):
        return self

    def __set_attr__(self, name, val):
        pass

    def __call__(self, *args, **kwargs):
        return self
