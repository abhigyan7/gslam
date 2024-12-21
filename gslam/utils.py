from multiprocessing import Queue

import numpy as np
import torch
from PIL import Image
from sklearn.neighbors import NearestNeighbors

import functools
from typing import Callable


def create_batch(things, getter):
    things = [getter(thing) for thing in things]
    return torch.stack(things, dim=0)


def knn(x: torch.Tensor, K: int = 4) -> torch.Tensor:
    x_np = x.cpu().numpy()
    model = NearestNeighbors(n_neighbors=K, metric="euclidean").fit(x_np)
    distances, _ = model.kneighbors(x_np)
    return torch.from_numpy(distances).to(x)


def torch_image_to_np(torch_img: torch.Tensor) -> np.ndarray:
    img = torch_img[0, ...].detach().cpu().numpy()
    img = np.uint8(img.clip(0.0, 1.0) * 255.0)
    return img


def torch_to_pil(torch_img: torch.Tensor) -> Image:
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


def kabsch_umeyama(A, B):
    """
    implementation from https://zpl.fi/aligning-point-patterns-with-kabsch-umeyama-algorithm/
    """
    assert A.shape == B.shape
    n, m = A.shape

    EA = np.mean(A, axis=0)
    EB = np.mean(B, axis=0)
    VarA = np.mean(np.linalg.norm(A - EA, axis=1) ** 2)

    H = ((A - EA).T @ (B - EB)) / n
    U, D, VT = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U) * np.linalg.det(VT))
    S = np.diag([1] * (m - 1) + [d])

    R = U @ S @ VT
    c = VarA / np.trace(np.diag(D) @ S)
    t = EA - c * R @ EB

    return R, c, t


def unvmap(func: Callable[[torch.Tensor], torch.Tensor]):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        args = [arg.unsqueeze(0) for arg in args]
        ret = func(*args, **kwargs)
        return ret.squeeze(0)

    return wrapper
