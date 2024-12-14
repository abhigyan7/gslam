from multiprocessing import Queue

import numpy as np
import torch
from PIL import Image
from sklearn.neighbors import NearestNeighbors


def knn(x: torch.Tensor, K: int = 4) -> torch.Tensor:
    x_np = x.cpu().numpy()
    model = NearestNeighbors(n_neighbors=K, metric="euclidean").fit(x_np)
    distances, _ = model.kneighbors(x_np)
    return torch.from_numpy(distances).to(x)


def torch_to_pil(torch_img: torch.Tensor) -> Image:
    img = torch_img[0, ...].detach().cpu().numpy()
    img = np.uint8(img.clip(0.0, 1.0) * 255.0)
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
