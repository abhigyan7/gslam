from torch import Tensor, from_numpy
from sklearn.neighbors import NearestNeighbors
from PIL import Image
import numpy as np

def knn(x: Tensor, K: int = 4) -> Tensor:
    x_np = x.cpu().numpy()
    model = NearestNeighbors(n_neighbors=K, metric="euclidean").fit(x_np)
    distances, _ = model.kneighbors(x_np)
    return from_numpy(distances).to(x)

def torch_to_pil(torch_img: Tensor) -> Image:
    img = torch_img[0, ...].detach().cpu().numpy()
    img = np.uint8(img.clip(0.0, 1.0) * 255.0)
    return Image.fromarray(img)