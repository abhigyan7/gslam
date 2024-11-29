import torch
from dataclasses import dataclass


@dataclass
class Camera:
    viewmat: torch.Tensor
    intrinsics: torch.Tensor
    height: int
    width: int


@dataclass
class Frame:
    img: torch.Tensor
    timestamp: float
    camera: Camera
    kind: str = 'rgb'
