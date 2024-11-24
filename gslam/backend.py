from dataclasses import dataclass, field
from gsplat.strategy import MCMCStrategy, DefaultStrategy
from typing import Union, Tuple

import torch

@dataclass
class BackendConfig:

    densification_strategy: Union[DefaultStrategy, MCMCStrategy] = field (
        default_factory=DefaultStrategy
    )

    opacity_regularization: float = 0.0
    scale_regularization: float = 0.0

    enable_pose_optimization: bool = False
    pose_optimization_lr: float = 1e-5
    pose_optimization_regularization = 1e-6
    pose_noise: float = 0.0

    # background rgb
    background_color: Tuple[float, 3] = [0.0, 0.0, 0.0]

    initialization_type: str = 'random'
    initial_number_of_gaussians: int = 30_000
    initial_extent: float = 3.0
    initial_opacity: float = 0.1
    initial_scale: float = 1.0
    scene_scale: float = 1.0

    device: str = 'cuda:0'


class Backend(torch.mp.Process):

    def __init__(self, backend_config: BackendConfig):
        self.backend_config = backend_config


    def run(self):
        return
