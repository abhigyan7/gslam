from typing import Tuple, Dict, Union
from dataclasses import dataclass, field

import torch
from gsplat.strategy import DefaultStrategy, MCMCStrategy

from .utils import knn

@dataclass
class MapConfig:

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
    background_color: Tuple[float, 3] = (0.0, 0.0, 0.0)

    initialization_type: str = 'random'
    initial_number_of_gaussians: int = 300_000
    initial_extent: float = 3000.0
    initial_opacity: float = 0.9
    initial_scale: float = 1.0
    scene_scale: float = 1.0

    device: str = 'cuda:0'


def initialize_map(
    map_conf: MapConfig,
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:

    points = torch.rand( (map_conf.initial_number_of_gaussians, 3) ) * 2.0 - 1.0
    points *= map_conf.initial_extent * map_conf.scene_scale
    rgbs = torch.rand( (map_conf.initial_number_of_gaussians, 3) )

    avg_distance_to_nearest_3_neighbors = torch.sqrt( knn(points, 4)[:, 1:] ** 2 ).mean(dim=-1)
    scales = torch.log(avg_distance_to_nearest_3_neighbors * map_conf.initial_scale).unsqueeze(-1).repeat(1, 3)

    N = points.shape[0]
    quats = torch.rand( (N, 4) )
    opacities = torch.logit( torch.full( (N,), map_conf.initial_opacity ))

    params = [
        ('means', torch.nn.Parameter(points), 1.6e-4 * map_conf.scene_scale),
        ('quats', torch.nn.Parameter(quats), 1e-3),
        ('scales', torch.nn.Parameter(scales), 5e-3),
        ('opacities', torch.nn.Parameter(opacities), 5e-2),
        ('colors', torch.nn.Parameter(rgbs), 2.5e-3 / 20),
    ]

    params_dict = torch.nn.ParameterDict(
        {name: param for name, param, _lr in params}
    ).to(map_conf.device)

    optimizers = {
        name: torch.optim.Adam(
            params=[params_dict[name],], lr=lr,
        ) for name, value, lr in params
    }
    return params_dict, optimizers


# consider the implications of all these structs being torch modules
class GaussianSplattingModel:

    def __init__(self,
                 means,        # gaussian centers
                 covar_quats,  # quaternions of covariance matrices
                 covar_scales, # scales of covariance matrices
                 opacities,    # alpha of gaussians
                 colors):      # RGB values of gaussians
        self.means: torch.Tensor = means
        self.covar_quats: torch.Tensor = covar_quats
        self.covar_scales: torch.Tensor = covar_scales
        self.opacities: torch.Tensor = opacities
        self.colors: torch.Tensor = colors


    def render(self,
               camera: Camera):
        rendered_rgb, rendered_alpha, render_info = rasterization(
            means = self.means,
            quats = self.covar_quats,
            scales = self.covar_scales,
            opacities = self.opacities,
            colors = self.colors,
            viewmats = camera.viewmat.unsqueeze(0),
            Ks = camera.intrinsics,
            width = camera.width,
            height = camera.height,
        )
        return rendered_rgb, rendered_alpha, render_info


    @staticmethod
    def new_empty_model(device: str = 'cuda'):
        return GaussianSplattingModel(
            torch.Tensor(device=device),
            torch.Tensor(device=device),
            torch.Tensor(device=device),
            torch.Tensor(device=device),
            torch.Tensor(device=device),
        )


    def clone(self):
        return GaussianSplattingModel(
            self.means.clone(),
            self.covar_quats.clone(),
            self.covar_scales.clone(),
            self.opacities.clone(),
            self.colors.clone(),
        )