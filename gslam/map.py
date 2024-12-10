from typing import Tuple, Dict, Union, Self, Literal
from dataclasses import dataclass, field

import torch
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from gsplat.rendering import rasterization

from .utils import knn
from .primitives import Camera, Pose


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


# consider the implications of all these structs being torch modules
class GaussianSplattingData(torch.nn.Module):

    def __init__(self,
                 means,        # gaussian centers
                 covar_quats,  # quaternions of covariance matrices
                 covar_scales, # scales of covariance matrices
                 opacities,    # alpha of gaussians
                 colors):      # RGB values of gaussians
        super().__init__()
        self.means: torch.nn.Parameter = torch.nn.Parameter(means)
        self.covar_quats: torch.nn.Parameter = torch.nn.Parameter(covar_quats)
        self.covar_scales: torch.nn.Parameter = torch.nn.Parameter(covar_scales)
        self.opacities: torch.nn.Parameter = torch.nn.Parameter(opacities)
        self.colors: torch.nn.Parameter = torch.nn.Parameter(colors)


    def forward(self,
               camera: Camera, pose: Pose) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        rendered_rgb, rendered_alpha, render_info = rasterization(
            means = self.means,
            quats = self.covar_quats,
            scales = self.covar_scales,
            opacities = self.opacities,
            colors = self.colors,
            viewmats = pose(),
            Ks = camera.intrinsics,
            width = camera.width,
            height = camera.height,
        )
        return rendered_rgb, rendered_alpha, render_info


    @staticmethod
    def new_empty_model(device: str = 'cuda'):
        return GaussianSplattingData(
            torch.tensor([], device=device),
            torch.tensor([], device=device),
            torch.tensor([], device=device),
            torch.tensor([], device=device),
            torch.tensor([], device=device),
        )


    def clone(self) -> Self:
        return GaussianSplattingData(
            self.means.clone().detach(),
            self.covar_quats.clone().detach(),
            self.covar_scales.clone().detach(),
            self.opacities.clone().detach(),
            self.colors.clone().detach(),
        )


    # don't use this
    def __initialize_map_random(self, map_conf: MapConfig):
        points = torch.rand( (map_conf.initial_number_of_gaussians, 3) ) * 2.0 - 1.0
        points *= map_conf.initial_extent * map_conf.scene_scale
        rgbs = torch.rand( (map_conf.initial_number_of_gaussians, 3) )

        avg_distance_to_nearest_3_neighbors = torch.sqrt( knn(points, 4)[:, 1:] ** 2 ).mean(dim=-1)
        scales = torch.log(avg_distance_to_nearest_3_neighbors * map_conf.initial_scale).unsqueeze(-1).repeat(1, 3)

        N = points.shape[0]
        quats = torch.rand( (N, 4) )
        opacities = torch.logit( torch.full( (N,), map_conf.initial_opacity ))

        self = GaussianSplattingData(points, quats, scales, opacities, rgbs)


class GaussianSplattingMap:

    def __init__(self,
                 map_config: MapConfig,
                 data: GaussianSplattingData = None,):
        self.data = data
        self.map_conf = map_config
        return


    def initialize_map_random(self):
        points = torch.rand( (self.map_conf.initial_number_of_gaussians, 3) ) * 2.0 - 1.0
        points *= self.map_conf.initial_extent * self.map_conf.scene_scale
        rgbs = torch.rand( (self.map_conf.initial_number_of_gaussians, 3) )

        avg_distance_to_nearest_3_neighbors = torch.sqrt( knn(points, 4)[:, 1:] ** 2 ).mean(dim=-1)
        scales = torch.log(avg_distance_to_nearest_3_neighbors * self.map_conf.initial_scale).unsqueeze(-1).repeat(1, 3)

        N = points.shape[0]
        quats = torch.rand( (N, 4) )
        opacities = torch.logit( torch.full( (N,), self.map_conf.initial_opacity ))

        self.data= GaussianSplattingData(points, quats, scales, opacities, rgbs).to(self.map_conf.device)


    def initialize_optimizers(self,):
        # TODO fix these LRs
        self.optimizers: Dict[str, torch.optim.Optimizer] = {}
        self.optimizers['means'] = torch.optim.Adam(params=[self.data.means,], lr=0.001)
        self.optimizers['quats'] = torch.optim.Adam(params=[self.data.covar_quats,], lr=0.001)
        self.optimizers['scales'] = torch.optim.Adam(params=[self.data.covar_scales,], lr=0.001)
        self.optimizers['opacities'] = torch.optim.Adam(params=[self.data.opacities,], lr=0.001)
        self.optimizers['rgbs'] = torch.optim.Adam(params=[self.data.colors,], lr=0.001)


    def zero_grad(self,):
        for optimizer in self.optimizers.values():
            optimizer.zero_grad()


    def step(self,):
        for optimizer in self.optimizers.values():
            optimizer.step()
