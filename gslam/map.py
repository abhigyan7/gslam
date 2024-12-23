from typing import Dict, Self, Tuple

import torch
from gsplat.rendering import rasterization

from .primitives import Camera, Pose
from .utils import knn, create_batch

from typing import List


# consider the implications of all these structs being torch modules
class GaussianSplattingData(torch.nn.Module):
    def __init__(
        self,
        means,  # gaussian centers
        quats,  # quaternions of covariance matrices
        scales,  # scales of covariance matrices
        opacities,  # alpha of gaussians
        colors,
    ):  # RGB values of gaussians
        super().__init__()
        self.means: torch.nn.Parameter = torch.nn.Parameter(means)
        self.quats: torch.nn.Parameter = torch.nn.Parameter(quats)
        self.scales: torch.nn.Parameter = torch.nn.Parameter(scales)
        self.opacities: torch.nn.Parameter = torch.nn.Parameter(opacities)
        self.colors: torch.nn.Parameter = torch.nn.Parameter(colors)

    def forward(
        self,
        cameras: List[Camera],
        poses: List[Pose],
        render_depth: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        render_mode = 'RGB+D' if render_depth else 'RGB'

        Ks = create_batch(cameras, lambda x: x.intrinsics)
        viewmats = create_batch(poses, lambda x: x())

        rendered_rgb, rendered_alpha, render_info = rasterization(
            means=self.means,
            quats=self.quats,
            scales=torch.exp(self.scales),
            opacities=torch.sigmoid(self.opacities),
            colors=self.colors,
            viewmats=viewmats,
            Ks=Ks,
            width=cameras[0].width,
            height=cameras[0].height,
            render_mode=render_mode,
            packed=False,
        )
        return rendered_rgb, rendered_alpha, render_info

    @staticmethod
    def empty(device: str = 'cuda'):
        return GaussianSplattingData(
            torch.tensor([], device=device),
            torch.tensor([], device=device),
            torch.tensor([], device=device),
            torch.tensor([], device=device),
            torch.tensor([], device=device),
        )

    @staticmethod
    def initialize_in_camera_frustum(
        n_gaussians: int,
        f: float,
        near: float,
        far: float,
        height: float,
        width: float,
        initial_scale: float = 0.1,
        initial_opacity: float = 0.9,
    ):
        zs = torch.rand((n_gaussians,)) * (far - near) + near
        us = (torch.rand((n_gaussians,)) - 0.5) * width
        vs = (torch.rand((n_gaussians,)) - 0.5) * height
        xs = us * zs / f
        ys = vs * zs / f

        points = torch.stack([xs, ys, zs], dim=1)

        rgbs = torch.rand((n_gaussians, 3))

        avg_distance_to_nearest_3_neighbors = torch.sqrt(
            knn(points, 4)[:, 1:] ** 2
        ).mean(dim=-1)
        scales = (
            torch.log(avg_distance_to_nearest_3_neighbors * initial_scale)
            .unsqueeze(-1)
            .repeat(1, 3)
        )

        quats = torch.rand((n_gaussians, 4))
        opacities = torch.logit(torch.full((n_gaussians,), initial_opacity))

        return GaussianSplattingData(points, quats, scales, opacities, rgbs)

    @staticmethod
    def initialize_map_random_cube(
        n_gaussians,
        initial_scale,
        initial_opacity,
        initial_extent,
    ):
        points = torch.rand((n_gaussians, 3))
        points *= initial_extent
        points[..., 2] = 20000.0
        rgbs = torch.rand((n_gaussians, 3))

        avg_distance_to_nearest_3_neighbors = torch.sqrt(
            knn(points, 4)[:, 1:] ** 2
        ).mean(dim=-1)
        scales = (
            torch.log(avg_distance_to_nearest_3_neighbors * initial_scale)
            .unsqueeze(-1)
            .repeat(1, 3)
        )

        N = points.shape[0]
        quats = torch.rand((N, 4))
        opacities = torch.logit(torch.full((N,), initial_opacity))

        return GaussianSplattingData(points, quats, scales, opacities, rgbs)

    def clone(self) -> Self:
        return GaussianSplattingData(
            self.means.clone().detach(),
            self.quats.clone().detach(),
            self.scales.clone().detach(),
            self.opacities.clone().detach(),
            self.colors.clone().detach(),
        )

    def as_dict(self):
        return {
            'means': self.means,
            'quats': self.quats,
            'scales': self.scales,
            'opacities': self.opacities,
            'colors': self.colors,
        }
