from typing import Self

import torch

from .primitives import Camera, Pose
from .utils import create_batch

from typing import List
from .rasterization import rasterization, RasterizationOutput


# consider the implications of all these structs being torch modules
class GaussianSplattingData(torch.nn.Module):
    _per_splat_params = [
        'means',
        'quats',
        'scales',
        'opacities',
        'colors',
        'log_uncertainties',
        'ages',
    ]

    def __init__(
        self,
        means,  # gaussian centers
        quats,  # quaternions of covariance matrices
        scales,  # scales of covariance matrices
        opacities,  # alpha of gaussians
        colors,
        log_uncertainties,  # log of something akin to variance of gaussians
        ages,  # which frame each gaussian was inserted in
    ):  # RGB values of gaussians
        super().__init__()
        self.means: torch.nn.Parameter = torch.nn.Parameter(means)
        self.quats: torch.nn.Parameter = torch.nn.Parameter(quats)
        self.scales: torch.nn.Parameter = torch.nn.Parameter(scales)
        self.opacities: torch.nn.Parameter = torch.nn.Parameter(opacities)
        self.colors: torch.nn.Parameter = torch.nn.Parameter(colors)
        self.log_uncertainties: torch.nn.Parameter = torch.nn.Parameter(
            log_uncertainties
        )
        self.ages: torch.nn.Parameter = torch.nn.Parameter(ages, requires_grad=False)

    def forward(
        self,
        cameras: List[Camera],
        poses: List[Pose],
        render_depth: bool = False,
        visibility_min_T: float = 0.5,
    ) -> RasterizationOutput:
        render_mode = 'RGB+D' if render_depth else 'RGB'

        Ks = create_batch(cameras, lambda x: x.intrinsics)
        viewmats = create_batch(poses, lambda x: x())

        return rasterization(
            means=self.means,
            quats=self.quats,
            log_scales=self.scales,
            logit_opacities=self.opacities,
            logit_colors=self.colors,
            viewmats=viewmats,
            Ks=Ks,
            width=cameras[0].width,
            height=cameras[0].height,
            render_mode=render_mode,
            packed=False,
            log_uncertainties=self.log_uncertainties,
            visibility_min_T=visibility_min_T,
            backgrounds=torch.Tensor([0.0, 0.0, 0.0])
            .tile([len(cameras), 1])
            .float()
            .to(self.means.device),
        )

    @staticmethod
    def empty(device: str = 'cuda') -> Self:
        return GaussianSplattingData(
            torch.tensor([], device=device),
            torch.tensor([], device=device),
            torch.tensor([], device=device),
            torch.tensor([], device=device),
            torch.tensor([], device=device),
            torch.tensor([], device=device),
            torch.tensor([], device=device).long(),
        )

    def clone(self) -> Self:
        return GaussianSplattingData(
            self.means.clone().detach(),
            self.quats.clone().detach(),
            self.scales.clone().detach(),
            self.opacities.clone().detach(),
            self.colors.clone().detach(),
            self.log_uncertainties.clone().detach(),
            self.ages.clone(),
        )

    def as_dict(self):
        return torch.nn.ParameterDict(
            {
                'means': self.means,
                'quats': self.quats,
                'scales': self.scales,
                'opacities': self.opacities,
                'colors': self.colors,
                'log_uncertainties': self.log_uncertainties,
                'ages': self.ages,
            }
        )
