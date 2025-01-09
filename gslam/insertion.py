from abc import ABC

from .map import GaussianSplattingData
from .primitives import Frame
from .utils import knn
from typing import Dict

from torch.optim import Optimizer
import torch


class InsertionStrategy(ABC):
    def step(self, splats: GaussianSplattingData, optimizers: Dict[str, Optimizer]):
        return

    @torch.no_grad()
    def _add_new_splats(
        self,
        splats: GaussianSplattingData,
        optimizers: Dict[str, Optimizer],
        new_params: Dict[str, torch.Tensor],
    ):
        N, _ = new_params['means'].shape

        # follows what gsplat does to make strategies work
        # https://github.com/nerfstudio-project/gsplat/blob/795161945b37747709d4da965b226a19fdf87d3f/gsplat/strategy/ops.py#L48
        for splat_param_name, splat_param in splats.named_parameters():
            new_splat_param = torch.nn.Parameter(
                torch.cat([splat_param, new_params[splat_param_name]], dim=0),
                requires_grad=splat_param.requires_grad,
            )

            # might have to allow some parameters to not have optimizers
            # for stuff like visibility counts, they aren't trainable
            # but we might need them for regularization
            optimizer = optimizers[splat_param_name]
            for i in range(len(optimizer.param_groups)):
                param_state = optimizer.state[splat_param]
                optimizer.state.pop(splat_param)
                for key in param_state.keys():
                    if key == 'step':
                        continue
                    value = param_state[key]
                    # append zero here
                    new_value = torch.zeros((N, *value.shape[1:]), device=value.device)
                    param_state[key] = torch.cat([value, new_value])
                optimizer.state[new_splat_param] = param_state
                optimizer.param_groups[i]['params'] = [new_splat_param]
            splats.__setattr__(splat_param_name, new_splat_param)
        print(f"Added {N} new gaussians.")
        return N


class InsertFromDepthMap(InsertionStrategy):
    def __init__(
        self,
        depth_variance: float,
        no_depth_variance: float,
        min_alpha_for_depth: float,
        initial_opacity: float,
    ):
        self.depth_variance = depth_variance
        self.no_depth_variance = no_depth_variance
        self.min_alpha_for_depth = min_alpha_for_depth
        self.initial_opacity = initial_opacity

    @torch.no_grad()
    def step(
        self,
        splats: GaussianSplattingData,
        optimizers: Dict[str, Optimizer],
        rendered_colors: torch.Tensor,
        rendered_alphas: torch.Tensor,
        meta: Dict,
        frame: Frame,
        N: int,
    ):
        depths = rendered_colors[..., -1]
        device = depths.device
        valid_depth_region = rendered_alphas[..., 0] > self.min_alpha_for_depth

        median_depth = depths[valid_depth_region].median()

        random_depths = torch.randn_like(depths)

        depths[valid_depth_region] += (
            random_depths[valid_depth_region] * self.depth_variance
        )
        depths[~valid_depth_region] = median_depth
        depths[~valid_depth_region] += (
            random_depths[~valid_depth_region] * self.depth_variance
        )

        means = frame.camera.backproject(depths)
        colors = frame.img.reshape([-1, 3])
        random_picks = torch.randint(
            means.shape[0],
            [
                N,
            ],
        )
        means = means[random_picks]
        colors = colors[random_picks]

        if splats.scales.size().numel() > 0:
            scales = splats.scales.mean(dim=0).tile([N, 1])
        else:
            print("Found no scales")
            avg_distance_to_nearest_3_neighbors = torch.sqrt(
                knn(means, 4)[:, 1:] ** 2
            ).mean(dim=-1)
            scales = (
                torch.log(avg_distance_to_nearest_3_neighbors)
                .unsqueeze(-1)
                .repeat(1, 3)
            )

        new_params = {
            'means': means,
            'scales': scales,
            'colors': colors,
            'opacities': torch.logit(
                torch.full((N,), self.initial_opacity, device=device)
            ),
            'quats': torch.rand((N, 4), device=device),
        }

        self._add_new_splats(splats, optimizers, new_params)
