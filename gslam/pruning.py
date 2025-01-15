from abc import ABC

from .map import GaussianSplattingData
from typing import Dict

from torch.optim import Optimizer
import torch


class PruningStrategy(ABC):
    def step(self, splats: GaussianSplattingData, optimizers: Dict[str, Optimizer]):
        return

    @torch.no_grad()
    def _prune_using_mask(
        self,
        splats: GaussianSplattingData,
        optimizers: Dict[str, Optimizer],
        keep_mask: torch.Tensor,
        *per_gaussian_params: Dict[str, torch.Tensor],
    ):
        if (1.0 - keep_mask.float()).sum() < 1:
            return 0
        n_pruned = keep_mask.shape[0] - keep_mask.sum()

        # follows what gsplat does to make strategies work
        # https://github.com/nerfstudio-project/gsplat/blob/795161945b37747709d4da965b226a19fdf87d3f/gsplat/strategy/ops.py#L48
        for splat_param_name, splat_param in splats.named_parameters():
            new_splat_param = torch.nn.Parameter(
                splat_param[keep_mask], requires_grad=splat_param.requires_grad
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
                    param_state[key] = value[keep_mask]
                optimizer.state[new_splat_param] = param_state
                optimizer.param_groups[i]['params'] = [new_splat_param]
            splats.__setattr__(splat_param_name, new_splat_param)

        for splat_param_name in per_gaussian_params:
            per_gaussian_params[splat_param_name] = per_gaussian_params[
                splat_param_name
            ][keep_mask]

        return n_pruned


class PruneLowOpacity(PruningStrategy):
    def __init__(self, min_opacity: float):
        self.min_opacity = min_opacity

    @torch.no_grad()
    def step(self, splats: GaussianSplattingData, optimizers: Dict[str, Optimizer]):
        opacities = torch.sigmoid(splats.opacities)
        keep_mask = opacities > self.min_opacity
        return self._prune_using_mask(splats, optimizers, keep_mask)


class PruneByVisibility(PruningStrategy):
    def __init__(
        self,
    ):
        return

    @torch.no_grad()
    def step(
        self,
        splats: GaussianSplattingData,
        optimizers: Dict[str, Optimizer],
        visibility_counts: torch.Tensor,
    ):
        raise NotImplementedError()


class PruneLargeGaussians(PruningStrategy):
    '''Prune gaussians which have a large screen-space footprint'''

    def __init__(self, min_radius: float):
        self.min_radius = min_radius

    @torch.no_grad()
    def step(
        self,
        splats: GaussianSplattingData,
        optimizers: Dict[str, Optimizer],
        radii: torch.Tensor,
    ):
        keep_mask = radii > self.min_radius
        return self._prune_using_mask(splats, optimizers, keep_mask)
