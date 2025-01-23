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
        per_gaussian_params: list[torch.Tensor],
    ):
        if keep_mask.sum() == 0:
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
            if splat_param_name not in optimizers:
                splats.__setattr__(splat_param_name, new_splat_param)
                continue

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

        if per_gaussian_params is None:
            return n_pruned
        for i, splat_param in enumerate(per_gaussian_params):
            per_gaussian_params[i] = splat_param[keep_mask]

        return n_pruned


class PruneLowOpacity(PruningStrategy):
    def __init__(self, min_opacity: float):
        self.min_opacity = min_opacity

    @torch.no_grad()
    def step(
        self,
        splats: GaussianSplattingData,
        optimizers: Dict[str, Optimizer],
        per_gaussian_params: list[torch.Tensor] = None,
    ):
        opacities = torch.sigmoid(splats.opacities)
        keep_mask = opacities > self.min_opacity
        return self._prune_using_mask(
            splats, optimizers, keep_mask, per_gaussian_params
        )


class PruneByVisibility(PruningStrategy):
    def __init__(
        self,
        window_size,
        min_visibility,
    ):
        self.window_size = window_size
        self.min_visibility = min_visibility
        return

    @torch.no_grad()
    def step(
        self,
        splats: GaussianSplattingData,
        optimizers: Dict[str, Optimizer],
        visibility_counts: torch.Tensor,  # [N,]
        latest_kf_age: int,
        per_gaussian_params: list[torch.Tensor] = None,
    ):
        newly_added_gaussians_mask = splats.ages > (latest_kf_age - self.window_size)
        gaussians_that_arent_visible_enough = visibility_counts < self.min_visibility

        remove_mask = newly_added_gaussians_mask & gaussians_that_arent_visible_enough
        keep_mask = torch.logical_not(remove_mask)
        n_pruned = self._prune_using_mask(
            splats, optimizers, keep_mask, per_gaussian_params
        )
        return n_pruned


class PruneLargeGaussians(PruningStrategy):
    '''Prune gaussians which have a large screen-space footprint'''

    def __init__(self, max_radius: float):
        self.max_radius = max_radius

    @torch.no_grad()
    def step(
        self,
        splats: GaussianSplattingData,
        optimizers: Dict[str, Optimizer],
        radii: torch.Tensor,  # [N,], maximum radius of each gaussian across all rendered views
        per_gaussian_params: list[torch.Tensor] = None,
    ):
        # using max because logical_or can't reduce along an axis
        keep_mask = radii < self.max_radius
        n_pruned = self._prune_using_mask(
            splats, optimizers, keep_mask, per_gaussian_params
        )
        return n_pruned
