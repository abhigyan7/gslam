from abc import ABC

from .map import GaussianSplattingData
from typing import Dict

from torch.optim import Optimizer
import torch


@torch.no_grad()
def prune_using_mask(
    splats: GaussianSplattingData,
    optimizers: Dict[str, Optimizer],
    keep_mask: torch.Tensor,
    per_gaussian_params: list[torch.Tensor] = None,
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


class PruningStrategy(ABC):
    def step(self, splats: GaussianSplattingData, optimizers: Dict[str, Optimizer]):
        return


class PruneLowOpacity(PruningStrategy):
    def __init__(self, min_opacity: float):
        self.min_opacity = min_opacity

    @torch.no_grad()
    def step(
        self,
        splats: GaussianSplattingData,
        optimizers: Dict[str, Optimizer],
    ) -> torch.Tensor:
        opacities = torch.sigmoid(splats.opacities)
        prune_mask = opacities < self.min_opacity
        return prune_mask


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
    ):
        newly_added_gaussians_mask = splats.ages > (latest_kf_age - 3)  # monogs uses 3
        gaussians_that_arent_visible_enough = visibility_counts < self.min_visibility

        prune_mask = newly_added_gaussians_mask & gaussians_that_arent_visible_enough
        return prune_mask


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
    ):
        # using max because logical_or can't reduce along an axis
        prune_mask = radii > self.max_radius
        return prune_mask


class PruneIllConditionedGaussians(PruningStrategy):
    '''Prune gaussians which aren't pulling their weight'''

    def __init__(self, max_frames_thing):
        self.max_frames_thing = max_frames_thing

    @torch.no_grad()
    def step(
        self,
        splats: GaussianSplattingData,
        optimizers: Dict[str, Optimizer],
        radii: torch.Tensor,  # [C,N], maximum radius of each gaussian in this view
        n_touched: torch.Tensor,  # [C,N], how many pixels this gaussian influenced in this view
    ):
        n_views_in_which_this_gaussian_didnt_pull_its_weight = (
            (radii > 0) & (n_touched == 0)
        ).sum(dim=0)
        prune_mask = (
            n_views_in_which_this_gaussian_didnt_pull_its_weight > self.max_frames_thing
        )
        return prune_mask
