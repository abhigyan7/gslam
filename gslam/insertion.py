from abc import ABC
import math

from .map import GaussianSplattingData
from .primitives import Frame
from .utils import knn
from typing import Dict, List

from torch.optim import Optimizer
import torch

import gsplat


class InsertionStrategy(ABC):
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
        return

    @torch.no_grad()
    def _add_new_splats(
        self,
        splats: GaussianSplattingData,
        optimizers: Dict[str, Optimizer],
        new_params: Dict[str, torch.Tensor],
    ) -> int:
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

    @torch.no_grad()
    def _duplicate(
        self,
        splats: GaussianSplattingData,
        mask: torch.BoolTensor,
    ) -> Dict[str, torch.Tensor]:
        ret = dict()
        for splat_param_name, splat_param in splats.named_parameters():
            ret[splat_param_name] = splat_param[mask].clone().detach()
        return ret

    @torch.no_grad()
    def _split(
        self,
        splats: GaussianSplattingData,
        mask: torch.BoolTensor,
    ) -> GaussianSplattingData:
        ret = dict()
        for splat_param_name, splat_param in splats.named_parameters():
            ret[splat_param_name] = splat_param[mask].clone().detach()

        covars, _precis = gsplat.quat_scale_to_covar_preci(
            ret['quats'],
            torch.exp(ret['scales']),
        )
        noise = torch.randn_like(ret['means'])
        noise = torch.einsum("bij,bj->bi", covars, noise)
        ret['means'].add_(noise)
        # scales *= 1/1.6 in log space
        ret['scales'].add_(-math.log(1.6))
        return ret


class InsertFromDepthMap(InsertionStrategy):
    def __init__(
        self,
        depth_variance: float,
        no_depth_variance: float,
        min_alpha_for_depth: float,
        initial_opacity: float,
        initial_beta: float,
    ):
        self.depth_variance = depth_variance
        self.no_depth_variance = no_depth_variance
        self.min_alpha_for_depth = min_alpha_for_depth
        self.initial_opacity = initial_opacity
        self.initial_beta = initial_beta

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
        depths = meta['depths'][0, ...]
        alphas = rendered_alphas[0, ..., 0]

        device = depths.device
        valid_depth_region = torch.logical_and(
            alphas > self.min_alpha_for_depth, depths > 0
        )

        median_depth = depths[valid_depth_region].median()

        random_depths = torch.randn_like(depths, device=device)

        depths[valid_depth_region] += (
            random_depths[valid_depth_region] * self.depth_variance
        )
        depths[~valid_depth_region] = median_depth
        depths[~valid_depth_region] += (
            random_depths[~valid_depth_region] * self.depth_variance
        )

        pixel_indices_where_depth_is_valid = torch.nonzero(
            valid_depth_region.reshape(-1)
        )
        pixel_indices_where_depth_is_not_valid = torch.nonzero(
            ~valid_depth_region.reshape(-1)
        )
        picks = []

        if pixel_indices_where_depth_is_not_valid.shape[0] > 0:
            picks.append(
                (
                    pixel_indices_where_depth_is_not_valid[
                        torch.randint(
                            pixel_indices_where_depth_is_not_valid.shape[0],
                            [
                                N,
                            ],
                        )
                    ]
                ).reshape(-1),
            )
        if pixel_indices_where_depth_is_valid.shape[0] > 0:
            picks.append(
                (
                    pixel_indices_where_depth_is_valid[
                        torch.randint(
                            pixel_indices_where_depth_is_valid.shape[0],
                            [
                                N,
                            ],
                        )
                    ]
                ).reshape(-1),
            )

        picks = torch.cat(picks)

        N = picks.shape[0]

        means = frame.camera.backproject(depths)
        colors = frame.img.reshape([-1, 3])
        means = means[picks]
        colors = colors[picks]

        c2w = torch.linalg.inv(frame.pose())
        R = c2w[:3, :3]
        t = c2w[:3, 3]
        means = means @ R.t() + t

        if splats.scales.size().numel() > 0:
            scales = torch.exp(splats.scales).mean(dim=0).tile([N, 1])
        else:
            print("Found no scales")
            avg_distance_to_nearest_3_neighbors = torch.sqrt(
                knn(means, 4)[:, 1:] ** 2
            ).mean(dim=-1)
            scales = avg_distance_to_nearest_3_neighbors.unsqueeze(-1).repeat(1, 3)

        new_params = {
            'means': means,
            'scales': torch.log(scales),
            'colors': torch.logit(colors),
            'opacities': torch.logit(
                torch.full((N,), self.initial_opacity, device=device)
            ),
            'quats': torch.rand((N, 4), device=device),
            'betas': torch.log(torch.full((N,), self.initial_beta, device=device)),
        }

        self._add_new_splats(splats, optimizers, new_params)


class InsertUsingImagePlaneGradients(InsertionStrategy):
    '''
    Insertion strategy that the original 3DGS paper uses
    '''

    def __init__(
        self,
        grow_grad2d: float,
        grow_scale3d: float,
    ):
        self.grow_grad2d = grow_grad2d
        self.grow_scale3d = grow_scale3d

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
        grads = meta['means2d'].grad.clone()

        # normalize grads by image size
        grads[..., 0] *= meta["width"] / 2.0 * meta["n_cameras"]
        grads[..., 1] *= meta["height"] / 2.0 * meta["n_cameras"]

        grads = grads.norm(dim=-1).mean(dim=0)

        has_high_image_plane_grad = grads > self.grow_grad2d
        is_small = torch.exp(splats.scales).max(dim=-1).values <= self.grow_scale3d

        to_duplicate = has_high_image_plane_grad & is_small

        is_large = ~is_small
        to_split = has_high_image_plane_grad & is_large

        num_split = to_split.sum().detach()
        num_duplicate = to_duplicate.sum().detach()

        if num_duplicate > 0:
            duplicated_splats = self._duplicate(splats, mask=to_duplicate)
        if num_split > 0:
            split_splats = self._split(splats, mask=to_split)

        if num_duplicate > 0:
            self._add_new_splats(splats, optimizers, duplicated_splats)
        if num_split > 0:
            self._add_new_splats(splats, optimizers, split_splats)

        return num_duplicate, num_split


class SequentialInsertion(InsertionStrategy):
    def __init__(self, strategies: List[InsertionStrategy]):
        self.strategies = strategies

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
        for strategy in self.strategies:
            strategy.step(
                splats,
                optimizers,
                rendered_colors,
                rendered_alphas,
                meta,
                frame,
                N,
            )
