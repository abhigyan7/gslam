from abc import ABC
import math

from .map import GaussianSplattingData
from .primitives import Frame
from .rasterization import RasterizationOutput
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
        rasterization_output: RasterizationOutput,
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
                    # append zero here
                    new_value = torch.zeros((N, *value.shape[1:]), device=value.device)
                    param_state[key] = torch.cat([value, new_value])
                optimizer.state[new_splat_param] = param_state
                optimizer.param_groups[i]['params'] = [new_splat_param]
            splats.__setattr__(splat_param_name, new_splat_param)

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
        insert_in_regions_with_depth: bool = True,
    ):
        self.depth_variance = depth_variance
        self.no_depth_variance = no_depth_variance
        self.min_alpha_for_depth = min_alpha_for_depth
        self.initial_opacity = initial_opacity
        self.insert_in_regions_with_depth = insert_in_regions_with_depth

    @torch.no_grad()
    def step(
        self,
        splats: GaussianSplattingData,
        optimizers: Dict[str, Optimizer],
        rasterization_output: RasterizationOutput,
        frame: Frame,
        N: int,
    ):
        depths = rasterization_output.depthmaps[0, ...]
        alphas = rasterization_output.alphas[0, ..., 0]

        device = depths.device
        valid_depth_region = torch.logical_and(
            alphas > self.min_alpha_for_depth, depths > 0
        )

        n_valid_depth_pixels = valid_depth_region.sum().detach().item()
        n_invalid_depth_pixels = depths.shape.numel() - n_valid_depth_pixels

        # prefer to add N nsplats in the region where we don't have geometry already
        n_invalid_depth_splats = min(N, n_invalid_depth_pixels)
        n_valid_depth_splats = max(
            0, min(N - n_invalid_depth_splats, n_valid_depth_pixels)
        )

        if valid_depth_region.any():
            median_depth = depths[valid_depth_region].median()
        else:
            median_depth = depths.median()

        random_depths = torch.randn_like(depths, device=device)

        depths[valid_depth_region] += (
            random_depths[valid_depth_region] * self.depth_variance
        )
        depths[~valid_depth_region] = median_depth
        depths[~valid_depth_region] += (
            random_depths[~valid_depth_region] * self.no_depth_variance
        )

        # TODO this is kind of arbritary, maybe we should
        # tie this with the nearplane of the camera
        depths.clamp_min_(0.1)

        pixel_indices_where_depth_is_valid = torch.nonzero(
            valid_depth_region.reshape(-1)
        )
        pixel_indices_where_depth_is_not_valid = torch.nonzero(
            ~valid_depth_region.reshape(-1)
        )
        picks = []

        if n_invalid_depth_splats > 0:
            picks.append(
                (
                    pixel_indices_where_depth_is_not_valid[
                        torch.randint(
                            pixel_indices_where_depth_is_not_valid.shape[0],
                            [
                                n_invalid_depth_splats,
                            ],
                        )
                    ]
                ).reshape(-1),
            )
        if self.insert_in_regions_with_depth and (n_valid_depth_splats > 0):
            picks.append(
                (
                    pixel_indices_where_depth_is_valid[
                        torch.randint(
                            pixel_indices_where_depth_is_valid.shape[0],
                            [
                                n_valid_depth_splats,
                            ],
                        )
                    ]
                ).reshape(-1),
            )

        if len(picks) == 0:
            return 0
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
            scales = torch.exp(splats.scales).median(dim=0)[0].tile([N, 1])
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
            # 'log_uncertainties': torch.rand(N, device=device),
            'log_uncertainties': torch.ones((N,), device=device),
            'ages': torch.full((N,), frame.index, device=device).long(),
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
        rasterization_output: RasterizationOutput,
        frame: Frame,
        N: int,
    ):
        grads = rasterization_output.means2d.grad.clone()

        # normalize grads by image size
        grads[..., 0] *= (
            rasterization_output.width / 2.0 * rasterization_output.n_cameras
        )
        grads[..., 1] *= (
            rasterization_output.height / 2.0 * rasterization_output.n_cameras
        )

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
            duplicated_splats['log_uncertainties'].fill_(1.0)
        if num_split > 0:
            split_splats = self._split(splats, mask=to_split)
            split_splats['log_uncertainties'].fill_(1.0)

        if num_duplicate > 0:
            self._add_new_splats(splats, optimizers, duplicated_splats)
        if num_split > 0:
            self._add_new_splats(splats, optimizers, split_splats)

        rasterization_output.radii = torch.cat(
            [
                rasterization_output.radii,
                torch.zeros(
                    [
                        rasterization_output.radii.shape[0],
                        num_duplicate + num_split,
                    ],
                    device=rasterization_output.radii.device,
                ),
            ],
            dim=1,
        )

        return num_duplicate, num_split


class SequentialInsertion(InsertionStrategy):
    def __init__(self, strategies: List[InsertionStrategy]):
        self.strategies = strategies

    def step(
        self,
        splats: GaussianSplattingData,
        optimizers: Dict[str, Optimizer],
        rasterization_output: RasterizationOutput,
        frame: Frame,
        N: int,
    ):
        for strategy in self.strategies:
            strategy.step(
                splats,
                optimizers,
                rasterization_output,
                frame,
                N,
            )
