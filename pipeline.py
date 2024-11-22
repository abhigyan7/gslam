from typing import Tuple, Dict, Union
from dataclasses import dataclass, field

import tqdm
import torch
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy

from tum import TumRGB
from utils import knn, torch_to_pil


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


def main():

    tum_dataset = TumRGB('datasets/tum/rgbd_dataset_freiburg1_desk')
    gt_img, camtoworld, timestamps = tum_dataset[0]
    gt_img = torch.FloatTensor(gt_img).cuda().unsqueeze(0)
    camtoworld = torch.eye(4)
    _, height, width, _ = gt_img.shape
    viewmats = torch.linalg.inv(torch.FloatTensor(camtoworld).unsqueeze(0).cuda())
    Ks = torch.FloatTensor([
        [525.0, 0.0, 319.5],
        [0.0, 525.5, 239.5],
        [0.0,   0.0,   0.0],
    ]).unsqueeze(0).cuda()

    splats, optimizers = initialize_map(MapConfig())
    print("Model initialized. Number of GS:", len(splats["means"]))

    render_colors, render_alphas, info = rasterization(
        means = splats['means'],
        quats = splats['quats'],
        scales = splats['scales'],
        opacities = splats['opacities'],
        colors = splats['colors'],
        viewmats = viewmats,
        Ks = Ks,
        width = width,
        height = height,
    )

    for _ in (pbar := tqdm.tqdm(range(10000))):
        for _,optimizer in optimizers.items():
            optimizer.zero_grad()

        render_colors, render_alphas, info = rasterization(
            means = splats['means'],
            quats = splats['quats'],
            scales = splats['scales'],
            opacities = splats['opacities'],
            colors = splats['colors'],
            viewmats = viewmats,
            Ks = Ks,
            width = width,
            height = height,
        )

        l1loss = (render_colors - gt_img).abs().sum()
        l1loss.backward()

        desc = f"loss={l1loss.item():.3f}"
        pbar.set_description(desc)

        for optimizer in optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=False)

    img = torch_to_pil(render_colors)
    img.save('img.png')
    img = torch_to_pil(gt_img)
    img.save('img_gt.png')

    print(f'{render_colors.max()=}')
    print(f'{render_alphas.max()=}')


if __name__ == '__main__':
    main()