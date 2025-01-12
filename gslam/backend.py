from dataclasses import dataclass
import logging
import random
import threading
from copy import deepcopy
from typing import List, Dict, Tuple

import torch
import tqdm

from .map import GaussianSplattingData
from .messages import BackendMessage, FrontendMessage
from .primitives import Frame
from .pruning import PruneLowOpacity
from .insertion import InsertFromDepthMap, InsertUsingImagePlaneGradients
from .utils import create_batch


@dataclass
class MapConfig:
    isotropic_regularization_weight: float = 10.0
    opacity_regularization_weight: float = 0.000005

    pose_optim_lr_translation: float = 0.001
    pose_optim_lr_rotation: float = 0.003
    pose_optimization_regularization = 1e-6

    # 3dgs schedules means_lr, might need to look into this
    means_lr: float = 0.00016
    opacity_lr: float = 0.025
    scale_lr: float = 0.005
    color_lr: float = 0.005
    quat_lr: float = 0.005
    beta_lr: float = 0.005

    # background rgb
    background_color: Tuple[float, 3] = (0.0, 0.0, 0.0)

    initialization_type: str = 'random'
    initial_number_of_gaussians: int = 10_000
    initial_opacity: float = 0.9
    initial_scale: float = 1.0
    initial_extent: float = 1.0
    initial_beta: float = 0.3

    device: str = 'cuda:0'

    optim_window_last_n_keyframes: int = 5
    optim_window_random_keyframes: int = 5

    num_iters_mapping: int = 200

    opacity_pruning_threshold: float = 0.6


class Backend(torch.multiprocessing.Process):
    def __init__(
        self,
        map_config: MapConfig,
        queue: torch.multiprocessing.Queue,
        frontend_queue: torch.multiprocessing.Queue,
        backend_done_event: threading.Event,
    ):
        super().__init__()
        self.map_config = map_config
        self.queue: torch.multiprocessing.Queue = queue
        self.frontend_queue = frontend_queue
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel("DEBUG")
        self.keyframes: List[Frame] = []
        self.backend_done_event = backend_done_event
        self.splats = GaussianSplattingData.empty()
        self.pruning = PruneLowOpacity(self.map_config.opacity_pruning_threshold)
        self.insertion_depth_map = InsertFromDepthMap(
            0.2, 0.5, 0.1, self.map_config.initial_opacity, self.map_config.initial_beta
        )
        self.insertion_3dgs = InsertUsingImagePlaneGradients(
            0.0002,
            0.01,
        )

    def optimization_window(self):
        window_size_total = (
            self.map_config.optim_window_last_n_keyframes
            + self.map_config.optim_window_random_keyframes
        )
        n_keyframes_total = len(self.keyframes)
        n_last_keyframes_to_choose = min(
            n_keyframes_total, self.map_config.optim_window_last_n_keyframes
        )
        n_random_keyframes_to_choose = min(
            0, window_size_total - n_last_keyframes_to_choose
        )

        window = []

        window = self.keyframes[-n_last_keyframes_to_choose:]
        window.extend(
            random.sample(
                self.keyframes[:-n_last_keyframes_to_choose],
                n_random_keyframes_to_choose,
            )
        )
        return window

    def optimize_map(self):
        for step in (pbar := tqdm.trange(self.map_config.num_iters_mapping)):
            window = self.optimization_window()
            cameras = [x.camera for x in window]
            poses = torch.nn.ModuleList([x.pose for x in window])
            gt_imgs = create_batch(window, lambda x: x.img)
            self.zero_grad_all_optimizers()

            render_colors, render_alphas, render_info = self.splats(
                cameras,
                poses,
            )

            photometric_loss = (render_colors - gt_imgs).abs().mean()
            mean_scales = self.splats.scales.mean(dim=1, keepdim=True).detach()
            isotropic_loss = (self.splats.scales - mean_scales).abs().mean()
            total_loss = (
                photometric_loss
                + self.map_config.isotropic_regularization_weight * isotropic_loss
                + self.map_config.opacity_regularization_weight
                * self.splats.opacities.mean()
            )

            render_info['means2d'].retain_grad()

            total_loss.backward()

            if step == (self.map_config.num_iters_mapping // 2):
                self.insertion_3dgs.step(
                    self.splats,
                    self.splat_optimizers,
                    render_colors,
                    render_alphas,
                    render_info,
                    None,
                    None,
                )

            desc = f"[Mapping] loss={total_loss.item():.3f}, n_splats={self.splats.means.shape[0]:07} opacmin={torch.sigmoid(self.splats.opacities).min():.2f}"
            pbar.set_description(desc)

            self.step_all_optimizers()

            self.pruning.step(self.splats, self.splat_optimizers)

        render_colors, _render_alphas, render_info = self.splats(
            [self.keyframes[-1].camera],
            [self.keyframes[-1].pose],
        )

        self.keyframes[-1].visible_gaussians = render_info['radii'][0] > 0
        return

    def sync_with_frontend(self):
        self.frontend_queue.put(
            (BackendMessage.SYNC, self.splats.clone(), deepcopy(self.keyframes)),
        )
        return

    def zero_grad_all_optimizers(self):
        for optimizer in self.splat_optimizers.values():
            optimizer.zero_grad()
        self.pose_optimizer.zero_grad()

    def step_all_optimizers(self):
        for optimizer in self.splat_optimizers.values():
            optimizer.step()
        self.pose_optimizer.step()

    def initialize_optimizers(self):
        # TODO fix these LRs
        self.splat_optimizers: Dict[str, torch.optim.Optimizer] = {}
        self.splat_optimizers['means'] = torch.optim.Adam(
            params=[self.splats.means],
            lr=self.map_config.means_lr,
        )
        self.splat_optimizers['quats'] = torch.optim.Adam(
            params=[self.splats.quats],
            lr=self.map_config.quat_lr,
        )
        self.splat_optimizers['scales'] = torch.optim.Adam(
            params=[self.splats.scales],
            lr=self.map_config.scale_lr,
        )
        self.splat_optimizers['opacities'] = torch.optim.Adam(
            params=[self.splats.opacities],
            lr=self.map_config.opacity_lr,
        )
        self.splat_optimizers['colors'] = torch.optim.Adam(
            params=[self.splats.colors],
            lr=self.map_config.color_lr,
        )
        self.splat_optimizers['betas'] = torch.optim.Adam(
            params=[self.splats.betas],
            lr=self.map_config.beta_lr,
        )
        self.pose_optimizer = torch.optim.Adam(
            params=[torch.empty(0)],
            lr=0.001,
        )

    def initialize(self, frame: Frame):
        self.logger.warning('Initializing')
        frame = frame.to(self.map_config.device)
        self.keyframes = [frame]

        self.splats = GaussianSplattingData.empty().to(self.map_config.device)
        self.initialize_optimizers()

        # f = frame.camera.intrinsics[0,0].item()

        self.insertion_depth_map.step(
            self.splats,
            self.splat_optimizers,
            None,
            torch.ones_like(frame.img[..., 0], device=self.map_config.device).unsqueeze(
                -1
            ),
            {
                'depths': torch.ones(
                    1, *frame.img.shape[:2], device=self.map_config.device
                )
            },
            frame,
            10000,
        )

        self.optimize_map()
        self.queue.task_done()
        self.logger.warning('Initialized')
        return

    def add_keyframe(self, frame: Frame):
        with torch.no_grad():
            render_colors, _render_alphas, render_info = self.splats(
                [frame.camera],
                [frame.pose],
                render_depth=True,
            )
        self.insertion_depth_map.step(
            self.splats,
            self.splat_optimizers,
            render_colors.squeeze(0),
            _render_alphas.squeeze(0),
            render_info,
            frame,
            N=5000,
        )

        self.keyframes.append(frame)
        self.pose_optimizer.add_param_group(
            {'params': frame.pose.dt, 'lr': self.map_config.pose_optim_lr_translation}
        )
        self.pose_optimizer.add_param_group(
            {'params': frame.pose.dR, 'lr': self.map_config.pose_optim_lr_rotation}
        )

    def run(self):
        while True:
            if self.queue.empty():
                continue
            match self.queue.get():
                case [FrontendMessage.REQUEST_INITIALIZE, frame]:
                    self.initialize(frame)
                    self.frontend_queue.put([BackendMessage.SIGNAL_INITIALIZED])
                    self.sync_with_frontend()
                case [FrontendMessage.ADD_KEYFRAME, frame]:
                    self.add_keyframe(frame)
                    self.optimize_map()
                    self.sync_with_frontend()
                case None:
                    break
                case message_from_frontend:
                    self.logger.warning(f"Unknown {message_from_frontend}")

        self.backend_done_event.set()
