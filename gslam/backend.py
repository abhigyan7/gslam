from dataclasses import dataclass, field
import logging
import random
import threading
from copy import deepcopy
from typing import List, Dict, Tuple

import torch
import tqdm

from gsplat.strategy import DefaultStrategy, MCMCStrategy

from .map import GaussianSplattingData
from .messages import BackendMessage, FrontendMessage
from .primitives import Frame
from .utils import create_batch


@dataclass
class MapConfig:
    densification_strategy: DefaultStrategy | MCMCStrategy = field(
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

    device: str = 'cuda:0'

    optim_window_size: int = 5


class Backend(torch.multiprocessing.Process):
    def __init__(
        self,
        map_config: MapConfig,
        queue: torch.multiprocessing.Queue,
        frontend_queue: torch.multiprocessing.Queue,
        backend_done_event: threading.Event,
        strategy: DefaultStrategy | MCMCStrategy,
    ):
        super().__init__()
        self.map_config = map_config
        self.queue: torch.multiprocessing.Queue = queue
        self.frontend_queue = frontend_queue
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel("DEBUG")
        self.keyframes: List[Frame] = []
        self.backend_done_event = backend_done_event
        self.strategy = strategy
        self.splats = GaussianSplattingData.empty()

    def optimize_map(self):
        window_size = min(len(self.keyframes), self.map_config.optim_window_size)
        to_use_strategy = window_size == self.map_config.optim_window_size
        to_use_strategy = False

        for step in (pbar := tqdm.trange(100)):
            window = random.sample(self.keyframes, window_size)
            cameras = [x.camera for x in window]
            poses = torch.nn.ModuleList([x.pose for x in window])
            gt_imgs = create_batch(window, lambda x: x.img)
            self.zero_grad_all_optimizers()

            render_colors, _render_alphas, render_info = self.splats(
                cameras,
                poses,
            )

            if to_use_strategy:
                self.strategy.step_pre_backward(
                    self.splats.as_dict(),
                    self.splat_optimizers,
                    self.strategy_state,
                    step,
                    render_info,
                )

            l1loss = (render_colors - gt_imgs).abs().mean()
            opacity_loss = self.splats.opacities.abs().mean()
            total_loss = l1loss + opacity_loss
            total_loss.backward()

            desc = f"[Mapping] loss={total_loss.item():.3f}"
            pbar.set_description(desc)

            self.step_all_optimizers()

            if to_use_strategy:
                self.strategy.step_post_backward(
                    self.splats.as_dict(),
                    self.splat_optimizers,
                    self.strategy_state,
                    step,
                    render_info,
                    # lr=0.03,
                )
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
            lr=0.001,
        )
        self.splat_optimizers['quats'] = torch.optim.Adam(
            params=[self.splats.quats],
            lr=0.001,
        )
        self.splat_optimizers['scales'] = torch.optim.Adam(
            params=[self.splats.scales],
            lr=0.001,
        )
        self.splat_optimizers['opacities'] = torch.optim.Adam(
            params=[self.splats.opacities],
            lr=0.001,
        )
        self.splat_optimizers['colors'] = torch.optim.Adam(
            params=[self.splats.colors],
            lr=0.001,
        )
        self.pose_optimizer = torch.optim.Adam(
            params=[torch.empty(0)],
            lr=0.001,
        )

    def initialize_densification(self):
        self.strategy.check_sanity(self.splats.as_dict(), self.splat_optimizers)
        self.strategy_state = self.strategy.initialize_state()
        return

    def initialize(self, frame):
        self.logger.warning('Initializing')
        frame = frame.to(self.map_config.device)
        self.keyframes = [frame]
        self.splats = GaussianSplattingData.initialize_map_random_cube(
            self.map_config.initial_number_of_gaussians,
            self.map_config.initial_scale,
            self.map_config.initial_opacity,
        ).to(self.map_config.device)
        self.initialize_optimizers()
        self.initialize_densification()
        self.optimize_map()
        self.queue.task_done()
        self.logger.warning('Initialized')
        return

    def add_keyframe(self, frame):
        self.keyframes.append(frame)
        self.optimize_map()
        self.queue.task_done()

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
                    self.sync_with_frontend()
                case None:
                    break
                case message_from_frontend:
                    self.logger.warning(f"Unknown {message_from_frontend}")

        self.backend_done_event.set()
