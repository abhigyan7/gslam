import logging
import random
import threading
from copy import deepcopy
from typing import List

import torch
import tqdm

from gsplat.strategy import DefaultStrategy, MCMCStrategy

from .map import GaussianSplattingMap, MapConfig
from .messages import BackendMessage, FrontendMessage
from .primitives import Frame
from .utils import create_batch


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
        self.map = GaussianSplattingMap(self.map_config)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel("DEBUG")
        self.keyframes: List[Frame] = []
        self.backend_done_event = backend_done_event
        self.strategy = strategy

    def optimize_map(self):
        for step in (pbar := tqdm.tqdm(range(100))):
            window_size = min(len(self.keyframes), self.map_config.optim_window_size)
            window = random.sample(self.keyframes, window_size)
            cameras = [x.camera for x in window]
            poses = torch.nn.ModuleList([x.pose for x in window])

            to_use_strategy = window_size == self.map_config.optim_window_size

            gt_imgs = create_batch(window, lambda x: x.img)

            pose_optimizer = torch.optim.SGD(poses.parameters())
            for i in range(100):
                self.map.zero_grad()
                pose_optimizer.zero_grad()

                render_colors, _render_alphas, render_info = self.map.data(
                    cameras,
                    poses,
                )

                if to_use_strategy:
                    self.strategy.step_pre_backward(
                        self.map.data.as_dict(),
                        self.map.optimizers,
                        self.strategy_state,
                        step,
                        render_info,
                    )

                l1loss = (render_colors - gt_imgs).abs().mean()
                opacity_loss = self.map.data.opacities.abs().mean()
                total_loss = l1loss + opacity_loss
                total_loss.backward()

                desc = f"loss={total_loss.item():.3f}"
                pbar.set_description(desc)

                self.map.step()
                pose_optimizer.step()

                if to_use_strategy:
                    self.strategy.step_post_backward(
                        self.map.data.as_dict(),
                        self.map.optimizers,
                        self.strategy_state,
                        step,
                        render_info,
                        # lr=0.03,
                    )

        return

    def sync_with_frontend(self):
        self.frontend_queue.put(
            (BackendMessage.SYNC, self.map.data.clone(), deepcopy(self.keyframes)),
        )
        return

    def initialize_densification(self):
        params = self.map.data.as_dict()
        optimizers = self.map.optimizers
        self.strategy.check_sanity(params, optimizers)
        self.strategy_state = self.strategy.initialize_state()
        return

    def initialize(self, frame):
        self.logger.warning('Initializing')
        frame = frame.to(self.map_config.device)
        self.keyframes = [frame]
        self.map.initialize_map_random()
        self.map.initialize_optimizers()
        self.initialize_densification()
        self.optimize_map()
        self.queue.task_done()
        self.logger.warning('Initialized')
        return

    def add_keyframe(self, frame):
        self.keyframes.append(frame)
        self.optimize_map()
        self.queue.task_done()

    # @rr.shutdown_at_exit
    def run(self):
        # rr.init('gslam', recording_id='gslam_1')
        # rr.save('runs/rr.rrd')
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
