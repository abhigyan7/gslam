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
        for step in (pbar := tqdm.tqdm(range(1200))):
            self.map.zero_grad()

            (random_keyframe,) = random.sample(self.keyframes, 1)

            render_colors, _render_alphas, _info = self.map.data(
                random_keyframe.camera,
                random_keyframe.pose,
            )

            self.strategy.step_pre_backward(
                self.get_params_dict(),
                self.get_optimizers_dict(),
                self.strategy_state,
                step,
                _info,
            )

            l1loss = (render_colors - self.keyframes[0].img).abs().mean()
            l1loss.backward()

            desc = f"loss={l1loss.item():.3f}"
            pbar.set_description(desc)

            self.strategy.step_post_backward(
                self.get_params_dict(),
                self.get_optimizers_dict(),
                self.strategy_state,
                step,
                _info,
                lr=0.0003,
            )

            self.map.step()

        return

    def sync_with_frontend(self):
        self.frontend_queue.put(
            (BackendMessage.SYNC, self.map.data.clone(), deepcopy(self.keyframes)),
        )
        return

    def get_params_dict(
        self,
    ):
        return {
            'means': self.map.data.means,
            'scales': self.map.data.covar_scales,
            'quats': self.map.data.covar_quats,
            'rgbs': self.map.data.colors,
            'opacities': self.map.data.opacities,
        }

    def get_optimizers_dict(self):
        return self.map.optimizers

    def initialize_densification(self):
        params = self.get_params_dict()
        optimizers = self.get_optimizers_dict()
        self.strategy.check_sanity(params, optimizers)
        self.strategy_state = self.strategy.initialize_state()
        return

    def initialize(self, frame):
        self.logger.warning('Initializing')
        self.keyframes = [frame]
        self.map.initialize_map_random()
        self.map.initialize_optimizers()
        self.initialize_densification()
        self.optimize_map()
        self.queue.task_done()
        self.logger.warning('Initialized')
        return

    def add_keyframe(self, frame):
        self.logger.warning('got add keyframe')
        self.keyframes.append(frame)
        self.optimize_map()
        self.queue.task_done()
        self.logger.warning('done adding keyframe')

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
