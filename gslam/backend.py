import logging
import random
from copy import deepcopy
from typing import List

import torch
import tqdm

from .map import GaussianSplattingMap, MapConfig
from .messages import BackendMessage, FrontendMessage
from .primitives import Frame
from .utils import q_get, torch_to_pil

import rerun as rr


class Backend(torch.multiprocessing.Process):
    def __init__(
        self,
        map_config: MapConfig,
        queue: torch.multiprocessing.Queue,
        frontend_queue: torch.multiprocessing.Queue,
    ):
        super().__init__()
        self.map_config = map_config
        self.queue: torch.multiprocessing.Queue = queue
        self.frontend_queue = frontend_queue
        self.map = GaussianSplattingMap(self.map_config)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel("DEBUG")
        self.keyframes: List[Frame] = []

    def optimize_map(self):
        for _ in (pbar := tqdm.tqdm(range(1500))):
            self.map.zero_grad()

            (random_keyframe,) = random.sample(self.keyframes, 1)

            render_colors, _render_alphas, _info = self.map.data(
                random_keyframe.camera,
                random_keyframe.pose,
            )

            l1loss = (render_colors - self.keyframes[0].img).abs().sum()
            l1loss.backward()

            rr.log(
                'backend/global_optim/loss',
                rr.Scalar(l1loss.item()),
            )

            desc = f"loss={l1loss.item():.3f}"
            pbar.set_description(desc)

            self.map.step()

        torch_to_pil(render_colors).save(f'out_{len(self.keyframes)}.png')

        return

    def sync_with_frontend(self):
        self.frontend_queue.put(
            (BackendMessage.SYNC, self.map.data.clone(), deepcopy(self.keyframes)),
        )
        return

    def initialize(self, frame):
        self.logger.warning('Initializing')
        self.keyframes = [frame]
        self.map.initialize_map_random()
        self.map.initialize_optimizers()
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
            match q_get(self.queue):
                case [FrontendMessage.REQUEST_INITIALIZE, frame]:
                    self.initialize(frame)
                    self.frontend_queue.put([BackendMessage.SIGNAL_INITIALIZED])
                    self.sync_with_frontend()
                case [FrontendMessage.ADD_KEYFRAME, frame]:
                    self.sync_with_frontend()
                case None:
                    pass
                case message_from_frontend:
                    self.logger.warning(f"Unknown {message_from_frontend}")
