from dataclasses import dataclass, field
from gsplat.strategy import MCMCStrategy, DefaultStrategy
from typing import Union, Tuple, List

from copy import deepcopy
import tqdm
import random
from .primitives import Frame

from .map import GaussianSplattingData, GaussianSplattingMap, MapConfig

import torch
import logging

from .utils import torch_to_pil


class Backend(torch.multiprocessing.Process):

    def __init__(
            self,
            map_config: MapConfig,
            queue: torch.multiprocessing.JoinableQueue,
            frontend_queue: torch.multiprocessing.JoinableQueue,
            ):

        super().__init__()
        self.map_config = map_config
        self.queue: torch.multiprocessing.JoinableQueue = queue
        self.frontend_queue = frontend_queue
        self.map = GaussianSplattingMap(self.map_config)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel("DEBUG")
        self.keyframes: List[Frame] = []


    def optimize_map(self):

        for _ in (pbar := tqdm.tqdm(range(15000))):
            self.map.zero_grad()

            random_keyframe, = random.sample(self.keyframes, 1)

            render_colors, _render_alphas, _info = self.map.data(
                random_keyframe.camera, random_keyframe.pose,
            )

            l1loss = (render_colors - self.keyframes[0].img).abs().sum()
            l1loss.backward()

            desc = f"loss={l1loss.item():.3f}"
            pbar.set_description(desc)

            self.map.step()

        torch_to_pil(render_colors).save(f'out_{len(self.keyframes)}.png')

        return


    def run(self):
        while True:
            if self.queue.empty():
                continue
            message = self.queue.get()

            match message:
                case ['request-init', *data]:
                    self.logger.warning('got init request')
                    frame, = data
                    self.keyframes = [frame,]
                    self.map.initialize_map_random()
                    self.map.initialize_optimizers()
                    self.optimize_map()
                    self.frontend_queue.put('init-done')
                    self.queue.task_done()
                    self.logger.warning('initialized')
                    self.sync_with_frontend()
                case ['add-keyframe', frame]:
                    self.logger.warning('got add keyframe')
                    self.keyframes.append(frame)
                    self.optimize_map()
                    self.queue.task_done()
                    self.logger.warning('done adding keyframe')
                    self.sync_with_frontend()
                case _:
                    self.logger.warning(f"Unknown {message=}")


    def sync_with_frontend(self):
        self.frontend_queue.put(
            ('map-sync', self.map.data.clone(), deepcopy(self.keyframes)),
        )