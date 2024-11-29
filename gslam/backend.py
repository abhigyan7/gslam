from dataclasses import dataclass, field
from gsplat.strategy import MCMCStrategy, DefaultStrategy
from typing import Union, Tuple

from .map import GaussianSplattingData, GaussianSplattingMap, MapConfig

import torch
import logging


class Backend(torch.multiprocessing.Process):

    def __init__(
            self,
            map_config: MapConfig,
            queue: torch.multiprocessing.JoinableQueue,
            frontend_queue: torch.multiprocessing.JoinableQueue,
            ):
        self.map_config = map_config
        self.queue: torch.multiprocessing.JoinableQueue = queue
        self.frontend_queue = queue
        self.map = GaussianSplattingMap(self.map_config)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel("DEBUG")


    def run(self):
        while True:
            if self.queue.empty():
                continue
            message = self.queue.get()
            if message == 'request-init':
                self.map.initialize_map_random()
                self.map.initialize_optimizers()
                self.frontend_queue.put('init-done')
                self.queue.task_done()
                continue
            self.logger.debug(f"{message=}")
