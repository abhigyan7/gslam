import torch
from dataclasses import dataclass
import time

import torch.multiprocessing as mp

from .rasterization import RasterizerConfig
from .primitives import Camera, Frame, Pose
from .utils import get_projection_matrix
from .map import GaussianSplattingData
from gsplat.rendering import rasterization
import tqdm
import logging
from copy import deepcopy

from typing import List


def tracking_loss(
        gt_img: torch.Tensor,
        rendered_img: torch.Tensor,
        ):
    return (rendered_img - gt_img).abs().sum()


@dataclass
class TrackingConfig:
    device: str = 'cuda'
    num_tracking_iters: int = 30000

    photometric_loss: str = 'l1'

    pose_lr: float = 0.003


class Frontend(mp.Process):

    def __init__(self,
                 tracking_conf: TrackingConfig,
                 rasterizer_conf: RasterizerConfig,
                 backend_queue: mp.JoinableQueue,
                 frontend_queue: mp.JoinableQueue,
                 sensor_queue: mp.JoinableQueue):

        super().__init__()
        self.tracking_config: TrackingConfig = tracking_conf
        self.rasterizer_conf: RasterizerConfig = rasterizer_conf
        self.map_queue: mp.JoinableQueue = backend_queue
        self.queue : mp.JoinableQueue[int] = frontend_queue
        self.keyframes: List[Frame] = []

        self.splats= GaussianSplattingData.new_empty_model()

        self.initialized: bool = False
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel("DEBUG")

        self.sensor_queue = sensor_queue


    def track(self,
              new_frame: Frame):
        # once initialized, we can start tracking new frames

        self.logger.warning(f"frame {new_frame.timestamp=}")
        previous_keyframe = self.keyframes[-1]

        # start with unit Rt difference?
        new_frame.pose = deepcopy(previous_keyframe.pose).cuda()

        pose_optimizer = torch.optim.Adam(new_frame.pose.parameters(), self.tracking_config.pose_lr)

        for i in (pbar := tqdm.trange(self.tracking_config.num_tracking_iters)):
            pose_optimizer.zero_grad()
            rendered_rgb, rendered_alpha, render_info = self.splats(new_frame.camera, new_frame.pose)
            loss = tracking_loss(rendered_rgb, new_frame.img)
            loss.backward()
            pose_optimizer.step()

            pbar.set_description(f"loss: {loss.item()}")

        return new_frame.pose()


    def request_initialization(self, frame: Frame):
        assert not self.initialized
        self.map_queue.put('request-init')


    def sync_maps(self):
        # TODO implement this thang
        # sync map with the backend, eventually we might need to sync only a subset of the map
        # because memory and stuff
        return


    def run(self):
        self.Ks = get_projection_matrix().to(self.tracking_config.device)

        self.logger.warning("test")

        self.requested_init = False

        tic = torch.cuda.Event(enable_timing=True)
        toc = torch.cuda.Event(enable_timing=True)

        while True:

            if not self.queue.empty():
                message_from_map = self.queue.get()
                if message_from_map == 'init-done':
                    self.logger.warning(f"Initialization successful!")
                    self.sync_maps()
                elif type(message_from_map) == tuple and message_from_map[0] == 'map-sync':
                    self.splats = message_from_map[1]
                    self.logger.warning('Map synced')
                    self.initialized = True

            if self.requested_init and not self.initialized:
                continue

            if not self.sensor_queue.empty():

                # remove an item from the queue
                frame = self.sensor_queue.get()

                if not self.initialized:
                    self.request_initialization(frame)
                    self.logger.warning(f'Requested initialization.')
                    self.requested_init = True
                    self.keyframes.append(frame)
                else:
                    self.logger.warning(f'Tracking.')
                    self.track(frame)

                # mark the job done
                self.sensor_queue.task_done()
