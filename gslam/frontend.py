import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import List

import torch
import torch.multiprocessing as mp
import tqdm

from .map import GaussianSplattingData
from .messages import BackendMessage, FrontendMessage
from .primitives import Frame
from .rasterization import RasterizerConfig
from .utils import get_projection_matrix, q_get, torch_to_pil


def tracking_loss(
    gt_img: torch.Tensor,
    rendered_img: torch.Tensor,
):
    return (rendered_img - gt_img).abs().sum()


@dataclass
class TrackingConfig:
    device: str = 'cuda'
    num_tracking_iters: int = 3000

    photometric_loss: str = 'l1'

    pose_lr: float = 0.003


class Frontend(mp.Process):
    def __init__(
        self,
        tracking_conf: TrackingConfig,
        rasterizer_conf: RasterizerConfig,
        backend_queue: mp.Queue,
        frontend_queue: mp.Queue,
        sensor_queue: mp.Queue,
    ):
        super().__init__()
        self.tracking_config: TrackingConfig = tracking_conf
        self.rasterizer_conf: RasterizerConfig = rasterizer_conf
        self.map_queue: mp.Queue = backend_queue
        self.queue: mp.Queue[int] = frontend_queue
        self.keyframes: List[Frame] = []

        self.splats = GaussianSplattingData.new_empty_model()

        self.requested_init = False
        self.initialized: bool = False
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel("DEBUG")

        self.sensor_queue = sensor_queue

    def track(self, new_frame: Frame):
        self.logger.warning(f" Tracking frame, ts={new_frame.timestamp}")
        previous_keyframe = self.keyframes[-1]

        # start with unit Rt difference?
        new_frame.pose = deepcopy(previous_keyframe.pose).cuda()

        pose_optimizer = torch.optim.Adam(
            new_frame.pose.parameters(), self.tracking_config.pose_lr
        )

        for i in (pbar := tqdm.trange(self.tracking_config.num_tracking_iters)):
            pose_optimizer.zero_grad()
            rendered_rgb, rendered_alpha, render_info = self.splats(
                new_frame.camera, new_frame.pose
            )
            loss = tracking_loss(rendered_rgb, new_frame.img)
            loss.backward()
            pose_optimizer.step()

            pbar.set_description(f"loss: {loss.item()}")

        torch_to_pil(rendered_rgb).save(f'tracking_{len(self.keyframes)}.png')

        return new_frame.pose()

    def request_initialization(self, frame: Frame):
        self.logger.warning('Requested initialization.')
        assert not self.initialized
        self.map_queue.put([FrontendMessage.REQUEST_INITIALIZE, deepcopy(frame)])

    def add_keyframe(self, frame: Frame):
        assert self.initialized
        self.map_queue.put([FrontendMessage.ADD_KEYFRAME, deepcopy(frame)])

    def sync_maps(self, splats, keyframes):
        self.logger.warning('Map synced')
        self.splats, self.keyframes = splats, keyframes
        return

    def run(self):
        self.Ks = get_projection_matrix().to(self.tracking_config.device)

        self.logger.warning("test")

        while True:
            match q_get(self.queue):
                case [BackendMessage.SIGNAL_INITIALIZED]:
                    self.logger.warning("Initialization successful!")
                    self.initialized = True
                case [BackendMessage.Sync, map_data, keyframes]:
                    self.sync_maps(map_data, keyframes)
                case None:
                    continue
                case message_from_map:
                    self.logger.warning(f"Unknown {message_from_map=}")

            if self.requested_init and not self.initialized:
                continue

            frame = q_get(self.sensor_queue)
            if frame is None:
                continue

            if not self.initialized:
                self.request_initialization(frame)
                self.requested_init = True
                self.keyframes.append(frame)
            else:
                self.track(frame)
                self.add_keyframe(frame)
