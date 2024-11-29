import torch
from dataclasses import dataclass

import torch.multiprocessing as mp

from .rasterization import RasterizerConfig
from .primitives import Camera, Frame
from .utils import get_projection_matrix
from gsplat.rendering import rasterization
import tqdm
import logging

from typing import List


def tracking_loss(
        gt_img,
        rendered_img,
        ):
    return (rendered_img - gt_img).abs().sum()


# consider the implications of all these structs being torch modules
class GaussianSplattingModel:

    def __init__(self,
                 means,        # gaussian centers
                 covar_quats,  # quaternions of covariance matrices
                 covar_scales, # scales of covariance matrices
                 opacities,    # alpha of gaussians
                 colors):      # RGB values of gaussians
        self.means: torch.Tensor = means
        self.covar_quats: torch.Tensor = covar_quats
        self.covar_scales: torch.Tensor = covar_scales
        self.opacities: torch.Tensor = opacities
        self.colors: torch.Tensor = colors


    def render(self,
               camera: Camera):
        rendered_rgb, rendered_alpha, render_info = rasterization(
            means = self.means,
            quats = self.covar_quats,
            scales = self.covar_scales,
            opacities = self.opacities,
            colors = self.colors,
            viewmats = camera.viewmat.unsqueeze(0),
            Ks = camera.intrinsics,
            width = camera.width,
            height = camera.height,
        )
        return rendered_rgb, rendered_alpha, render_info


    @staticmethod
    def new_empty_model(device: str = 'cuda'):
        return GaussianSplattingModel(
            torch.Tensor(device=device),
            torch.Tensor(device=device),
            torch.Tensor(device=device),
            torch.Tensor(device=device),
            torch.Tensor(device=device),
        )


    def clone(self):
        return GaussianSplattingModel(
            self.means.clone(),
            self.covar_quats.clone(),
            self.covar_scales.clone(),
            self.opacities.clone(),
            self.colors.clone(),
        )


@dataclass
class TrackingConfig:
    device: str = 'cuda'
    num_tracking_iters: int = 150

    photometric_loss: str = 'l1'


class Frontend(mp.Process):

    def __init__(self,
                 tracking_conf: TrackingConfig,
                 rasterizer_conf: RasterizerConfig,
                 backend_queue: mp.JoinableQueue,
                 frontend_queue: mp.JoinableQueue,
                 sensor_queue: mp.JoinableQueue):

        self.tracking_config: TrackingConfig = tracking_conf
        self.rasterizer_conf: RasterizerConfig = rasterizer_conf
        self.map_queue: mp.JoinableQueue = backend_queue
        self.queue : mp.JoinableQueue[int] = frontend_queue
        self.keyframes: List[Frame] = []

        self.map = GaussianSplattingModel.new_empty_model()

        self.initialized: bool = False
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel("DEBUG")

    def track(self,
              new_frame: Frame):
        # once initialized, we can start tracking new frames

        previous_keyframe = self.keyframes[-1]

        # start with unit Rt difference
        new_frame.viewpoint = previous_keyframe.viewpoint.clone()

        width, height, _ = new_frame.img.shape

        for i in (pbar := tqdm.trange(self.config.num_tracking_iters)):
            rendered_rgb, rendered_alpha, render_info = self.map.render()
            loss = tracking_loss(rendered_rgb, new_frame.img)
            loss.backward()
            self.map.tracking_step()
            self.map.tracking_zero_grad()

            pbar.set_description(f"loss: {loss.item()}")

        return


    def request_initialization(self, frame: Frame):
        assert not self.initialized
        self.map_queue.put('request-init')


    def sync_maps(self):
        # TODO implement this thang
        # sync map with the backend, eventually we might need to sync only a subset of the map
        # because memory and stuff
        return


    def run(self):
        self.Ks = get_projection_matrix().to(self.config.device)

        tic = torch.cuda.Event(enable_timing=True)
        toc = torch.cuda.Event(enable_timing=True)

        while True:

            if not self.queue.empty():
                message_from_map = self.queue.get()
                if message_from_map == 'init-success':
                    self.logger.debug(f"Initialization successful!")
                    self.sync_maps()

            if not self.sensor_queue.empty():

                # remove an item from the queue
                frame = self.sensor_queue.get()

                if not self.initialized:
                    self.request_initialization(frame)
                    self.logger.debug(f'Requested initialization.')
                else:
                    self.logger.debug(f'Tracking.')
                    raise NotImplementedError()
                    self.track(frame)

                # mark the job done
                self.sensor_queue.task_done()
