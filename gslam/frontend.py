import logging
from copy import deepcopy
from dataclasses import dataclass
from threading import Event
from typing import List
import os

import torch
import torch.multiprocessing as mp

import rerun as rr
import tqdm

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from .map import GaussianSplattingData
from .messages import BackendMessage, FrontendMessage
from .primitives import Frame, Pose
from .rasterization import RasterizerConfig
from .utils import get_projection_matrix, q_get, torch_image_to_np, torch_to_pil


import numpy as np


def tracking_loss(
    gt_img: torch.Tensor,
    rendered_img: torch.Tensor,
):
    return (rendered_img - gt_img).abs().mean()


@dataclass
class TrackingConfig:
    device: str = 'cuda'
    num_tracking_iters: int = 100
    photometric_loss: str = 'l1'
    pose_optim_lr_translation: float = 0.001
    pose_optim_lr_rotation: float = 0.003


class Frontend(mp.Process):
    def __init__(
        self,
        conf: TrackingConfig,
        rasterizer_conf: RasterizerConfig,
        backend_queue: mp.Queue,
        frontend_queue: mp.Queue,
        sensor_queue: mp.Queue,
        frontend_done_event: Event = None,
        backend_done_event: Event = None,
    ):
        super().__init__()
        self.conf: TrackingConfig = conf
        self.rasterizer_conf: RasterizerConfig = rasterizer_conf
        self.map_queue: mp.Queue = backend_queue
        self.queue: mp.Queue[int] = frontend_queue
        self.keyframes: List[Frame] = []

        self.splats = GaussianSplattingData.empty()

        self.requested_init = False
        self.initialized: bool = False
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(
            logging.Formatter(fmt='[%(levelname)s] %(name)s:%(lineno)s %(message)s')
        )
        self.logger.addHandler(handler)

        self.sensor_queue = sensor_queue
        self.frontend_done_event = frontend_done_event
        self.backend_done_event = backend_done_event
        os.makedirs('runs/final', exist_ok=True)
        os.makedirs('runs/gt', exist_ok=True)
        os.makedirs('runs/renders', exist_ok=True)

    def track(self, new_frame: Frame):
        previous_keyframe = self.keyframes[-1]

        # start with unit Rt difference?
        new_frame.pose = Pose(previous_keyframe.pose()).to(self.conf.device)

        pose_optimizer = torch.optim.Adam(
            [
                {'params': [new_frame.pose.dR], 'lr': self.conf.pose_optim_lr_rotation},
                {
                    'params': [new_frame.pose.dt],
                    'lr': self.conf.pose_optim_lr_translation,
                },
            ]
        )

        for i in (pbar := tqdm.trange(self.conf.num_tracking_iters)):
            pose_optimizer.zero_grad()
            rendered_rgb, rendered_alpha, render_info = self.splats(
                [new_frame.camera], [new_frame.pose]
            )
            rendered_rgb = rendered_rgb[0]
            loss = tracking_loss(rendered_rgb, new_frame.img)
            loss.backward()
            pose_optimizer.step()

            rr.log(
                'frontend/tracking/loss',
                rr.Scalar(loss.item()),
            )

            pbar.set_description(
                f"Tracking frame {len(self.keyframes)}, loss: {loss.item():.3f}"
            )

        self.last_radii = render_info['radii'].detach().cpu().numpy()

        print(f'{self.last_radii.shape=}')
        print(f'{self.last_radii.mean()=}')
        print(f'{self.last_radii.min()=}')
        print(f'{self.last_radii.max()=}')

        rendered_rgbd, rendered_alpha, render_info = self.splats(
            [new_frame.camera], [new_frame.pose], render_depth=True
        )

        rr.log(
            'frontend/tracking/rendered_rgb',
            rr.Image(torch_image_to_np(rendered_rgbd[..., :3])).compress(95),
        )

        rr.log(
            'frontend/tracking/rendered_depth',
            rr.Image(torch_image_to_np(rendered_rgbd[..., 3])).compress(95),
        )

        rr.log(
            'frontend/tracking/gt_rgb',
            rr.Image(torch_image_to_np(new_frame.img)).compress(95),
        )

        rr.log(
            'frontend/tracking/psnr',
            rr.Scalar(
                psnr(
                    torch_image_to_np(rendered_rgb),
                    torch_image_to_np(new_frame.img),
                )
            ),
        )

        print(f'{rendered_rgb.shape=}')
        print(f'{new_frame.img.shape=}')

        rr.log(
            'frontend/tracking/ssim',
            rr.Scalar(
                ssim(
                    torch_image_to_np(rendered_rgb),
                    torch_image_to_np(new_frame.img),
                    channel_axis=2,
                )
            ),
        )

        torch_to_pil(rendered_rgb).save(f'runs/renders/{len(self.keyframes):08}.png')
        torch_to_pil(new_frame.img).save(f'runs/gt/{len(self.keyframes):08}.png')

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

    def dump_pointcloud(self):
        rr.log(
            'frontend/tracking/pc',
            rr.Points3D(
                positions=self.splats.means.detach().cpu().numpy(),
                radii=self.splats.scales.min(dim=-1).values.detach().cpu().numpy(),
                colors=self.splats.colors.detach().cpu().numpy(),
                class_ids=self.last_radii,
            ),
            static=True,
        )

    def dump_video(self):
        for i, kf in enumerate(self.keyframes):
            rendered_rgb, rendered_alpha, render_info = self.splats(
                [kf.camera],
                [kf.pose],
            )
            torch_to_pil(rendered_rgb[0]).save(f'runs/final/{i:08}.png')
        os.system(
            'ffmpeg -y -framerate 30 -pattern_type glob -i "runs/final/*.png" -c:v libx264 -pix_fmt yuv420p runs/final.mp4'
        )
        os.system(
            'ffmpeg -y -framerate 30 -pattern_type glob -i "runs/gt/*.png" -c:v libx264 -pix_fmt yuv420p runs/gt.mp4'
        )
        os.system(
            'ffmpeg -y -framerate 30 -pattern_type glob -i "runs/renders/*.png" -c:v libx264 -pix_fmt yuv420p runs/renders.mp4'
        )
        # os.system(f'rm runs/*/*.png')

    def dump_trajectory(self):
        for i, kf in enumerate(self.keyframes):
            q, t = kf.pose.to_qt()
            q = np.roll(q.detach().cpu().numpy().reshape(-1), -1)
            t = t.detach().cpu().numpy().reshape(-1)
            print(f'{i=}, {q=}, {t=}')
            rr.log(
                f'frontend/tracking/pose_{i}',
                rr.Transform3D(rotation=rr.datatypes.Quaternion(xyzw=q), translation=t),
                static=True,
            )

    @rr.shutdown_at_exit
    def run(self):
        rr.init('gslam', recording_id='gslam_1')
        rr.save('runs/rr.rrd')

        self.Ks = get_projection_matrix().to(self.conf.device)

        self.logger.warning("test")

        self.waiting_for_sync = False

        while True:
            match q_get(self.queue):
                case [BackendMessage.SIGNAL_INITIALIZED]:
                    self.logger.warning("Initialization successful!")
                case [BackendMessage.SYNC, map_data, keyframes]:
                    self.sync_maps(map_data, keyframes)
                    self.initialized = True
                    self.waiting_for_sync = False
                case None:
                    pass
                case message_from_map:
                    self.logger.warning(f"Unknown {message_from_map=}")

            if self.waiting_for_sync:
                continue

            if self.sensor_queue.empty():
                continue
            frame: Frame = self.sensor_queue.get()
            if frame is None:
                # data stream exhausted
                self.map_queue.put(None)
                break

            frame = frame.to(self.conf.device)
            if not self.initialized:
                self.request_initialization(frame)
                self.keyframes.append(frame)
                self.waiting_for_sync = True
            else:
                self.track(frame)
                self.add_keyframe(frame)
                self.waiting_for_sync = True

        self.backend_done_event.wait()
        self.logger.warning('Got backend done.')
        self.logger.warning('emitted frontend done.')

        self.dump_trajectory()
        self.dump_pointcloud()
        self.dump_video()

        self.frontend_done_event.set()
