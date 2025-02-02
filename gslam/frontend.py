from copy import deepcopy
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
from threading import Event
import time
from typing import List, Literal, assert_never

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

from PIL import Image
import rerun as rr
import tqdm

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from .map import GaussianSplattingData
from .messages import BackendMessage, FrontendMessage
from .primitives import Frame, Pose
from .trajectory import evaluate_trajectories
from .utils import (
    q_get,
    torch_image_to_np,
    torch_to_pil,
    false_colormap,
    ForkedPdb,
)
from .warp import Warp


import numpy as np

fpdb = ForkedPdb()


@dataclass
class TrackingConfig:
    device: str = 'cuda'
    num_tracking_iters: int = 100
    photometric_loss: Literal['l1', 'mse', 'active-nerf'] = 'l1'
    pose_optim_lr_translation: float = 0.001
    pose_optim_lr_rotation: float = 0.003

    dt_regularization: float = 0.01
    dR_regularization: float = 0.001

    kf_cov = 0.8
    kf_oc = 0.4
    kf_m = 0.08


class Frontend(mp.Process):
    def __init__(
        self,
        conf: TrackingConfig,
        backend_queue: mp.Queue,
        frontend_queue: mp.Queue,
        sensor_queue: mp.Queue,
        frontend_done_event: Event = None,
        backend_done_event: Event = None,
        output_dir: Path = None,
    ):
        super().__init__()
        self.conf: TrackingConfig = conf
        self.map_queue: mp.Queue = backend_queue
        self.queue: mp.Queue[int] = frontend_queue
        self.keyframes: dict[int, Frame] = dict()

        self.requested_init = False
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(
            logging.Formatter(fmt='[%(levelname)s] %(name)s:%(lineno)s %(message)s')
        )
        self.logger.addHandler(handler)

        self.frames: List[Frame] = []
        self.frozen_keyframes: List[Frame] = []

        self.sensor_queue = sensor_queue
        self.frontend_done_event = frontend_done_event
        self.backend_done_event = backend_done_event

        self.output_dir = output_dir
        os.makedirs(self.output_dir / 'final', exist_ok=True)
        os.makedirs(self.output_dir / 'gt', exist_ok=True)
        os.makedirs(self.output_dir / 'renders', exist_ok=True)
        os.makedirs(self.output_dir / 'alphas', exist_ok=True)
        os.makedirs(self.output_dir / 'depths', exist_ok=True)
        os.makedirs(self.output_dir / 'betas', exist_ok=True)
        os.makedirs(self.output_dir / 'final_renders', exist_ok=True)
        os.makedirs(self.output_dir / 'final_depths', exist_ok=True)

    def tracking_loss(
        self,
        gt_img: torch.Tensor,
        rendered_img: torch.Tensor,
        betas: torch.Tensor = None,
    ) -> torch.Tensor:
        error = rendered_img - gt_img
        match self.conf.photometric_loss:
            case 'l1':
                return error.abs().mean()
            case 'mse':
                return error.square().mean()
            case 'active-nerf':
                return (error * betas.pow(-1.0).unsqueeze(-1)).square().mean()
            case _:
                assert_never(self.conf.photometric_loss)

    def initialize(self, new_frame: Frame):
        pose = torch.eye(4, device=self.conf.device)
        new_frame.pose = Pose(pose.detach()).to(self.conf.device)
        self.keyframes[new_frame.index] = new_frame
        self.reference_frame = new_frame
        self.reference_depthmap = torch.ones_like(
            new_frame.gt_depth, requires_grad=True
        )
        return

    def track(self, new_frame: Frame):
        if len(self.frames) == 0:
            self.initialize(new_frame)
            n_iters = 0
        else:
            pose = self.reference_frame.pose()
            new_frame.pose = Pose(pose.detach()).to(self.conf.device)
            optimizer = torch.optim.Adam(
                [
                    {
                        'params': [new_frame.pose.dR],
                        'lr': self.conf.pose_optim_lr_rotation,
                    },
                    {
                        'params': [new_frame.pose.dt],
                        'lr': self.conf.pose_optim_lr_translation,
                    },
                    {
                        'params': [self.reference_depthmap],
                    },
                ]
            )
            n_iters = self.conf.num_tracking_iters

        last_loss = float('inf')

        for i in (
            pbar := tqdm.trange(
                n_iters,
                desc=f"[Tracking] frame {len(self.frames)}",
            )
        ):
            optimizer.zero_grad()

            result, _normalized_warps, keep_mask = self.warp(
                self.reference_frame.pose(),
                new_frame.pose(),
                self.reference_frame.img,
                self.reference_depthmap,
            )
            result = result[keep_mask, ...]
            gt = new_frame.img[keep_mask, ...]
            loss = F.l1_loss(result, gt)

            loss += new_frame.pose.dR.norm() * self.conf.dR_regularization
            loss += new_frame.pose.dt.norm() * self.conf.dt_regularization

            loss.backward()
            optimizer.step()

            if 0 < ((last_loss - loss) / loss) < (1.0 / 2550.0):
                # we've 'converged'!
                pbar.set_description(
                    f"[Tracking] frame {len(self.frames)}, loss: {loss.item():.3f}"
                )
                break

            last_loss = loss.item()

        self.save_tracking_stats(new_frame, last_loss)
        self.frames.append(new_frame.strip())

        self.add_frame_to_backend(new_frame)
        return new_frame.pose()

    def add_frame_to_backend(self, new_frame: Frame):
        self.map_queue.put((FrontendMessage.ADD_FRAME, deepcopy(new_frame.strip())))
        return

    def sync(self, keyframes: dict[int, Frame], depthmap: torch.Tensor):
        self.keyframes, self.last_kf_depths = keyframes, depthmap
        return

    def sync_at_end(self, splats: GaussianSplattingData, keyframes: dict[int, Frame]):
        self.splats, self.keyframes = splats, keyframes
        return

    def dump_pointcloud(self):
        rr.log(
            '/tracking/pc',
            rr.Points3D(
                positions=self.splats.means.detach().cpu().numpy(),
                radii=torch.exp(self.splats.scales)
                .min(dim=-1)
                .values.detach()
                .cpu()
                .numpy()
                * 0.5,
                colors=torch.sigmoid(
                    torch.cat(
                        [
                            self.splats.colors,
                            self.splats.opacities[..., None],
                        ],
                        dim=1,
                    )
                )
                .detach()
                .cpu()
                .numpy(),
            ),
            static=True,
        )

    def evaluate_reconstruction(self):
        for i, kf in enumerate(
            tqdm.tqdm(self.keyframes.values(), 'Rendering all keyframes')
        ):
            outputs = self.splats(
                [kf.camera],
                [kf.pose],
            )
            torch_to_pil(outputs.rgbs[0]).save(self.output_dir / f'final/{i:08}.jpg')

            rr.log(
                f'/tracking/pose_{i}/image',
                rr.Image(
                    torch_image_to_np(outputs.rgbs[0]), color_model=rr.ColorModel.RGB
                ).compress(jpeg_quality=95),
            )

        psnrs = []
        ssims = []

        for i, f in enumerate(tqdm.tqdm(self.frames, 'Rendering all frames')):
            outputs = self.splats(
                [f.camera],
                [f.pose],
                render_depth=True,
            )
            torch_to_pil(outputs.rgbs[0]).save(
                self.output_dir / f'final_renders/{i:08}.jpg'
            )
            false_colormap(
                outputs.depthmaps[0], mask=outputs.alphas[0, ..., 0] > 0.3
            ).save(self.output_dir / f'final_depths/{i:08}.jpg')

            if f.img_file is None:
                continue
            gt_img = np.array(Image.open(f.img_file))
            psnrs.append(
                psnr(
                    torch_image_to_np(outputs.rgbs[0]),
                    gt_img,
                )
            )
            ssims.append(
                ssim(
                    torch_image_to_np(outputs.rgbs[0]),
                    gt_img,
                    channel_axis=2,
                )
            )

        return {'ssim': np.mean(ssims), 'psnr': np.mean(psnrs)}

    @torch.no_grad
    def evaluate_trajectory(self) -> dict:
        fig, ates = evaluate_trajectories(
            {
                'frozen': self.frozen_keyframes,
                'optimized': self.keyframes.values(),
                'tracking': self.frames,
            }
        )
        fig.savefig(self.output_dir / 'traj.png')

        for i, f in enumerate(self.frames):
            q, t = f.pose.to_qt()
            q = np.roll(q.detach().cpu().numpy().reshape(-1), -1)
            t = t.detach().cpu().numpy().reshape(-1)
            rr.log(
                f'/tracking/pose_{i}',
                rr.Transform3D(
                    rotation=rr.datatypes.Quaternion(xyzw=q),
                    translation=t,
                    from_parent=True,
                ),
                static=True,
            )
            rr.log(
                f"/tracking/pose_{i}", rr.ViewCoordinates.RDF, static=True
            )  # X=Right, Y=Down, Z=Forward

            rr.log(
                f'/tracking/pose_{i}/image',
                rr.Pinhole(
                    resolution=[f.camera.width, f.camera.height],
                    focal_length=[
                        f.camera.intrinsics[0, 0].item(),
                        f.camera.intrinsics[1, 1].item(),
                    ],
                    principal_point=[
                        f.camera.intrinsics[0, 2].item(),
                        f.camera.intrinsics[1, 2].item(),
                    ],
                ),
            )
        return ates

    def create_videos(self):
        os.system(
            f'ffmpeg -hide_banner -loglevel error -y -framerate 30 -pattern_type glob -i "{os.path.normpath(self.output_dir)}/final/*.jpg" -c:v libx264 -pix_fmt yuv420p {self.output_dir/"final.mp4"}'
        )
        os.system(
            f'ffmpeg -hide_banner -loglevel error -y -framerate 30 -pattern_type glob -i "{os.path.normpath(self.output_dir)}/gt/*.jpg" -c:v libx264 -pix_fmt yuv420p {self.output_dir/"gt.mp4"}'
        )
        os.system(
            f'ffmpeg -hide_banner -loglevel error -y -framerate 30 -pattern_type glob -i "{os.path.normpath(self.output_dir)}/renders/*.jpg" -c:v libx264 -pix_fmt yuv420p {self.output_dir/"renders.mp4"}'
        )
        os.system(
            f'ffmpeg -hide_banner -loglevel error -y -framerate 30 -pattern_type glob -i "{os.path.normpath(self.output_dir)}/final_renders/*.jpg" -c:v libx264 -pix_fmt yuv420p {self.output_dir/"final_renders.mp4"}'
        )
        os.system(
            f'ffmpeg -hide_banner -loglevel error -y -framerate 30 -pattern_type glob -i "{os.path.normpath(self.output_dir)}/final_depths/*.jpg" -c:v libx264 -pix_fmt yuv420p {self.output_dir/"final_depths.mp4"}'
        )

    def save_tracking_stats(self, new_frame, loss):
        depth = self.reference_depthmap

        rr.log(
            '/tracking/loss',
            rr.Scalar(loss),
        )

        torch_to_pil(new_frame.img).save(
            self.output_dir / f'gt/{len(self.frames):08}.jpg'
        )

        false_colormap(depth).save(
            self.output_dir / f'depths/{len(self.frames):08}.jpg'
        )

    @rr.shutdown_at_exit
    def run(self):
        rr.init('gslam', recording_id='gslam_1')
        rr.save(self.output_dir / 'rr-fe.rrd')
        rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

        self.warp = None

        self.waiting_for_sync = False

        last_time_we_heard_from_backend = time.time()

        self.done = False

        while True:
            match q_get(self.queue):
                case [BackendMessage.SYNC, keyframes, depthmap]:
                    self.sync(keyframes, depthmap)
                    print("We synced depthmaps")
                    last_time_we_heard_from_backend = time.time()
                case [BackendMessage.END_SYNC, map_data, keyframes]:
                    print("We have end sync!")
                    self.sync_at_end(map_data, keyframes)
                    self.waiting_for_sync = False
                    last_time_we_heard_from_backend = time.time()
                case None:
                    pass
                case message_from_map:
                    self.logger.warning(f"Unknown {message_from_map=}")
                    last_time_we_heard_from_backend = time.time()

            if self.waiting_for_sync:
                continue

            if (time.time() - last_time_we_heard_from_backend) > 30.0:
                print('Looks like backend\'s dead')
                break

            if self.done:
                break
            if self.sensor_queue.empty():
                continue
            frame: Frame = self.sensor_queue.get()
            if frame is None:
                # data stream exhausted
                self.map_queue.put(None)
                self.done = True
                self.waiting_for_sync = True
                continue

            frame = frame.to(self.conf.device)
            if self.warp is None:
                self.warp = Warp(
                    frame.camera.intrinsics, frame.camera.height, frame.camera.width
                )
            self.track(frame)

        self.backend_done_event.wait()
        self.logger.warning('Got backend done.')
        self.dump_pointcloud()
        metrics = dict()
        metrics.update(self.evaluate_reconstruction())
        metrics.update(self.evaluate_trajectory())
        self.create_videos()

        metrics['N'] = self.splats.means.shape[0]
        metrics['C'] = len(self.keyframes)
        metrics['L'] = len(self.frames)

        print(f'{metrics=}')
        with open(self.output_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f)

        checkpoint_file = self.output_dir / 'splats.ckpt'
        torch.save(self.splats, checkpoint_file)
        print(f'Saved Checkpoints to {checkpoint_file}')

        self.logger.warning('frontend done.')

        self.frontend_done_event.set()
