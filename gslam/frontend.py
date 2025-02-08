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

import rerun as rr
import tqdm

import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from PIL import Image

from .map import GaussianSplattingData
from .messages import BackendMessage, FrontendMessage
from .primitives import Frame, Pose
from .trajectory import evaluate_trajectories
from .utils import (
    torch_image_to_np,
    torch_to_pil,
    false_colormap,
    ForkedPdb,
    total_variation_loss,
)
from .warp import Warp


fpdb = ForkedPdb()


@dataclass
class TrackingConfig:
    device: str = 'cuda'
    num_tracking_iters: int = 150
    photometric_loss: Literal['l1', 'mse', 'active-nerf'] = 'l1'
    pose_optim_lr_translation: float = 0.001
    pose_optim_lr_rotation: float = 0.003

    method: Literal['igs', 'warp'] = 'igs'

    dt_regularization: float = 0.01
    dR_regularization: float = 0.001


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

        self.sensor_queue = sensor_queue
        self.frontend_done_event = frontend_done_event
        self.backend_done_event = backend_done_event

        self.output_dir = output_dir
        os.makedirs(self.output_dir / 'final', exist_ok=True)
        os.makedirs(self.output_dir / 'gt', exist_ok=True)
        os.makedirs(self.output_dir / 'renders', exist_ok=True)
        os.makedirs(self.output_dir / 'alphas', exist_ok=True)
        os.makedirs(self.output_dir / 'depths', exist_ok=True)
        os.makedirs(self.output_dir / 'ddepths', exist_ok=True)
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
        self.reference_depthmap = torch.ones_like(new_frame.gt_depth)
        self.reference_ddepthmap = torch.zeros_like(
            new_frame.gt_depth, requires_grad=True
        )
        self.reference_rgbs = new_frame.img

        if self.conf.method == 'igs':
            self.request_initialization(new_frame)
        return

    def request_initialization(self, f: Frame):
        self.map_queue.put((FrontendMessage.REQUEST_INIT, deepcopy(f)))
        self.waiting_for_sync = True

    def track(self, new_frame: Frame):
        if len(self.frames) == 0:
            self.initialize(new_frame)
            n_iters = 0
        else:
            pose = self.frames[-1].pose()
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
                ]
            )

            if self.conf.method == 'warp':
                optimizer.add_param_group(
                    {
                        'params': [self.reference_ddepthmap],
                    },
                )

            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99)
            n_iters = self.conf.num_tracking_iters

        last_drdt = np.zeros((6,))
        new_drdt = np.zeros((6,))
        _loss = 0.0

        for i in (
            pbar := tqdm.trange(n_iters, desc=f"[Tracking] frame {len(self.frames)}")
        ):
            last_drdt[:3] = new_frame.pose.dR.detach().cpu().numpy()
            last_drdt[3:] = new_frame.pose.dt.detach().cpu().numpy()
            if self.conf.method == 'igs':
                outputs = self.splats([new_frame.camera], [new_frame.pose])
                rendered_rgb = outputs.rgbs[0]
                betas = outputs.betas[0]
                loss = self.tracking_loss(rendered_rgb, new_frame.img, betas)
            else:
                rendered_rgb, _normalized_warps, keep_mask = self.warp(
                    self.reference_frame.pose(),
                    new_frame.pose(),
                    self.reference_frame.img,
                    # self.reference_rgbs,
                    self.reference_depthmap + self.reference_ddepthmap,
                    # self.reference_depthmap,
                )
                masked_result = rendered_rgb[keep_mask, ...]
                masked_gt = new_frame.img[keep_mask, ...]
                loss = F.l1_loss(masked_result, masked_gt)

            _loss = loss.item()

            loss += new_frame.pose.dR.norm() * self.conf.dR_regularization
            loss += new_frame.pose.dt.norm() * self.conf.dt_regularization

            loss += total_variation_loss(self.reference_ddepthmap) * 30.0
            pbar.set_description(
                f"[Tracking] frame {len(self.frames)}| loss: {_loss:.3f}"
            )

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            new_drdt[:3] = new_frame.pose.dR.detach().cpu().numpy()
            new_drdt[3:] = new_frame.pose.dt.detach().cpu().numpy()

            if np.linalg.norm(last_drdt - new_drdt) < 1e-5:
                break

        self.log_frame(new_frame)
        self.save_tracking_stats(new_frame, _loss)
        self.frames.append(new_frame.strip())

        self.add_frame_to_backend(new_frame)

        i = len(self.frames)
        if n_iters == 0:
            torch_to_pil(self.reference_frame.img).save(
                self.output_dir / f"renders/{i:08}.jpg"
            )
        else:
            torch_to_pil(rendered_rgb).save(self.output_dir / f"renders/{i:08}.jpg")
        return new_frame.pose()

    def add_frame_to_backend(self, new_frame: Frame):
        self.map_queue.put((FrontendMessage.ADD_FRAME, deepcopy(new_frame)))
        return

    def sync(
        self,
        keyframes: dict[int, Frame],
        depthmap: torch.Tensor,
        rgbs: torch.Tensor,
        splats: GaussianSplattingData,
        pose_graph: dict[int, set],
    ):
        self.keyframes = keyframes
        self.reference_depthmap = depthmap.clone()
        self.reference_ddepthmap = torch.zeros_like(depthmap).requires_grad_(True)
        self.reference_frame = keyframes[sorted(keyframes.keys())[-1]]
        self.reference_rgbs = rgbs
        self.splats = splats
        self.pose_graph = pose_graph

        for kf in self.keyframes.values():
            self.log_frame(kf, name=f'/tracking/kf/{kf.index}')
        line_strips = []
        for kf_i in self.pose_graph:
            for kf_j in self.pose_graph[kf_i]:
                if kf_i > kf_j:
                    continue
                t1 = self.keyframes[kf_i].pose().inverse()[:3, 3].detach().cpu().numpy()
                t2 = self.keyframes[kf_j].pose().inverse()[:3, 3].detach().cpu().numpy()
                line_strips.append(np.vstack((t1, t2)).tolist())

        rr.log(
            '/tracking/pose_graph',
            rr.LineStrips3D(line_strips, colors=[[255, 255, 255]]),
        )

        self.dump_pointcloud()
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
        )

    @torch.no_grad()
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
                outputs.depthmaps[0],
                near=0.5,
                far=2.0,
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

    def log_frame(self, f: Frame, name: str = "/tracking/pose") -> None:
        q, t = f.pose.to_qt()
        q = np.roll(q.detach().cpu().numpy().reshape(-1), -1)
        t = t.detach().cpu().numpy().reshape(-1)
        rr.log(
            name,
            rr.Transform3D(
                rotation=rr.datatypes.Quaternion(xyzw=q),
                translation=t,
                from_parent=True,
            ),
        )
        # rr.log("/tracking/pose", rr.ViewCoordinates.RDF)  # X=Right, Y=Down, Z=Forward

        rr.log(
            f"{name}/image",
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

    @torch.no_grad
    def evaluate_trajectory(self) -> dict:
        fig, ates = evaluate_trajectories(
            {
                'keyframes': self.keyframes.values(),
                'tracking': self.frames,
            }
        )
        fig.savefig(self.output_dir / 'traj.png')
        plt.close(fig)
        return ates

    def create_videos(self):
        os.system(
            f'ffmpeg -hide_banner -loglevel error -y -framerate 30 -pattern_type glob -i "{os.path.normpath(self.output_dir)}/final/*.jpg" -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -pix_fmt yuv420p {self.output_dir/"final.mp4"}'
        )
        os.system(
            f'ffmpeg -hide_banner -loglevel error -y -framerate 30 -pattern_type glob -i "{os.path.normpath(self.output_dir)}/gt/*.jpg" -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -pix_fmt yuv420p {self.output_dir/"gt.mp4"}'
        )
        os.system(
            f'ffmpeg -hide_banner -loglevel error -y -framerate 30 -pattern_type glob -i "{os.path.normpath(self.output_dir)}/renders/*.jpg" -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -pix_fmt yuv420p {self.output_dir/"renders.mp4"}'
        )
        os.system(
            f'ffmpeg -hide_banner -loglevel error -y -framerate 30 -pattern_type glob -i "{os.path.normpath(self.output_dir)}/final_renders/*.jpg" -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -pix_fmt yuv420p {self.output_dir/"final_renders.mp4"}'
        )
        os.system(
            f'ffmpeg -hide_banner -loglevel error -y -framerate 30 -pattern_type glob -i "{os.path.normpath(self.output_dir)}/final_depths/*.jpg" -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -pix_fmt yuv420p {self.output_dir/"final_depths.mp4"}'
        )

    def save_tracking_stats(self, new_frame, loss):
        depth = self.reference_depthmap + self.reference_ddepthmap

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

        ddepth = self.reference_ddepthmap.clone()
        ddepth = (ddepth - ddepth.min()) / (ddepth.max() - ddepth.min() + 1e-10)
        ddepth[self.reference_ddepthmap.abs() < 0.01] = 0

        false_colormap(ddepth).save(
            self.output_dir / f'ddepths/{len(self.frames):08}.jpg'
        )

    def handle_message_from_backend(self, message):
        match message:
            case [BackendMessage.SYNC, keyframes, depthmap, rgbs, splats, pose_graph]:
                self.sync(keyframes, depthmap, rgbs, splats, pose_graph)
                self.waiting_for_sync = False
            case [BackendMessage.END_SYNC, map_data, keyframes]:
                self.sync_at_end(map_data, keyframes)
                self.waiting_for_end_sync = False
                self.done = True
                print("We have end sync!")
            case message_from_map:
                raise ValueError(f"Unknown {message_from_map=}")

    @rr.shutdown_at_exit
    def run(self):
        rr.init('gslam', recording_id=f'gslam_1_{int(time.time())%10000}', spawn=True)
        # rr.save(self.output_dir / 'rr-fe.rrd')
        rr.log("/tracking", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

        self.warp = None

        self.waiting_for_end_sync = False
        self.waiting_for_sync = False

        last_time_we_heard_from_backend = time.time()

        self.done = False

        while True:
            if not self.queue.empty():
                self.handle_message_from_backend(self.queue.get())
                last_time_we_heard_from_backend = time.time()

            if self.waiting_for_end_sync:
                if (time.time() - last_time_we_heard_from_backend) > 3000.0:
                    print('Looks like backend\'s dead')
                    break
                continue

            if self.waiting_for_sync:
                continue

            if self.done:
                break

            if not self.sensor_queue.empty():
                frame: Frame = self.sensor_queue.get()
                if frame is None:
                    # data stream exhausted
                    self.map_queue.put(None)
                    self.waiting_for_end_sync = True
                    last_time_we_heard_from_backend = time.time()
                    continue

                frame = frame.to(self.conf.device)
                if self.warp is None:
                    self.warp = Warp(
                        frame.camera.intrinsics, frame.camera.height, frame.camera.width
                    )
                self.track(frame)

                if len(self.frames) % 30 == 0:
                    checkpoint_file = self.output_dir / 'splats.ckpt'
                    torch.save(self.splats, checkpoint_file)
                    print(f'Saved Checkpoints to {checkpoint_file}')

                    metrics = dict()
                    metrics['L'] = len(self.frames)
                    metrics['C'] = len(self.keyframes)
                    metrics['N'] = self.splats.means.shape[0]
                    metrics.update(self.evaluate_trajectory())
                    print(f'{metrics=}')
                    with open(self.output_dir / 'metrics.json', 'a') as f:
                        json.dump(metrics, f)
                        f.write('\n')

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
        with open(self.output_dir / 'metrics.json', 'a') as f:
            json.dump(metrics, f)

        checkpoint_file = self.output_dir / 'splats.ckpt'
        torch.save(self.splats, checkpoint_file)
        print(f'Saved Checkpoints to {checkpoint_file}')

        self.logger.warning('frontend done.')

        self.frontend_done_event.set()
