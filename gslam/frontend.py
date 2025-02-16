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
from pyquaternion import Quaternion
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from PIL import Image

from .map import GaussianSplattingData
from .messages import BackendMessage, FrontendMessage
from .primitives import Frame, Pose
from .rasterization import RasterizationOutput
from .trajectory import evaluate_trajectories
from .utils import (
    torch_image_to_np,
    torch_to_pil,
    false_colormap,
    ForkedPdb,
)
from .warp import Warp


fpdb = ForkedPdb()


@dataclass
class TrackingConfig:
    device: str = 'cuda'
    num_tracking_iters: int = 150
    photometric_loss: Literal['l1', 'mse', 'active-nerf'] = 'active-nerf'
    pose_optim_lr: float = 0.002

    method: Literal['igs', 'warp'] = 'igs'

    pose_regularization: float = 0.01


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
                return (error.square().sum(dim=-1) * betas.pow(-2.0)).mean()
            case _:
                assert_never(self.conf.photometric_loss)

    def initialize(self, new_frame: Frame):
        pose = torch.eye(4, device=self.conf.device)
        new_frame.pose = Pose(pose.detach()).to(self.conf.device)
        self.keyframes[new_frame.index] = new_frame
        self.reference_frame = new_frame
        self.reference_depthmap = torch.ones_like(new_frame.gt_depth)
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
        elif len(self.frames) == 1:
            pose = self.frames[-1].pose()
            new_frame.pose = Pose(pose.detach()).to(self.conf.device)
            optimizer = torch.optim.Adam(
                [
                    {
                        'params': new_frame.pose.parameters(),
                        'lr': self.conf.pose_optim_lr,
                    }
                ]
            )

            # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99)
            n_iters = self.conf.num_tracking_iters
        else:
            # constant motion model
            pose_a = self.frames[-2].pose()
            pose_b = self.frames[-1].pose()
            pose_c = pose_b @ torch.linalg.inv(pose_a) @ pose_b
            new_frame.pose = Pose(pose_c.detach()).to(self.conf.device)
            optimizer = torch.optim.Adam(
                [
                    {
                        'params': new_frame.pose.se3,
                        'lr': self.conf.pose_optim_lr,
                    }
                ]
            )

            # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 1.0)
            n_iters = self.conf.num_tracking_iters

        _loss = 0.0
        outputs = None

        for i in (
            pbar := tqdm.trange(n_iters, desc=f"[Tracking] frame {len(self.frames)}")
        ):
            if self.conf.method == 'igs':
                outputs = self.splats(
                    [new_frame.camera], [new_frame.pose], render_depth=True
                )
                rendered_rgb = outputs.rgbs[0]
                betas = outputs.betas[0]
                loss = self.tracking_loss(rendered_rgb, new_frame.img, betas)
            else:
                rendered_rgb, _normalized_warps, keep_mask = self.warp(
                    self.reference_frame.pose(),
                    new_frame.pose(),
                    self.reference_frame.img,
                    # self.reference_rgbs,
                    self.reference_depthmap,
                    # self.reference_depthmap,
                )
                masked_result = rendered_rgb[keep_mask, ...]
                masked_gt = new_frame.img[keep_mask, ...]
                loss = F.l1_loss(masked_result, masked_gt)

            _loss = loss.item()

            loss += new_frame.pose.se3.norm() * self.conf.pose_regularization

            loss.backward()
            optimizer.step()
            # scheduler.step()
            optimizer.zero_grad()

            drdt = new_frame.pose.se3.detach().cpu().numpy()
            pbar.set_description(
                f"[Tracking] frame {len(self.frames)}| loss: {_loss:.8f}| {np.linalg.norm(drdt)}"
            )
            new_frame.pose.normalize()

            if np.linalg.norm(drdt) < 5e-5:
                break

        self.log_frame(new_frame)
        self.save_tracking_stats(new_frame, _loss, outputs)
        self.frames.append(new_frame.strip())

        self.add_frame_to_backend(new_frame)

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
        self.keyframes = deepcopy(keyframes)
        self.reference_depthmap = depthmap.clone()
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
        self.splats, self.keyframes = splats, deepcopy(keyframes)
        return

    @torch.no_grad()
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

        transparency = torch.sigmoid(self.splats.opacities)
        radii = self.splats.scales.exp() * transparency.unsqueeze(-1) * 2.0 + 0.004
        q = self.splats.quats.cpu().numpy()
        q = np.roll(q, -1, axis=1)
        rr.log(
            '/tracking/splats',
            rr.Ellipsoids3D(
                half_sizes=radii.cpu().numpy(),
                centers=self.splats.means.cpu().numpy(),
                quaternions=q,
                # colors=self.splats.colors.sigmoid().cpu().numpy(),
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
                fill_mode=rr.components.FillMode.Solid,
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
                near=0.0,
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
            },
            keyframe_indices=sorted(self.keyframes.keys()),
        )
        fig.savefig(self.output_dir / 'traj.png')
        plt.close(fig)
        return ates

    @torch.no_grad
    def save_trajectories(self) -> dict:
        def format_pose(pose: torch.Tensor, timestamp: float):
            Rt = pose.detach().cpu().numpy()
            q = Quaternion(matrix=Rt[:3, :3], rtol=1e-3, atol=1e-5)
            t = Rt[:3, 3]
            q = q.unit.elements.tolist()
            values = [timestamp, *(t.tolist()), q[1], q[2], q[3], q[0]]
            return ' '.join(map(str, values)) + '\n'

        with (
            open(self.output_dir / 'gt_traj_frames.txt', 'w') as gt_traj_file,
            open(self.output_dir / 'gslam_traj_frames.txt', 'w') as gslam_traj_file,
        ):
            for f in self.frames:
                gslam_traj_file.write(format_pose(f.pose(), f.timestamp))
                gt_traj_file.write(format_pose(f.gt_pose, f.timestamp))

        with (
            open(self.output_dir / 'gt_traj_keyframes.txt', 'w') as gt_traj_file,
            open(self.output_dir / 'gslam_traj_keyframes.txt', 'w') as gslam_traj_file,
        ):
            for kf_index in sorted(self.keyframes.keys()):
                f = self.keyframes[kf_index]
                gslam_traj_file.write(format_pose(f.pose(), f.timestamp))
                gt_traj_file.write(format_pose(f.gt_pose, f.timestamp))

    def create_videos(self):
        os.system(
            f'ffmpeg -hide_banner -loglevel error -y -framerate 30 -pattern_type glob -i "{os.path.normpath(self.output_dir)}/final/*.jpg" -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -pix_fmt yuv420p {self.output_dir / "final.mp4"}'
        )
        os.system(
            f'ffmpeg -hide_banner -loglevel error -y -framerate 30 -pattern_type glob -i "{os.path.normpath(self.output_dir)}/gt/*.jpg" -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -pix_fmt yuv420p {self.output_dir / "gt.mp4"}'
        )
        os.system(
            f'ffmpeg -hide_banner -loglevel error -y -framerate 30 -pattern_type glob -i "{os.path.normpath(self.output_dir)}/renders/*.jpg" -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -pix_fmt yuv420p {self.output_dir / "renders.mp4"}'
        )
        os.system(
            f'ffmpeg -hide_banner -loglevel error -y -framerate 30 -pattern_type glob -i "{os.path.normpath(self.output_dir)}/final_renders/*.jpg" -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -pix_fmt yuv420p {self.output_dir / "final_renders.mp4"}'
        )
        os.system(
            f'ffmpeg -hide_banner -loglevel error -y -framerate 30 -pattern_type glob -i "{os.path.normpath(self.output_dir)}/final_depths/*.jpg" -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -pix_fmt yuv420p {self.output_dir / "final_depths.mp4"}'
        )

    def save_tracking_stats(self, new_frame, loss, outputs: RasterizationOutput = None):
        i = new_frame.index

        rr.log('/tracking/loss', rr.Scalar(loss))

        torch_to_pil(new_frame.img).save(self.output_dir / f'gt/{i:08}.jpg')

        if outputs is not None:
            false_colormap(outputs.betas[0], near=0.0, far=2.0).save(
                self.output_dir / f'betas/{i:08}.jpg'
            )

            false_colormap(
                outputs.depthmaps[0],
                near=0.0,
                far=min(1.5, outputs.depthmaps[0].max().item()),
            ).save(self.output_dir / f'depths/{i:08}.jpg')

            torch_to_pil(outputs.rgbs[0]).save(self.output_dir / f"renders/{i:08}.jpg")

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
        rr.init('gslam', recording_id=f'gslam_1_{int(time.time()) % 10000}', spawn=True)
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
        self.save_trajectories()
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
