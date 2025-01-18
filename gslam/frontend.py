from copy import deepcopy
from dataclasses import dataclass
import logging
import os
from pathlib import Path
from threading import Event
from typing import List, Literal, assert_never

import torch
import torch.multiprocessing as mp

import rerun as rr
import tqdm

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from .map import GaussianSplattingData
from .messages import BackendMessage, FrontendMessage
from .primitives import Frame, Pose
from .trajectory import kabsch_umeyama, average_translation_error
from .utils import get_projection_matrix, q_get, torch_image_to_np, torch_to_pil


import matplotlib.pyplot as plt
import numpy as np


@dataclass
class TrackingConfig:
    device: str = 'cuda'
    num_tracking_iters: int = 50
    photometric_loss: Literal['l1', 'mse', 'active-nerf'] = 'l1'
    pose_optim_lr_translation: float = 0.001
    pose_optim_lr_rotation: float = 0.003

    kf_cov = 0.9
    kf_oc = 0.4


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

    def to_insert_keyframe(self, iou, _oc, _new_frame):
        # TODO implement insertion on pose diffs
        # TODO implement insertion on the last keyframe insertion
        #      being too far away in time
        # TODO implement insertion on visibility criterion like they do in MonoGS
        return iou < self.conf.kf_cov

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
                return (error * betas.pow(-1.0)).square().mean()
            case _:
                assert_never(self.conf.photometric_loss)

    def track(self, new_frame: Frame):
        previous_keyframe = self.keyframes[-1]
        previous_frame = self.frames[-1]

        # start with unit Rt difference?
        new_frame.pose = Pose(previous_frame.pose()).to(self.conf.device)

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
            outputs = self.splats([new_frame.camera], [new_frame.pose])

            rendered_rgb = outputs.rgbs[0]
            loss = self.tracking_loss(rendered_rgb, new_frame.img)
            loss.backward()
            pose_optimizer.step()

            rr.log(
                '/tracking/loss',
                rr.Scalar(loss.item()),
            )

            pbar.set_description(
                f"Tracking frame {len(self.frames)}, loss: {loss.item():.3f}"
            )

        with torch.no_grad():
            outputs = self.splats(
                [new_frame.camera], [new_frame.pose], render_depth=True
            )
            rendered_rgb = outputs.rgbs[0]
            rendered_depth = outputs.depths[0]
            rendered_beta = outputs.betas[0]

            new_frame.visible_gaussians = outputs.radii > 0

            n_visible_gaussians = new_frame.visible_gaussians.sum()
            n_visible_gaussians_last_kf = previous_keyframe.visible_gaussians.sum()

            intersection = torch.logical_and(
                new_frame.visible_gaussians, previous_keyframe.visible_gaussians
            )
            union = torch.logical_or(
                new_frame.visible_gaussians, previous_keyframe.visible_gaussians
            )

            iou = intersection.sum() / union.sum()
            oc = intersection.sum() / (
                min(n_visible_gaussians.sum().item(), n_visible_gaussians_last_kf.sum())
            )

        rr.log(
            '/tracking/psnr',
            rr.Scalar(
                psnr(
                    torch_image_to_np(rendered_rgb),
                    torch_image_to_np(new_frame.img),
                )
            ),
        )

        rr.log(
            '/tracking/ssim',
            rr.Scalar(
                ssim(
                    torch_image_to_np(rendered_rgb),
                    torch_image_to_np(new_frame.img),
                    channel_axis=2,
                )
            ),
        )

        torch_to_pil(rendered_rgb).save(
            self.output_dir / f'renders/{len(self.frames):08}.jpg'
        )
        torch_to_pil(new_frame.img).save(
            self.output_dir / f'gt/{len(self.frames):08}.jpg'
        )

        torch_to_pil(outputs.alphas[0, ..., 0]).save(
            self.output_dir / f'alphas/{len(self.frames):08}.jpg'
        )

        torch_to_pil(rendered_depth).save(
            self.output_dir / f'depths/{len(self.frames):08}.jpg'
        )

        torch_to_pil(rendered_beta, minmax_norm=True).save(
            self.output_dir / f'betas/{len(self.frames):08}.jpg'
        )

        self.frames.append(
            Frame(
                None,
                new_frame.timestamp,
                new_frame.camera.clone(),
                Pose(new_frame.pose(), False),
                new_frame.gt_pose,
                None,
            )
        )

        if self.to_insert_keyframe(iou, oc, new_frame):
            self.add_keyframe(new_frame)

        return new_frame.pose()

    def request_initialization(self, frame: Frame):
        self.frames.append(
            Frame(
                None,
                frame.timestamp,
                frame.camera.clone(),
                Pose(frame.pose(), False),
                frame.gt_pose,
                None,
            )
        )
        self.frozen_keyframes.append(frame)
        assert not self.initialized
        self.map_queue.put([FrontendMessage.REQUEST_INITIALIZE, deepcopy(frame)])

    def add_keyframe(self, frame: Frame):
        assert self.initialized
        self.frozen_keyframes.append(frame)
        self.map_queue.put([FrontendMessage.ADD_KEYFRAME, deepcopy(frame)])
        self.waiting_for_sync = True

    def sync_maps(self, splats, keyframes):
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

    def dump_video(self):
        for i, kf in enumerate(self.keyframes):
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

        for i, kf in enumerate(self.frames):
            outputs = self.splats(
                [kf.camera],
                [kf.pose],
            )
            torch_to_pil(outputs.rgbs[0]).save(
                self.output_dir / f'final_renders/{i:08}.jpg'
            )

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

    @torch.no_grad
    def dump_trajectory(self):
        Rts = [f.pose().cpu().numpy() for f in self.frozen_keyframes]
        ts = np.array([Rt[:3, 3] for Rt in Rts])
        gt_Rts = [f.gt_pose.cpu().numpy() for f in self.frozen_keyframes]
        gt_ts = np.array([Rt[:3, 3] for Rt in gt_Rts])

        R, c, t = kabsch_umeyama(gt_ts, ts)
        # TODO might need to get rid of this, it sets the first frames
        # to be in the same location for the gt and estimated trajectory
        # for easy comparision in plots
        t = gt_ts[0, ...]
        aligned_ts = np.array([t + c * R @ b for b in ts])

        fig, ax = plt.subplots()
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.plot(gt_ts[..., 0], gt_ts[..., 1], label='ground truth')
        ax.plot(aligned_ts[..., 0], aligned_ts[..., 1], label='gslam')
        ax.set_aspect('equal')
        ax.legend()
        fig.savefig(self.output_dir / 'frozen_keyframes.png')

        Rts = [f.pose().cpu().numpy() for f in self.frames]
        ts = np.array([Rt[:3, 3] for Rt in Rts])
        gt_Rts = [f.gt_pose.cpu().numpy() for f in self.frames]
        gt_ts = np.array([Rt[:3, 3] for Rt in gt_Rts])

        R, c, t = kabsch_umeyama(gt_ts, ts)
        # TODO might need to get rid of this, it sets the first frames
        # to be in the same location for the gt and estimated trajectory
        # for easy comparision in plots
        t = gt_ts[0, ...]
        aligned_ts = np.array([t + c * R @ b for b in ts])

        fig, ax = plt.subplots()
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.plot(gt_ts[..., 0], gt_ts[..., 1], label='ground truth')
        ax.plot(aligned_ts[..., 0], aligned_ts[..., 1], label='gslam')
        ax.set_aspect('equal')
        ax.legend()
        fig.savefig(self.output_dir / 'traj.png')

        ate = average_translation_error(gt_ts, ts)
        print(f'{ate=}')

        Rts = [f.pose().cpu().numpy() for f in self.keyframes]
        ts = np.array([Rt[:3, 3] for Rt in Rts])
        gt_Rts = [f.gt_pose.cpu().numpy() for f in self.keyframes]
        gt_ts = np.array([Rt[:3, 3] for Rt in gt_Rts])

        R, c, t = kabsch_umeyama(gt_ts, ts)
        # TODO might need to get rid of this, it sets the first frames
        # to be in the same location for the gt and estimated trajectory
        # for easy comparision in plots
        t = gt_ts[0, ...]
        aligned_ts = np.array([t + c * R @ b for b in ts])

        fig, ax = plt.subplots()
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.plot(gt_ts[..., 0], gt_ts[..., 1], label='ground truth')
        ax.plot(aligned_ts[..., 0], aligned_ts[..., 1], label='gslam')
        ax.set_aspect('equal')
        ax.legend()
        fig.savefig(self.output_dir / 'map_traj.png')

        ate = average_translation_error(gt_ts, ts)
        print(f'{ate=}')

        with open(self.output_dir / 'ate.txt', 'w') as f:
            f.write(f'{ate=}')

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

    @rr.shutdown_at_exit
    def run(self):
        rr.init('gslam', recording_id='gslam_1')
        rr.save(self.output_dir / 'rr-fe.rrd')
        rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

        self.Ks = get_projection_matrix().to(self.conf.device)

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

        self.backend_done_event.wait()
        self.logger.warning('Got backend done.')

        self.dump_trajectory()
        self.dump_pointcloud()
        self.dump_video()
        print('Done dumping everything')
        self.logger.warning('emitted frontend done.')

        self.frontend_done_event.set()
