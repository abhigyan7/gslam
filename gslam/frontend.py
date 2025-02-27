from copy import deepcopy
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
from threading import Event, Thread
import time
from typing import List, Literal, assert_never

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

import rerun as rr
import tqdm

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from PIL import Image

import pypose as pp
from pypose.function.geometry import pixel2point, reprojerr
from pypose.optim import LM
from pypose.optim.kernel import Huber
from pypose.optim.solver import Cholesky
from pypose.optim.strategy import TrustRegion
from pypose.optim.corrector import FastTriggs
from pypose.optim.scheduler import StopOnPlateau

from .map import GaussianSplattingData
from .messages import BackendMessage, FrontendMessage
from .primitives import Frame, Pose, matrix_to_quaternion
from .rasterization import RasterizationOutput
from .trajectory import evaluate_trajectories
from .utils import torch_image_to_np, torch_to_pil, false_colormap, ForkedPdb, unvmap
from .visualization import log_frame, get_blueprint
from .warp import Warp


fpdb = ForkedPdb()


@dataclass
class TrackingConfig:
    device: str = 'cuda'
    num_tracking_iters: int = 100
    photometric_loss: Literal['l1', 'mse', 'active-nerf'] = 'active-nerf'

    pose_optim_lr: float = 0.002
    pose_optim_lr_decay: float = 0.99

    method: Literal['igs', 'warp', 'flow'] = 'igs'

    pose_regularization: float = 0


class LocalBundleAdjustment(torch.nn.Module):
    def __init__(self, K, pts1, pts2, depth, init_T) -> None:
        super().__init__()
        self.register_buffer("K", K)
        self.register_buffer("pts1", pts1)  # N x 2, uv coordinate
        self.register_buffer("pts2", pts2)  # N x 2, uv coordinate

        self.T = pp.Parameter(init_T)
        self.depth = torch.nn.Parameter(depth)

    def forward(self) -> torch.Tensor:
        pts3d = pixel2point(self.pts1, self.depth, self.K)
        return reprojerr(pts3d, self.pts2, self.K, self.T, reduction='none')


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

    def tracking_residual(
        self,
        gt_img: torch.Tensor,
        rendered_img: torch.Tensor,
        betas: torch.Tensor = None,
    ) -> torch.Tensor:
        error = rendered_img - gt_img
        return error.square().sum(dim=-1) * betas.pow(-2.0)

    def initialize(self, new_frame: Frame):
        pose = torch.eye(4, device=self.conf.device)
        new_frame.pose = Pose(pose.detach()).to(self.conf.device)
        self.keyframes[new_frame.index] = new_frame
        self.reference_frame = new_frame
        self.reference_depthmap = torch.ones_like(new_frame.gt_depth)
        self.reference_rgbs = new_frame.img

        if self.conf.method in ('igs', 'flow'):
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

            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, self.conf.pose_optim_lr_decay
            )
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
                        'params': new_frame.pose.parameters(),
                        'lr': self.conf.pose_optim_lr,
                    }
                ]
            )

            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, self.conf.pose_optim_lr_decay
            )
            n_iters = self.conf.num_tracking_iters

        outputs = None
        start_time = time.time()

        loss = 0
        if n_iters > 0:
            if self.conf.method == 'warp':
                loss = self.warp_track(new_frame, n_iters, optimizer, scheduler)
            elif self.conf.method == 'igs':
                loss = self.igs_track(new_frame, n_iters, optimizer, scheduler)
            elif self.conf.method == 'flow':
                loss = self.flow_track(new_frame)
            else:
                assert_never(self.conf.method)

        if hasattr(self, 'splats'):
            outputs = self.splats(
                [new_frame.camera], [new_frame.pose], render_depth=True
            )

        Thread(
            target=log_frame,
            args=(new_frame,),
            kwargs={
                "outputs": outputs,
                "loss": loss,
                "tracking_time": time.time() - start_time if n_iters > 0 else -1,
            },
        ).start()
        Thread(
            target=self.save_tracking_stats,
            args=(new_frame, loss),
            kwargs={
                "tracking_time": time.time() - start_time,
                'outputs': outputs,
            },
        )
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
        depthmap_variances: torch.Tensor,
    ):
        self.keyframes = deepcopy(keyframes)
        self.reference_depthmap = depthmap.clone()
        self.reference_frame = self.keyframes[sorted(self.keyframes.keys())[-1]]
        self.reference_rgbs = rgbs
        self.reference_rgbs_np = np.uint8(
            self.reference_frame.img.detach().cpu().numpy() * 255.0
        )
        self.reference_rgbs_np_gray = cv2.cvtColor(
            self.reference_rgbs_np, cv2.COLOR_RGB2GRAY
        )
        self.splats = splats
        self.pose_graph = pose_graph
        if depthmap_variances is not None:
            self.reference_depthmap_variances = depthmap_variances.clone()

        if self.conf.method == 'flow':
            self.flow_points = cv2.goodFeaturesToTrack(
                self.reference_rgbs_np_gray, mask=None, **self.klt_feature_params
            )

        for kf in self.keyframes.values():
            log_frame(kf, name=f'/tracking/kf/{kf.index}')
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

        # dump splats only after 15 frames have passed since the last time we did it
        if (self.frames[-1].index - self.last_time_we_sent_splats_to_rerun) > 15:
            self.dump_pointcloud()
            self.last_time_we_sent_splats_to_rerun = self.frames[-1].index

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
            q = unvmap(matrix_to_quaternion)(pose[:3, :3])
            q = q.detach().cpu().numpy().tolist()
            Rt = pose.detach().cpu().numpy()
            t = Rt[:3, 3]
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

    def save_tracking_stats(
        self,
        new_frame,
        loss,
        outputs: RasterizationOutput = None,
        tracking_time: float = None,
    ):
        i = new_frame.index

        torch_to_pil(new_frame.img).save(self.output_dir / f'gt/{i:08}.jpg')

        if outputs is not None:
            false_colormap(outputs.betas[0], near=0.0, far=2.0).save(
                self.output_dir / f'betas/{i:08}.jpg'
            )

            false_colormap(
                outputs.depthmaps[0],
                near=0.2,
                far=min(2.5, outputs.depthmaps[0].max().item()),
            ).save(self.output_dir / f'depths/{i:08}.jpg')

            torch_to_pil(outputs.rgbs[0]).save(self.output_dir / f"renders/{i:08}.jpg")

    def handle_message_from_backend(self, message):
        match message:
            case [
                BackendMessage.SYNC,
                keyframes,
                depthmap,
                rgbs,
                splats,
                pose_graph,
                depthmap_variances,
            ]:
                self.sync(
                    keyframes, depthmap, rgbs, splats, pose_graph, depthmap_variances
                )
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
        rr.send_blueprint(get_blueprint())

        self.warp = None

        self.waiting_for_end_sync = False
        self.waiting_for_sync = False

        last_time_we_heard_from_backend = time.time()
        self.last_time_we_sent_splats_to_rerun = -100000000

        self.done = False

        if self.conf.method == 'flow':
            import cv2

            self.klt_feature_params = dict(
                maxCorners=500, qualityLevel=0.1, minDistance=7, blockSize=3
            )
            self.lk_params = dict(
                winSize=(15, 15),
                maxLevel=4,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
            )
            self.colors = np.random.randint(0, 255, (500, 3))

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
                    self.save_trajectories()

                    rr.log(
                        '/tracking/ate/pg', rr.Scalar(metrics.get('ate_keyframes', 0.0))
                    )
                    rr.log(
                        '/tracking/ate/tracking',
                        rr.Scalar(metrics.get('ate_tracking', 0.0)),
                    )

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

    def warp_track(self, new_frame, n_iters, optimizer, scheduler):
        _loss = 0.0
        for i in (
            _pbar := tqdm.trange(n_iters, desc=f"[Tracking] frame {len(self.frames)}")
        ):
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

            # _loss = loss.item()

            # loss += new_frame.pose.se3.norm() * self.conf.pose_regularization

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # drdt = new_frame.pose.se3.detach().cpu().numpy()
            # pbar.set_description(
            #     f"[Tracking] frame {len(self.frames)}| loss: {_loss:.8f} "
            # )
            # new_frame.pose.normalize()

            # if np.linalg.norm(drdt) < 2e-4:
            #     break

            if (i + 1) == n_iters:
                _loss = loss.item()
        return _loss

    def igs_track(self, new_frame, n_iters, optimizer, scheduler):
        _loss = 0.0
        for i in (
            _pbar := tqdm.trange(n_iters, desc=f"[Tracking] frame {len(self.frames)}")
        ):
            outputs = self.splats(
                [new_frame.camera], [new_frame.pose], render_depth=True
            )
            rendered_rgb = outputs.rgbs[0]
            betas = outputs.betas[0]
            loss = self.tracking_loss(rendered_rgb, new_frame.img, betas)

            # _loss = loss.item()

            # loss += new_frame.pose.se3.norm() * self.conf.pose_regularization

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # drdt = new_frame.pose.se3.detach().cpu().numpy()
            # pbar.set_description(
            #     f"[Tracking] frame {len(self.frames)}| loss: {_loss:.8f} "
            # )
            # new_frame.pose.normalize()

            # if np.linalg.norm(drdt) < 2e-4:
            #     break
            if (i + 1) == n_iters:
                _loss = loss.item()
        return _loss

    def flow_track(self, new_frame: Frame):
        device = 'cpu'

        new_frame_np = np.uint8(new_frame.img.detach().cpu().numpy() * 255.0)
        new_gray = cv2.cvtColor(new_frame_np, cv2.COLOR_RGB2GRAY)
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            self.reference_rgbs_np_gray,
            new_gray,
            self.flow_points,
            None,
            **self.lk_params,
        )
        good_old = self.flow_points[st == 1]
        good_new = p1[st == 1]
        err = err[st == 1]

        good_old_indices = np.roll(np.int32(good_old), 1, 1)
        depths = self.reference_depthmap.cpu().numpy()[
            good_old_indices[..., 0], good_old_indices[..., 1]
        ]

        good_new = np.roll(good_new, 1, 1)
        good_old = np.roll(good_old, 1, 1)

        height, width = new_gray.shape
        keep_mask = (good_new[..., 0] < height) & (good_new[..., 0] > 0)
        keep_mask = (good_new[..., 1] < width) & (good_new[..., 1] > 0) & keep_mask
        keep_mask = (err < np.quantile(err, 0.8)) & keep_mask
        good_old = good_old[keep_mask]
        good_new = good_new[keep_mask]
        depths = depths[keep_mask]
        err = err[keep_mask]

        # fpdb.set_trace()

        self.flow_points = np.roll(good_old, 1, 1).reshape(-1, 1, 2)

        frame = new_frame_np.copy()
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            b, a = new.astype(np.int32).ravel()
            d, c = old.astype(np.int32).ravel()
            frame = cv2.line(frame, (a, b), (c, d), self.colors[i].tolist(), 2)
            frame = cv2.circle(frame, (c, d), 5, self.colors[i].tolist(), -1)
        rr.log('/tracking/flow', rr.Image(frame).compress(90))

        K = new_frame.camera.intrinsics
        # init_T = pp.mat2SE3(self.reference_frame.pose()).to(device).requires_grad_(True)
        # init from keyframe pose. we'll have to do the constant vel thing maybe.
        init_T = pp.identity_SE3(requires_grad=True, device=device)
        graph = LocalBundleAdjustment(
            K,
            torch.from_numpy(good_old),
            torch.from_numpy(good_new),
            torch.from_numpy(depths),
            init_T,
        ).to(device)

        # optimizer = torch.optim.Adam(graph.parameters(), 1e-2)
        # for i in (pbar := tqdm.trange(100)):
        #     loss = graph().square().mean()
        #     residual_square = graph().square().sum(dim=-1)
        #     loss = F.huber_loss(residual_square, torch.zeros_like(residual_square))
        #     loss.backward()
        #     pbar.set_description(f'Tracking {loss.item()=}')
        #     optimizer.step()
        #     optimizer.zero_grad()
        # new_frame.pose = Pose(initial_pose=graph.T.matrix().detach().to(self.conf.device) @ self.reference_frame.pose()).to(self.conf.device)
        # return loss

        kernel = Huber(delta=1.0)
        corrector = FastTriggs(kernel)
        optimizer = LM(
            graph,
            solver=Cholesky(),
            strategy=TrustRegion(radius=1e3),
            kernel=kernel,
            corrector=corrector,
            min=1e-3,
            reject=128,
            vectorize=True,
        )

        scheduler = StopOnPlateau(
            optimizer, steps=100, patience=100, decreasing=1e-3, verbose=False
        )

        pbar = tqdm.tqdm(desc=f'Tracking frame {new_frame.index}')
        while scheduler.continual():
            loss = optimizer.step(input=())
            scheduler.step(loss)
            pbar.update(1)
        pbar.close()

        new_frame.pose = Pose(
            initial_pose=graph.T.matrix().detach().to(self.conf.device)
            @ self.reference_frame.pose()
        ).to(self.conf.device)

        return loss
