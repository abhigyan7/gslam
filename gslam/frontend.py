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

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from .data import OakdSensor
from .map import GaussianSplattingData
from .messages import BackendMessage, FrontendMessage
from .primitives import Frame, PoseZhou as Pose, matrix_to_quaternion
from .rasterization import RasterizationOutput
from .trajectory import evaluate_trajectories
from .utils import (
    torch_to_pil,
    false_colormap,
    unvmap,
    StopOnPlateau,
    torch_image_to_np,
    # ForkedPdb,
)
from .visualization import log_frame, log_splats
from .warp import Warp

plt.switch_backend('agg')


@dataclass
class TrackingConfig:
    device: str = 'cuda'
    num_tracking_iters: int = 200
    photometric_loss: Literal['l1', 'mse', 'active-nerf'] = 'active-nerf'

    pose_optim_lr: float = 0.002
    pose_optim_lr_decay: float = 0.99

    method: Literal['igs', 'warp'] = 'igs'

    pose_regularization: float = 0

    learn_exposure_params: bool = True

    use_gt_depths: bool = False

    traj_interval: float = 0.4


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
        global_pause_event: Event = None,
        run_name: str = 'fe',
        is_oak: bool = False,
    ):
        super().__init__()
        self.conf: TrackingConfig = conf
        self.map_queue: mp.Queue = backend_queue
        self.queue: mp.Queue[int] = frontend_queue
        self.keyframes: dict[int, Frame] = dict()
        self.run_name: str = run_name
        self.is_oak = is_oak

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
        self.global_pause_event = global_pause_event

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
        rendered_depth: torch.Tensor = None,
        gt_depth: torch.Tensor = None,
    ) -> torch.Tensor:
        error = rendered_img - gt_img
        loss = None
        match self.conf.photometric_loss:
            case 'l1':
                loss = error.abs().mean()
            case 'mse':
                loss = error.square().mean()
            case 'active-nerf':
                loss = (error.square().sum(dim=-1) * betas.pow(-2.0)).mean()
            case 'none':
                loss = error
            case _:
                assert_never(self.conf.photometric_loss)
        if self.conf.use_gt_depths:
            depth_error = (rendered_depth - gt_depth)[gt_depth > 0.0]
            depth_loss = depth_error.abs().mean()
            loss += depth_loss * 0.01
        return loss

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
        new_frame.exposure_params = torch.nn.Parameter(
            torch.zeros([2], device=new_frame.img.device, requires_grad=False)
        )

        if self.conf.method == 'igs':
            self.request_initialization(new_frame)
        return

    def request_initialization(self, f: Frame):
        self.map_queue.put((FrontendMessage.REQUEST_INIT, deepcopy(f)))
        self.waiting_for_sync = True

    def track(self, new_frame: Frame):
        n_iters = self.conf.num_tracking_iters

        if len(self.frames) == 0:
            self.initialize(new_frame)
            n_iters = 0
        elif len(self.frames) == 1:
            pose = self.frames[-1].pose()
        else:
            # constant motion model
            pose_a = self.frames[-2].pose()
            pose_b = self.frames[-1].pose()
            pose = pose_b @ torch.linalg.inv(pose_a) @ pose_b

        new_frame.exposure_params = torch.nn.Parameter(
            torch.zeros(
                [2],
                device=new_frame.img.device,
                requires_grad=self.conf.learn_exposure_params,
            )
        )

        if n_iters > 0:
            new_frame.pose = Pose(pose.detach()).to(self.conf.device)
            if self.conf.method in ['warp', 'igs']:
                optimizer = torch.optim.SGD(
                    new_frame.pose.parameters(),
                    self.conf.pose_optim_lr,
                    fused=True,
                    momentum=0.8,
                    nesterov=True,
                )
                scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    optimizer, self.conf.pose_optim_lr_decay
                )
                if self.conf.learn_exposure_params:
                    optimizer.add_param_group(
                        {
                            'params': new_frame.exposure_params,
                            'lr': 0.01,
                        }
                    )

        loss = 0.0
        outputs = None
        start_time = time.time()

        # self.trajectory.extend_to_time(new_frame.timestamp)

        if n_iters > 0:
            if self.conf.method == 'warp':
                loss = self.warp_track(new_frame, n_iters, optimizer, scheduler)
            elif self.conf.method == 'igs':
                loss = self.igs_track_lbfgs(new_frame, n_iters)
                # loss = self.igs_track(new_frame, n_iters, optimizer, scheduler)
            else:
                assert_never(self.conf.method)

        outputs = None
        if hasattr(self, 'splats'):
            outputs = self.splats(
                [new_frame.camera], [new_frame.pose], render_depth=True
            )
        self.frames.append(new_frame.strip())
        if new_frame.index > 0:
            self.add_frame_to_backend(new_frame)

        Thread(
            target=log_frame,
            args=(new_frame,),
            kwargs={
                "outputs": outputs,
                "loss": loss,
                "tracking_time": time.time() - start_time if n_iters > 0 else None,
                "is_tracking_frame": True,
            },
        ).start()

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
        self.reference_frame = self.keyframes[sorted(self.keyframes.keys())[-1]]
        self.reference_rgbs = rgbs
        self.splats = deepcopy(splats)
        self.pose_graph = pose_graph
        # self.trajectory = trajectory
        print('FE/BE Sync')
        return

    def sync_at_end(self, splats: GaussianSplattingData, keyframes: dict[int, Frame]):
        self.splats, self.keyframes = splats, deepcopy(keyframes)
        return

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

    @torch.no_grad()
    def evaluate_reconstruction(self):
        for i, kf in enumerate(
            tqdm.tqdm(self.keyframes.values(), 'Rendering all keyframes')
        ):
            outputs = self.splats(
                [kf.camera],
                [kf.pose],
            )
            # torch_to_pil(outputs.rgbs[0]).save(self.output_dir / f'final/{i:08}.jpg')

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

            rgb_np = torch_image_to_np(outputs.rgbs[0])
            # false_colormap(
            #     outputs.depthmaps[0],
            #     near=0.0,
            #     far=2.0,
            # ).save(self.output_dir / f'final_depths/{i:08}.jpg')
            # Image.fromarray(rgb_np).save(self.output_dir / f'final_renders/{i:08}.jpg')

            if f.img_file is None:
                continue
            gt_img = np.array(Image.open(f.img_file))
            psnrs.append(
                psnr(
                    rgb_np,
                    gt_img,
                )
            )
            ssims.append(
                ssim(
                    rgb_np,
                    gt_img,
                    channel_axis=2,
                )
            )

        return {'ssim': np.mean(ssims), 'psnr': np.mean(psnrs)}

    def handle_message_from_backend(self, message):
        match message:
            case [
                BackendMessage.SYNC,
                keyframes,
                depthmap,
                rgbs,
                splats,
                pose_graph,
            ]:
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
        rr.init('gslam', recording_id=self.run_name, spawn=True)
        # rr.save(self.output_dir / 'rr-fe.rrd')
        # rr.log("/tracking", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
        # rr.send_blueprint(get_blueprint())

        if self.is_oak:
            print('oak')
            self.sensor = OakdSensor(fps=15.0)
            self.sensor.start()
            self.sensor_queue = self.sensor.output_queue

        self.warp = None

        self.waiting_for_end_sync = False
        self.waiting_for_sync = False

        last_time_we_heard_from_backend = time.time()

        self.done = False

        while True:
            if not self.queue.empty():
                print('backend is not empty')
                self.handle_message_from_backend(self.queue.get())
                last_time_we_heard_from_backend = time.time()

            if self.waiting_for_end_sync:
                if (time.time() - last_time_we_heard_from_backend) > 3000.0:
                    print('Looks like backend\'s dead')
                    break
                continue

            if self.waiting_for_sync:
                continue

            if self.global_pause_event.is_set():
                self.global_pause_event.wait()
                continue

            if self.done:
                break

            queue_is_empty = False
            if hasattr(self.sensor_queue, "empty") and self.sensor_queue.empty():
                queue_is_empty = True
            if hasattr(self.sensor_queue, "has") and not self.sensor_queue.has():
                queue_is_empty = True
            if not queue_is_empty:
                if self.is_oak:
                    frame: Frame = self.sensor.next()
                else:
                    frame: Frame = self.sensor_queue.get()
                if frame is None:
                    # data stream exhausted
                    self.map_queue.put(None)
                    self.waiting_for_end_sync = True
                    last_time_we_heard_from_backend = time.time()
                    continue

                frame = frame.to(self.conf.device)
                self.track(frame)

                if len(self.frames) % 30 == 0:
                    checkpoint_file = self.output_dir / 'splats.ckpt'
                    torch.save(self.splats, checkpoint_file)
                    print(f'Saved Checkpoints to {checkpoint_file}')

        self.backend_done_event.wait()
        self.logger.warning('Got backend done.')
        log_splats(self.splats)
        metrics = dict()

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
        if self.warp is None:
            self.warp = Warp(
                new_frame.camera.intrinsics,
                new_frame.camera.height,
                new_frame.camera.width,
            )
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
            if self.conf.learn_exposure_params:
                rendered_rgb = (
                    rendered_rgb * new_frame.exposure_params[0].exp()
                    + new_frame.exposure_params[1]
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

    def igs_track(self, new_frame: Frame, n_iters, optimizer, scheduler):
        _loss = 0.0
        stop_criterion = StopOnPlateau(20, 0.1)
        for i in (
            pbar := tqdm.trange(n_iters, desc=f"[Tracking] frame {len(self.frames)}")
        ):
            outputs = self.splats(
                [new_frame.camera], [new_frame.pose], render_depth=True
            )
            if self.conf.learn_exposure_params:
                rendered_rgb = (
                    outputs.rgbs[0] * new_frame.exposure_params[0].exp()
                    + new_frame.exposure_params[1]
                )
            betas = outputs.betas[0]
            loss = self.tracking_loss(rendered_rgb, new_frame.img, betas)

            _loss = loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            pbar.set_description(
                f"[Tracking] frame {len(self.frames)}| loss: {_loss:.8f} "
            )

            if stop_criterion.stop(_loss):
                print(f'Early stop at step {i}, loss: {_loss}')
                break
        return loss

    def igs_track_lbfgs(self, new_frame: Frame, n_iters):
        start_time = time.time()
        n_iters = 0
        params = list(new_frame.pose.parameters())
        if self.conf.learn_exposure_params:
            new_frame.exposure_params.data = (
                self.frames[-1].exposure_params.clone().detach()
            )
            params.append(new_frame.exposure_params)
        optimizer = torch.optim.LBFGS(
            params,
            history_size=5,
            line_search_fn='strong_wolfe',
            tolerance_change=1e-9,
            lr=self.conf.pose_optim_lr,
        )
        last_loss = None

        def closure():
            nonlocal n_iters, last_loss
            n_iters += 1
            if torch.is_grad_enabled():
                optimizer.zero_grad()
            outputs: RasterizationOutput = self.splats(
                [new_frame.camera],
                [new_frame.pose],
                render_depth=True,
            )
            if self.conf.learn_exposure_params:
                rendered_rgb = (
                    outputs.rgbs[0] * new_frame.exposure_params[0].exp()
                    + new_frame.exposure_params[1]
                )
            betas = outputs.betas[0]
            loss = self.tracking_loss(
                rendered_rgb,
                new_frame.img,
                betas,
                outputs.depthmaps[0],
                new_frame.gt_depth,
            )
            if loss.requires_grad:
                loss.backward()
            # this sync is okay because lbfgs syncs anyway
            last_loss = loss.item()
            return loss

        sgd_optimizer = torch.optim.Adam(params, self.conf.pose_optim_lr)
        for i in range(5):
            loss = closure()
            sgd_optimizer.step()
            sgd_optimizer.zero_grad()
            print(f'SGD: {i=}, {loss.item()=}')

        optimizer.step(closure)
        print(
            f'LBFGS: {new_frame.index} {last_loss=} {n_iters=}, time: {((time.time() - start_time) * 1000.0):.1f}ms'
        )
        return last_loss
