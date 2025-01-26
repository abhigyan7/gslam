from copy import deepcopy
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
from threading import Event
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
from .rasterization import RasterizationOutput
from .messages import BackendMessage, FrontendMessage
from .primitives import Frame, Pose
from .trajectory import evaluate_trajectories
from .utils import (
    get_projection_matrix,
    q_get,
    torch_image_to_np,
    torch_to_pil,
    false_colormap,
    ForkedPdb,
)
from .warp import get_jit_warp


import numpy as np

fpdb = ForkedPdb()


@dataclass
class TrackingConfig:
    device: str = 'cuda'
    num_tracking_iters: int = 50
    photometric_loss: Literal['l1', 'mse', 'active-nerf'] = 'l1'
    pose_optim_lr_translation: float = 0.001
    pose_optim_lr_rotation: float = 0.003

    kf_cov = 0.7
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
        os.makedirs(self.output_dir / 'final_depths', exist_ok=True)

    def to_insert_keyframe(
        self,
        previous_keyframe: Frame,
        new_frame: Frame,
        render_outputs: RasterizationOutput,
    ):
        n_visible_gaussians = new_frame.visible_gaussians.sum()
        n_visible_gaussians_last_kf = previous_keyframe.visible_gaussians.sum()

        intersection = torch.logical_and(
            new_frame.visible_gaussians, previous_keyframe.visible_gaussians
        )
        union = torch.logical_or(
            new_frame.visible_gaussians, previous_keyframe.visible_gaussians
        )

        iou = intersection.sum() / union.sum()
        _oc = intersection.sum() / (
            min(n_visible_gaussians.sum().item(), n_visible_gaussians_last_kf.sum())
        )
        if iou < self.conf.kf_cov:
            return True
        pose_difference = torch.linalg.inv(new_frame.pose()) @ previous_keyframe.pose()
        translation = pose_difference[:3, 3].pow(2.0).sum().pow(0.5).item()
        median_depth = render_outputs.depthmaps[
            render_outputs.alphas[..., 0] > 0.1
        ].median()
        if translation > self.conf.kf_m * median_depth:
            return True
        return False

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

    def track(self, new_frame: Frame):
        with torch.no_grad():
            if len(self.frames) < 2:
                previous_frame = self.frames[-1]
                pose = previous_frame.pose()
            else:
                # constant velocity propagation model
                d_pose = self.frames[-1].pose() @ torch.linalg.inv(
                    self.frames[-2].pose()
                )
                pose = self.frames[-1].pose() @ d_pose

        new_frame.pose = Pose(pose.detach()).to(self.conf.device)
        pose_optimizer = torch.optim.Adam(
            [
                {'params': [new_frame.pose.dR], 'lr': self.conf.pose_optim_lr_rotation},
                {
                    'params': [new_frame.pose.dt],
                    'lr': self.conf.pose_optim_lr_translation,
                },
            ]
        )

        last_loss = float('inf')

        for i in (pbar := tqdm.trange(self.conf.num_tracking_iters)):
            pose_optimizer.zero_grad()
            outputs = self.splats([new_frame.camera], [new_frame.pose])

            rendered_rgb = outputs.rgbs[0]
            betas = outputs.betas[0]
            loss = self.tracking_loss(rendered_rgb, new_frame.img, betas)
            loss.backward()
            pose_optimizer.step()

            if 0 < ((last_loss - loss) / loss) < 0.0001:
                # we've 'converged'!
                pbar.set_description(
                    f"[Tracking] frame {len(self.frames)}, loss: {loss.item():.3f}"
                )
                break

            last_loss = loss.item()

        rr.log(
            '/tracking/loss',
            rr.Scalar(last_loss),
        )

        with torch.no_grad():
            outputs = self.splats(
                [new_frame.camera], [new_frame.pose], render_depth=True
            )
            rendered_rgb = outputs.rgbs[0]
            rendered_depth = outputs.depthmaps[0]
            rendered_beta = outputs.betas[0]

            new_frame.visible_gaussians = outputs.n_touched.sum(dim=0) > 0

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

        false_colormap(outputs.alphas[0, ..., 0]).save(
            self.output_dir / f'alphas/{len(self.frames):08}.jpg'
        )

        false_colormap(rendered_depth, mask=outputs.alphas[0, ..., 0] > 0.3).save(
            self.output_dir / f'depths/{len(self.frames):08}.jpg'
        )

        false_colormap(rendered_beta).save(
            self.output_dir / f'betas/{len(self.frames):08}.jpg'
        )

        self.frames.append(new_frame.strip())

        last_kf = list(self.keyframes.values())[-1]
        if self.to_insert_keyframe(last_kf, new_frame, outputs):
            self.add_keyframe(new_frame)

        return new_frame.pose()

    def warp_track(self, new_frame: Frame):
        with torch.no_grad():
            if len(self.frames) < 2:
                previous_frame = self.frames[-1]
                pose = previous_frame.pose()
            else:
                # constant velocity propagation model
                d_pose = self.frames[-1].pose() @ torch.linalg.inv(
                    self.frames[-2].pose()
                )
                pose = self.frames[-1].pose() @ d_pose

        new_frame = new_frame.to('cpu')
        new_frame.pose = Pose(pose.detach()).to(self.conf.device).cpu()
        pose_optimizer = torch.optim.Adam(
            [
                {'params': [new_frame.pose.dR], 'lr': self.conf.pose_optim_lr_rotation},
                {
                    'params': [new_frame.pose.dt],
                    'lr': self.conf.pose_optim_lr_translation,
                },
            ]
        )

        last_loss = float('inf')

        last_keyframe = self.keyframes[sorted(self.keyframes.keys())[-1]]
        with torch.no_grad():
            outputs = self.splats(
                [last_keyframe.camera], [last_keyframe.pose], render_depth=True
            )
            rgb = outputs.rgbs[0].cpu()
            depthmap = outputs.depthmaps[0].cpu()

        last_keyframe = last_keyframe.to('cpu')
        last_keyframe.pose = last_keyframe.pose.to('cpu')

        for i in (pbar := tqdm.trange(self.conf.num_tracking_iters)):
            pose_optimizer.zero_grad()

            result, _normalized_warps, keep_mask = self.warp_jit(
                last_keyframe.pose(),
                new_frame.pose(),
                last_keyframe.camera.intrinsics.cpu(),
                rgb,
                depthmap,
            )

            result = result[keep_mask, ...]
            gt = new_frame.img[keep_mask, ...]
            # loss = F.huber_loss(result, gt)
            loss = F.l1_loss(result, gt)
            loss.backward()
            pose_optimizer.step()

            if 0 < ((last_loss - loss) / loss) < 0.0001:
                # we've 'converged'!
                pbar.set_description(
                    f"[Tracking] frame {len(self.frames)}, loss: {loss.item():.3f}"
                )
                break

            last_loss = loss.item()

        new_frame = new_frame.to(self.conf.device)
        new_frame.pose = new_frame.pose.to(self.conf.device)
        last_keyframe = last_keyframe.to(self.conf.device)
        last_keyframe.pose = last_keyframe.pose.to(self.conf.device)
        with torch.no_grad():
            outputs = self.splats(
                [new_frame.camera], [new_frame.pose], render_depth=True
            )
            rendered_rgb = outputs.rgbs[0]
            rendered_depth = outputs.depthmaps[0]
            rendered_beta = outputs.betas[0]

            new_frame.visible_gaussians = outputs.n_touched.sum(dim=0) > 0

        rr.log(
            '/tracking/loss',
            rr.Scalar(loss.item()),
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

        false_colormap(outputs.alphas[0, ..., 0]).save(
            self.output_dir / f'alphas/{len(self.frames):08}.jpg'
        )

        false_colormap(rendered_depth, mask=outputs.alphas[0, ..., 0] > 0.3).save(
            self.output_dir / f'depths/{len(self.frames):08}.jpg'
        )

        false_colormap(rendered_beta).save(
            self.output_dir / f'betas/{len(self.frames):08}.jpg'
        )

        self.frames.append(new_frame.strip())

        last_kf = list(self.keyframes.values())[-1]
        if self.to_insert_keyframe(last_kf, new_frame, outputs):
            self.add_keyframe(new_frame)

        return new_frame.pose()

    def request_initialization(self, frame: Frame):
        self.frames.append(frame.strip())
        self.frozen_keyframes.append(frame.strip())
        assert not self.initialized
        self.map_queue.put([FrontendMessage.REQUEST_INITIALIZE, deepcopy(frame)])

    def add_keyframe(self, frame: Frame):
        assert self.initialized
        self.frozen_keyframes.append(frame)
        self.map_queue.put([FrontendMessage.ADD_KEYFRAME, deepcopy(frame)])
        self.waiting_for_sync = True

    def sync_maps(self, splats: GaussianSplattingData, keyframes: dict[int, Frame]):
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

    @rr.shutdown_at_exit
    def run(self):
        rr.init('gslam', recording_id='gslam_1')
        rr.save(self.output_dir / 'rr-fe.rrd')
        rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

        self.Ks = get_projection_matrix().to(self.conf.device)
        self.warp_jit = get_jit_warp(self.conf.device)

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
                self.keyframes[frame.index] = frame
                self.waiting_for_sync = True
            else:
                # self.track(frame)
                self.warp_track(frame)

        self.backend_done_event.wait()
        self.logger.warning('Got backend done.')

        self.dump_pointcloud()
        metrics = dict()
        metrics.update(self.evaluate_reconstruction())
        metrics.update(self.evaluate_trajectory())
        self.create_videos()

        print(f'{metrics=}')
        with open(self.output_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f)

        checkpoint_file = self.output_dir / 'splats.ckpt'
        torch.save(self.splats, checkpoint_file)
        print(f'Saved Checkpoints to {checkpoint_file}')

        self.logger.warning('frontend done.')

        self.frontend_done_event.set()
