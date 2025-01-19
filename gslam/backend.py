from dataclasses import dataclass
import logging
import random
import threading
from copy import deepcopy
from typing import Dict, List

from fused_ssim import fused_ssim
import torch
import tqdm

from .insertion import InsertFromDepthMap, InsertUsingImagePlaneGradients
from .map import GaussianSplattingData
from .messages import BackendMessage, FrontendMessage
from .primitives import Frame, Pose
from .pruning import PruneLowOpacity, PruneLargeGaussians
from .rasterization import RasterizationOutput
from .utils import create_batch


@dataclass
class MapConfig:
    isotropic_regularization_weight: float = 10.0
    opacity_regularization_weight: float = 0.000005
    betas_regularization_weight: float = 200.0
    depth_regularization_weight: float = 0.0001

    pose_optim_lr_translation: float = 0.001
    pose_optim_lr_rotation: float = 0.003
    pose_optimization_regularization: float = 1e-6

    # 3dgs schedules means_lr, might need to look into this
    means_lr: float = 0.00016
    opacity_lr: float = 0.025
    scale_lr: float = 0.005
    color_lr: float = 0.005
    quat_lr: float = 0.005
    beta_lr: float = 0.00005

    # background rgb
    background_color: tuple = (0.0, 0.0, 0.0)

    initial_number_of_gaussians: int = 10_000
    initial_opacity: float = 0.9
    initial_scale: float = 2.0
    initial_extent: float = 1.0
    initial_beta: float = 0.3

    device: str = 'cuda:0'

    optim_window_last_n_keyframes: int = 5
    optim_window_random_keyframes: int = 5

    num_iters_mapping: int = 150
    num_iters_initialization: int = 300

    opacity_pruning_threshold: float = 0.6
    size_pruning_threshold: int = 256

    reset_opacity: bool = False
    opacity_after_reset: float = 0.5

    # used in the final optimization
    ssim_weight: float = 0.2  # in [0,1]
    num_iters_final: int = 200

    use_betas: bool = True


def total_variation_loss(img: torch.Tensor) -> torch.Tensor:
    tv_h = (img[..., 1:, :] - img[..., :-1, :]).pow(2).sum()
    tv_w = (img[..., :, 1:] - img[..., :, :-1]).pow(2).sum()
    return tv_h + tv_w


class Backend(torch.multiprocessing.Process):
    def __init__(
        self,
        conf: MapConfig,
        queue: torch.multiprocessing.Queue,
        frontend_queue: torch.multiprocessing.Queue,
        backend_done_event: threading.Event,
    ):
        super().__init__()
        self.conf = conf
        self.queue: torch.multiprocessing.Queue = queue
        self.frontend_queue = frontend_queue
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel("DEBUG")
        self.keyframes: List[Frame] = []
        self.backend_done_event = backend_done_event
        self.splats = GaussianSplattingData.empty()
        self.pruning_opacity = PruneLowOpacity(self.conf.opacity_pruning_threshold)
        self.pruning_size = PruneLargeGaussians(self.conf.size_pruning_threshold)
        self.insertion_depth_map = InsertFromDepthMap(
            0.2, 0.5, 0.1, self.conf.initial_opacity, self.conf.initial_beta
        )
        self.insertion_3dgs = InsertUsingImagePlaneGradients(
            0.0002,
            0.01,
        )
        self.initialized: bool = False

        if not self.conf.use_betas:
            self.conf.betas_regularization_weight = 0.0

    def optimization_window(self):
        window_size_total = (
            self.conf.optim_window_last_n_keyframes
            + self.conf.optim_window_random_keyframes
        )
        n_keyframes_total = len(self.keyframes)
        n_last_keyframes_to_choose = min(
            n_keyframes_total, self.conf.optim_window_last_n_keyframes
        )
        n_random_keyframes_to_choose = min(
            0, window_size_total - n_last_keyframes_to_choose
        )

        window = []

        window = self.keyframes[-n_last_keyframes_to_choose:]
        window.extend(
            random.sample(
                self.keyframes[:-n_last_keyframes_to_choose],
                n_random_keyframes_to_choose,
            )
        )
        return window

    def optimize_map(self):
        n_iters = (
            self.conf.num_iters_mapping
            if self.initialized
            else self.conf.num_iters_initialization
        )

        if self.conf.reset_opacity:
            with torch.no_grad():
                # reset opacities. not sure if no_grad is needed here.
                self.splats.opacities.data = torch.logit(
                    torch.sigmoid(self.splats.opacities.data) * 0.5
                )

        for step in (pbar := tqdm.trange(n_iters)):
            window = self.optimization_window()
            cameras = [x.camera for x in window]
            poses = torch.nn.ModuleList([x.pose for x in window])
            gt_imgs = create_batch(window, lambda x: x.img)
            self.zero_grad_all_optimizers()

            outputs: RasterizationOutput = self.splats(
                cameras,
                poses,
                render_depth=True,
            )

            # unsqueeze so that broadcast multiplication works out
            inv_betas = 1.0
            if self.conf.use_betas:
                inv_betas = outputs.betas.pow(-1.0).unsqueeze(-1)
            photometric_loss = ((outputs.rgbs - gt_imgs) * inv_betas).square().mean()

            visible_gaussians = outputs.radii.sum(dim=0) > 0
            mean_scales = (
                self.splats.scales[visible_gaussians]
                .mean(dim=1, keepdim=True)
                .exp()
                .detach()
            )
            isotropic_loss = (
                (self.splats.scales.exp()[visible_gaussians] - mean_scales).abs().mean()
            )
            betas_loss = self.splats.betas[visible_gaussians].mean()
            opacity_loss = self.splats.opacities[visible_gaussians].mean()
            depth_loss = total_variation_loss(outputs.depthmaps)
            ssim_loss = 1.0 - fused_ssim(
                outputs.rgbs.permute(0, 3, 1, 2),
                gt_imgs.permute(0, 3, 1, 2),
                padding='valid',
            )
            total_loss = (
                photometric_loss
                + self.conf.isotropic_regularization_weight * isotropic_loss
                + self.conf.opacity_regularization_weight * opacity_loss
                + self.conf.betas_regularization_weight * betas_loss
                + self.conf.depth_regularization_weight * depth_loss
                + self.conf.ssim_weight * ssim_loss
            )

            outputs.means2d.retain_grad()

            total_loss.backward()

            if step in (n_iters // 3, 2 * n_iters // 3):
                self.insertion_3dgs.step(
                    self.splats,
                    self.splat_optimizers,
                    outputs,
                    None,
                    None,
                )

            desc = (
                f"[Mapping] loss={photometric_loss.item():.3f}, "
                f"n_splats={self.splats.means.shape[0]:07}, "
                f"mean_beta={self.splats.betas.exp().mean().item():.3f}"
            )
            pbar.set_description(desc)

            self.step_all_optimizers()

            self.pruning_size.step(
                self.splats,
                self.splat_optimizers,
                outputs.radii,
            )

            if step in (n_iters // 2, n_iters):
                self.pruning_opacity.step(self.splats, self.splat_optimizers)

        outputs = self.splats(
            [self.keyframes[-1].camera],
            [self.keyframes[-1].pose],
        )

        self.keyframes[-1].visible_gaussians = (outputs.radii[0] > 0).detach()
        return

    def optimize_final(self):
        n_iters = self.conf.num_iters_final

        with torch.no_grad():
            # reset opacities. not sure if no_grad is needed here.
            self.splats.opacities.data = torch.logit(
                torch.sigmoid(self.splats.opacities.data) * 0.0
                + self.conf.opacity_after_reset
            )
        for step in (pbar := tqdm.trange(n_iters)):
            window = random.sample(self.keyframes, min(10, len(self.keyframes)))
            cameras = [x.camera for x in window]
            poses = torch.nn.ModuleList([x.pose for x in window])
            gt_imgs = create_batch(window, lambda x: x.img)
            self.zero_grad_all_optimizers()

            outputs = self.splats(
                cameras,
                poses,
            )

            photometric_loss = (outputs.rgbs - gt_imgs).square().mean()
            ssim_loss = 1.0 - fused_ssim(
                outputs.rgbs.permute(0, 3, 1, 2),
                gt_imgs.permute(0, 3, 1, 2),
                padding='valid',
            )
            loss = (
                self.conf.ssim_weight * ssim_loss
                + (1.0 - self.conf.ssim_weight) * photometric_loss
            )
            loss.backward()

            if ((step + 1) % (n_iters // 3)) == 0:
                self.pruning_opacity.step(self.splats, self.splat_optimizers)

            desc = (
                f"[Mapping] loss={loss.item():.3f}, "
                f"n_splats={self.splats.means.shape[0]:07}"
            )
            pbar.set_description(desc)

            self.step_all_optimizers()

        return

    def sync_with_frontend(self):
        self.frontend_queue.put(
            (
                BackendMessage.SYNC,
                self.splats.clone(),
                deepcopy(self.keyframes),
            )
        )
        return

    def zero_grad_all_optimizers(self):
        for optimizer in self.splat_optimizers.values():
            optimizer.zero_grad()
        self.pose_optimizer.zero_grad()

    def step_all_optimizers(self):
        for optimizer in self.splat_optimizers.values():
            optimizer.step()
        self.pose_optimizer.step()

    def initialize_optimizers(self):
        # TODO fix these LRs
        self.splat_optimizers: Dict[str, torch.optim.Optimizer] = {}
        self.splat_optimizers['means'] = torch.optim.Adam(
            params=[self.splats.means],
            lr=self.conf.means_lr,
        )
        self.splat_optimizers['quats'] = torch.optim.Adam(
            params=[self.splats.quats],
            lr=self.conf.quat_lr,
        )
        self.splat_optimizers['scales'] = torch.optim.Adam(
            params=[self.splats.scales],
            lr=self.conf.scale_lr,
        )
        self.splat_optimizers['opacities'] = torch.optim.Adam(
            params=[self.splats.opacities],
            lr=self.conf.opacity_lr,
        )
        self.splat_optimizers['colors'] = torch.optim.Adam(
            params=[self.splats.colors],
            lr=self.conf.color_lr,
        )
        self.splat_optimizers['betas'] = torch.optim.Adam(
            params=[self.splats.betas],
            lr=self.conf.beta_lr,
        )
        self.pose_optimizer = torch.optim.Adam(
            params=[torch.empty(0)],
            lr=0.001,
        )

    def initialize(self, frame: Frame):
        frame = frame.to(self.conf.device)
        self.keyframes = [frame]

        self.splats = GaussianSplattingData.empty().to(self.conf.device)
        self.initialize_optimizers()

        H, W, _ = frame.img.shape
        mock_depth_map = torch.ones((1, H, W), device=self.conf.device)
        mock_depth_map = (
            mock_depth_map + (torch.randn_like(mock_depth_map) - 0.5) * 0.05
        )
        mock_depth_map *= self.conf.initial_scale
        mock_alphas = torch.ones((1, H, W, 1), device=self.conf.device)

        mock_outputs = RasterizationOutput(None, mock_alphas, mock_depth_map)

        self.insertion_depth_map.step(
            self.splats,
            self.splat_optimizers,
            mock_outputs,
            frame,
            10000,
        )

        self.optimize_map()
        self.logger.warning('Initialized')
        self.initialized = True

        return

    def add_keyframe(self, frame: Frame):
        with torch.no_grad():
            outputs = self.splats(
                [frame.camera],
                [frame.pose],
                render_depth=True,
            )
        outputs.depthmaps *= self.conf.initial_scale
        self.insertion_depth_map.step(
            self.splats,
            self.splat_optimizers,
            outputs,
            frame,
            N=500,
        )

        new_frame = Frame(
            img=frame.img.clone(),
            timestamp=frame.timestamp,
            camera=frame.camera.clone(),
            pose=Pose(frame.pose()).to(self.conf.device),
            gt_pose=frame.gt_pose,
            gt_depth=frame.gt_depth,
            img_file=frame.img_file,
        )

        self.keyframes.append(new_frame)
        self.pose_optimizer.add_param_group(
            {
                'params': new_frame.pose.dt,
                'lr': self.conf.pose_optim_lr_translation,
            }
        )
        self.pose_optimizer.add_param_group(
            {
                'params': new_frame.pose.dR,
                'lr': self.conf.pose_optim_lr_rotation,
            }
        )

    def run(self):
        while True:
            if self.queue.empty():
                continue
            match self.queue.get():
                case [FrontendMessage.REQUEST_INITIALIZE, frame]:
                    self.initialize(frame)
                    self.frontend_queue.put([BackendMessage.SIGNAL_INITIALIZED])
                    self.sync_with_frontend()
                case [FrontendMessage.ADD_KEYFRAME, frame]:
                    self.add_keyframe(frame)
                    self.optimize_map()
                    self.sync_with_frontend()
                case None:
                    print('Running final optimization.')
                    self.optimize_final()
                    break
                case message_from_frontend:
                    self.logger.warning(f"Unknown {message_from_frontend}")

        self.backend_done_event.set()
