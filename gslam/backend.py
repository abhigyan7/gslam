from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from itertools import combinations
import logging
import math
import random
import threading
import time
from typing import Dict, assert_never

from fused_ssim import fused_ssim
import nerfview
import torch
import tqdm
import viser

import numpy as np

from .insertion import InsertFromDepthMap, InsertUsingImagePlaneGradients
from .map import GaussianSplattingData
from .messages import BackendMessage, FrontendMessage
from .pose_graph import add_constraint
from .primitives import Frame, PoseZhou as Pose, Camera
from .pruning import (
    PruneLowOpacity,
    PruneLargeGaussians,
    PruneIllConditionedGaussians,
    prune_using_mask,
)
from .rasterization import RasterizationOutput
from .utils import create_batch, StopOnPlateau

# from .utils import create_batch,  StopOnPlateau
from .viewer import Viewer
from .warp import Warp


@dataclass
class MapConfig:
    isotropic_regularization_weight: float = 0.0005
    opacity_regularization_weight: float = 0.000005
    depth_regularization_weight: float = 0.000001
    beta_ema_weight: float = 0.98

    pose_optim_lr: float = 0.003

    # 3dgs schedules means_lr, might need to look into this
    means_lr: float = 0.0016
    opacity_lr: float = 0.025
    scale_lr: float = 0.005
    color_lr: float = 0.01
    quat_lr: float = 0.005
    log_uncertainty_lr: float = 0.0025
    # from binocular3DGS
    opacity_decay: float = 0.995

    # background rgb
    background_color: tuple = (0.0, 0.0, 0.0)

    initial_number_of_gaussians: int = 10_000
    initial_opacity: float = 0.3
    initial_scale: float = 1.0

    device: str = 'cuda'

    optim_window_last_n_keyframes: int = 8  # MonoGS does 8
    optim_window_random_keyframes: int = 2

    num_iters_mapping: int = 15
    num_iters_initialization: int = 400

    opacity_pruning_threshold: float = 0.2
    size_pruning_threshold: int = 256

    prune_every: int = 199
    insert_every: int = 600

    reset_opacity: bool = False
    opacity_after_reset: float = 0.5

    # used in the final optimization
    ssim_weight: float = 0.2  # in [0,1]
    num_iters_final: int = 2000

    active_gs: bool = True

    min_visibility: int = 3
    visibility_pruning_window_size: int = 3
    enable_visibility_pruning: bool = False

    # pose graph optimization
    enable_pgo: bool = False
    pgo_loss_weight: float = 0.01

    kf_cov = 0.9
    kf_oc = 0.99
    kf_m = 0.2  # 0.3
    kf_cos = math.cos(math.pi / 30)


def total_variation_loss(img: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    v_h = img[..., 1:, :] - img[..., :-1, :]
    v_w = img[..., :, 1:] - img[..., :, :-1]
    if mask is not None:
        v_h = v_h * mask[..., 1:, :]
        v_w = v_w * mask[..., :, 1:]
    tv_h = (v_h).pow(2).sum()
    tv_w = (v_w).pow(2).sum()
    return tv_h + tv_w


def edge_aware_tv(
    depth: torch.Tensor, rgb: torch.Tensor, mask: torch.Tensor = None
) -> torch.Tensor:
    """
    Args:
        depth: [batch, H, W]
        rgb: [batch, H, W, 3]
        mask: [mask, H, W, 1]
    """
    grad_depth_x = torch.abs(depth[..., :, :-1, None] - depth[..., :, 1:, None])
    grad_depth_y = torch.abs(depth[..., :-1, :, None] - depth[..., 1:, :, None])

    grad_img_x = torch.mean(
        torch.abs(rgb[..., :, :-1, :] - rgb[..., :, 1:, :]), -1, keepdim=True
    )
    grad_img_y = torch.mean(
        torch.abs(rgb[..., :-1, :, :] - rgb[..., 1:, :, :]), -1, keepdim=True
    )

    grad_depth_x *= torch.exp(-grad_img_x)
    grad_depth_y *= torch.exp(-grad_img_y)

    return (
        grad_depth_x[mask[..., :, :-1, None]].sum()
        + grad_depth_y[mask[..., :-1, :, None]].sum()
    )


class Backend(torch.multiprocessing.Process):
    def __init__(
        self,
        conf: MapConfig,
        queue: torch.multiprocessing.Queue,
        frontend_queue: torch.multiprocessing.Queue,
        backend_done_event: threading.Event,
        global_pause_event: threading.Event,
        enable_viser_server: bool = False,
    ):
        super().__init__()
        self.conf = conf
        self.queue: torch.multiprocessing.Queue = queue
        self.frontend_queue = frontend_queue
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel("DEBUG")
        self.keyframes: dict[int, Frame] = dict()
        self.backend_done_event = backend_done_event
        self.global_pause_event = global_pause_event

        self.splats = GaussianSplattingData.empty()
        self.pruning_opacity = PruneLowOpacity(self.conf.opacity_pruning_threshold)
        self.pruning_size = PruneLargeGaussians(self.conf.size_pruning_threshold)
        self.insertion_depth_map = InsertFromDepthMap(
            0.1 * self.conf.initial_scale,
            0.2 * self.conf.initial_scale,
            0.1,
            self.conf.initial_opacity,
            False,  # TODO parameterize this
            global_pause_event=self.global_pause_event,
        )
        self.insertion_3dgs = InsertUsingImagePlaneGradients(
            0.0002,
            0.01,
        )
        self.pruning_conditioning = PruneIllConditionedGaussians(3)

        self.pose_graph = defaultdict(set)
        self.total_step = 0

        self.splats_mutex = torch.multiprocessing.Lock()
        self.enable_viser_server = enable_viser_server

    @torch.no_grad()
    def viewer_render_fn(
        self,
        camera_state: nerfview.CameraState,
        img_wh: tuple[int, int],
    ):
        device = self.conf.device
        width, height = img_wh
        c2w = camera_state.c2w
        K = camera_state.get_K(img_wh)
        c2w = torch.from_numpy(c2w).float().to(device)
        K = torch.from_numpy(K).float().to(device)
        viewmat = c2w.inverse()

        camera = Camera(K, height, width)
        pose = Pose(viewmat, False)

        with self.splats_mutex:
            outputs = self.splats([camera], [pose], True)

        match self.viewer.state.target_type:
            case "rgb":
                render_rgbs = outputs.rgbs[0, ...].cpu().numpy()
                render_depths = outputs.depthmaps[0, ...].cpu().numpy()
                return render_rgbs, render_depths
            case "n_touched":
                render_rgbs = outputs.n_touched[0, ...].cpu().numpy()
                render_rgbs = np.tile(render_rgbs / render_rgbs.max(), (1, 1, 3))
                render_depths = outputs.depthmaps[0, ...].cpu().numpy()
            case x:
                assert_never(x)

        return render_rgbs

    def optimization_window(self) -> list[Frame]:
        window_size_total = (
            self.conf.optim_window_last_n_keyframes
            + self.conf.optim_window_random_keyframes
        )
        n_keyframes_total = len(self.keyframes)

        if self.conf.enable_pgo:
            # get neighborhood from fan out
            latest_keyframe = sorted(self.keyframes.keys())[-1]
            window = set()
            window.add(latest_keyframe)
            neighbors = self.pose_graph[latest_keyframe]
            if 0 > len(neighbors) > window_size_total:
                window.update(random.sample(neighbors, window_size_total))
            elif 0 < len(neighbors):
                window.update(list(neighbors))
            spots_left_in_window = window_size_total - len(window)
            for i in range(spots_left_in_window):
                if len(neighbors) == 0:
                    break
                # ForkedPdb(self.global_pause_event).set_trace()
                neighbors_of_ith_neighbor = self.pose_graph[
                    random.sample(sorted(neighbors), 1)[0]
                ]
                if len(neighbors_of_ith_neighbor) == 0:
                    continue
                random_neighbor_of_ith_neighbor = random.sample(
                    sorted(neighbors_of_ith_neighbor), 1
                )[0]
                if random_neighbor_of_ith_neighbor in window:
                    continue
                window.add(random_neighbor_of_ith_neighbor)
            # print(f'neighbor sample: {window=}')
        else:
            n_last_keyframes_to_choose = min(
                n_keyframes_total, self.conf.optim_window_last_n_keyframes
            )
            n_random_keyframes_to_choose = min(
                0, window_size_total - n_last_keyframes_to_choose
            )

            window = list(self.keyframes.keys())[-n_last_keyframes_to_choose:]
            window.extend(
                random.sample(
                    list(self.keyframes.keys())[:-n_last_keyframes_to_choose],
                    n_random_keyframes_to_choose,
                )
            )
        window = [self.keyframes[i] for i in sorted(window)]
        return window

    def optimize_map(self, n_iters: int = None, prune=True, regularize=True):
        if n_iters is None:
            n_iters = self.conf.num_iters_mapping

        radii = None
        n_touched = None

        early_stopper = StopOnPlateau(3, 0.012)

        for step in (pbar := tqdm.trange(n_iters)):
            self.total_step += 1
            window = self.optimization_window()
            cameras = [x.camera for x in window]
            poses = torch.nn.ModuleList([x.pose for x in window])
            gt_imgs = create_batch(window, lambda x: x.img)
            exposure_params = create_batch(window, lambda x: x.exposure_params)
            self.zero_grad_all_optimizers()

            outputs: RasterizationOutput = self.splats(
                cameras,
                poses,
                render_depth=True,
            )

            rendered_rgbs = outputs.rgbs * exposure_params[..., 0].view(
                -1, 1, 1, 1
            ) + exposure_params[..., 1].view(-1, 1, 1, 1)

            if self.conf.active_gs:
                photometric_loss = (rendered_rgbs - gt_imgs).square().sum(dim=-1)
                photometric_loss = photometric_loss / (2 * outputs.betas.square())
                photometric_loss = photometric_loss.mean()
                photometric_loss = (
                    photometric_loss + (outputs.betas.log().square() * 0.5).mean()
                )
            else:
                photometric_loss = (outputs.rgbs - gt_imgs).square().mean()

            visible_gaussians = outputs.radii.sum(dim=0) > 0
            mean_scales = (
                self.splats.scales[visible_gaussians]
                .mean(dim=1, keepdim=True)
                .exp()
                .detach()
            )
            isotropic_loss = (
                (self.splats.scales.exp()[visible_gaussians] - mean_scales).abs().sum()
            )
            depth_loss = edge_aware_tv(
                outputs.depthmaps,
                outputs.rgbs,
                outputs.alphas[..., 0] > 0.4,
            )
            ssim_loss = 1.0 - fused_ssim(
                outputs.rgbs.permute(0, 3, 1, 2),
                gt_imgs.permute(0, 3, 1, 2),
                padding='valid',
            )
            total_loss = (
                (1.0 - self.conf.ssim_weight) * photometric_loss
                + self.conf.ssim_weight * ssim_loss
                + self.conf.isotropic_regularization_weight * isotropic_loss
            )
            if regularize:
                total_loss += +self.conf.depth_regularization_weight * depth_loss

            outputs.means2d.retain_grad()

            total_loss.backward()

            if (self.total_step % 100) == 0:
                self.insertion_3dgs.step(
                    self.splats,
                    self.splat_optimizers,
                    outputs,
                    None,
                    None,
                )
                prune = False

            desc = (
                f"[Mapping] keyframe {len(self.keyframes)} pm={photometric_loss.item():.4f}, "
                f"loss={total_loss.item():.5f}, "
                f"n_splats={self.splats.means.shape[0]:07}, "
                f"mean_beta={self.splats.log_uncertainties.exp().mean().item():.3f}"
            )
            pbar.set_description(desc)

            self.step_all_optimizers()

            if early_stopper.stop(photometric_loss.item()):
                print('Pausing map optimization')
                self.pause_map_optim = True
                break

            with torch.no_grad():
                self.splats.opacities.data[(outputs.radii > 0).sum(dim=0) > 1] *= (
                    self.conf.opacity_decay
                )

        for f, d in zip(window, outputs.depthmaps):
            f.est_depths = d.detach().clone()

        gaussians_max_screen_size = torch.max(outputs.radii, axis=0).values

        remove_mask = torch.zeros(
            (self.splats.means.shape[0]), dtype=torch.bool, device=self.conf.device
        )

        if self.conf.enable_visibility_pruning and prune and (len(window) >= 2):
            radii = outputs.radii[: self.conf.optim_window_last_n_keyframes]
            n_touched = outputs.n_touched[: self.conf.optim_window_last_n_keyframes]
            remove_mask = remove_mask | self.pruning_conditioning.step(
                self.splats, self.splat_optimizers, radii, n_touched
            )

        # size pruning
        remove_mask = remove_mask | self.pruning_size.step(
            self.splats,
            self.splat_optimizers,
            gaussians_max_screen_size,
        )

        # opacity pruning
        remove_mask = remove_mask | self.pruning_opacity.step(
            self.splats,
            self.splat_optimizers,
        )

        if prune:
            prune_using_mask(self.splats, self.splat_optimizers, ~remove_mask)

        last_kf = list(self.keyframes.values())[-1]

        outputs = self.splats(
            [last_kf.camera],
            [last_kf.pose],
            True,
        )

        # last_kf.visible_gaussians = (outputs.n_touched.sum(dim=0) > 0).detach()
        last_kf.visible_gaussians = outputs.radii.sum(dim=0) > 0
        self.last_outputs = outputs
        self.last_kf_depthmap = outputs.depthmaps[0]
        self.last_kf_rgbs = outputs.rgbs[0]

        return

    def optimize_final(self):
        n_iters = self.conf.num_iters_final

        for step in (pbar := tqdm.trange(n_iters)):
            window = random.sample(
                sorted(self.keyframes.keys()), min(10, len(self.keyframes))
            )
            window = [self.keyframes[i] for i in window]
            cameras = [x.camera for x in window]
            poses = torch.nn.ModuleList([x.pose for x in window])
            gt_imgs = create_batch(window, lambda x: x.img)

            cameras = [x.camera for x in window]
            poses = torch.nn.ModuleList([x.pose for x in window])
            gt_imgs = create_batch(window, lambda x: x.img)

            for optimizer in self.splat_optimizers.values():
                optimizer.zero_grad()

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

            desc = (
                f"[Final Optimization] loss={loss.item():.3f}, "
                f"n_splats={self.splats.means.shape[0]:07}"
            )
            pbar.set_description(desc)

            for optimizer in self.splat_optimizers.values():
                optimizer.step()

        return

    def sync(self):
        self.frontend_queue.put(
            (
                BackendMessage.SYNC,
                deepcopy(self.keyframes),
                self.last_kf_depthmap.detach(),
                self.last_kf_rgbs.detach(),
                self.splats.no_grad_clone(),
                # self.splats.mask(self.last_outputs.radii[0] > 0).no_grad_clone(),
                deepcopy(self.pose_graph),
            )
        )
        return

    def end_sync(self):
        self.frontend_queue.put(
            (
                BackendMessage.END_SYNC,
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
        # [kf.pose.normalize() for kf in self.keyframes.values()]

    def initialize_optimizers(self):
        # TODO fix these LRs
        self.splat_optimizers: Dict[str, torch.optim.Optimizer] = {}
        self.splat_optimizers['means'] = torch.optim.Adam(
            params=[self.splats.means],
            lr=self.conf.means_lr,
            fused=True,
        )
        self.splat_optimizers['quats'] = torch.optim.Adam(
            params=[self.splats.quats],
            lr=self.conf.quat_lr,
            fused=True,
        )
        self.splat_optimizers['scales'] = torch.optim.Adam(
            params=[self.splats.scales],
            lr=self.conf.scale_lr,
            fused=True,
        )
        self.splat_optimizers['opacities'] = torch.optim.Adam(
            params=[self.splats.opacities],
            lr=self.conf.opacity_lr,
            fused=True,
        )
        self.splat_optimizers['colors'] = torch.optim.Adam(
            params=[self.splats.colors],
            lr=self.conf.color_lr,
            fused=True,
        )
        self.splat_optimizers['log_uncertainties'] = torch.optim.Adam(
            params=[self.splats.log_uncertainties],
            lr=self.conf.log_uncertainty_lr,
            fused=True,
        )
        self.pose_optimizer = torch.optim.Adam(
            params=[torch.empty(0)],
            lr=0.001,
            fused=True,
        )

    def initialize(self, frame: Frame):
        frame = frame.to(self.conf.device)
        self.keyframes[frame.index] = frame
        self.splats = GaussianSplattingData.empty().to(self.conf.device)
        self.initialize_optimizers()

        self.pose_graph[frame.index] = set()

        H, W, _ = frame.img.shape
        mock_depth_map = torch.ones((1, H, W), device=self.conf.device)
        mock_depth_map = mock_depth_map + (torch.randn_like(mock_depth_map) - 0.5) * 0.3
        mock_depth_map *= self.conf.initial_scale
        mock_alphas = torch.ones((1, H, W, 1), device=self.conf.device) * 0.01

        mock_outputs = RasterizationOutput(None, mock_alphas, mock_depth_map)

        self.insertion_depth_map.step(
            self.splats,
            self.splat_optimizers,
            mock_outputs,
            frame,
            5000,
            keyframes=list(self.keyframes.values()),
        )

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
            N=100,
            keyframes=list(self.keyframes.values()),
        )

        new_frame = Frame(
            img=frame.img.clone(),
            timestamp=frame.timestamp,
            camera=frame.camera.clone(),
            pose=Pose(frame.pose().detach()).to(self.conf.device),
            gt_pose=frame.gt_pose,
            gt_depth=frame.gt_depth,
            img_file=frame.img_file,
            index=frame.index,
            est_depths=outputs.depths,
            exposure_params=frame.exposure_params.detach()
            .clone()
            .requires_grad_(False),
        )

        self.keyframes[new_frame.index] = new_frame
        self.pose_optimizer.add_param_group(
            {
                'params': new_frame.pose.parameters(),
                'lr': self.conf.pose_optim_lr,
            }
        )

        if len(self.keyframes) >= 1:
            add_constraint(self.pose_graph, *(list(self.keyframes.keys())[-2:]))

    def to_add_pg_edge(
        self,
        previous_keyframe: Frame,
        new_frame: Frame,
    ):
        intersection = torch.logical_and(
            new_frame.visible_gaussians, previous_keyframe.visible_gaussians
        )
        union = torch.logical_or(
            new_frame.visible_gaussians, previous_keyframe.visible_gaussians
        )
        iou = intersection.sum() / union.sum()
        return iou.item() > self.conf.kf_cov

    def to_remove_keyframe(
        self,
        kf_i: Frame,
        kf_j: Frame,
    ) -> tuple[bool, float]:
        intersection = torch.logical_and(kf_j.visible_gaussians, kf_i.visible_gaussians)
        oc = intersection.sum() / (
            min(
                kf_i.visible_gaussians.sum().item(), kf_j.visible_gaussians.sum().item()
            )
        )
        return oc.item() > self.conf.kf_oc, oc.item()

    @torch.no_grad()
    def add_pgo_constraints(
        self,
    ):
        for kf in self.keyframes.values():
            outputs = self.splats(
                [kf.camera],
                [kf.pose],
            )

            kf.visible_gaussians = outputs.radii.sum(dim=0) > 0

        for i, j in combinations(sorted(self.keyframes), 2):
            if i not in self.keyframes:
                continue
            if j not in self.keyframes:
                continue

            kf_i = self.keyframes[i]
            kf_j = self.keyframes[j]

            # to_remove, oc = self.to_remove_keyframe(kf_i, kf_j)
            # if to_remove:
            #     print(f'removing kf {i} because oc({i},{j}) = {oc} > conf.kf_oc={self.conf.kf_oc}')
            #     self.keyframes.pop(i, None)
            #     remove_keyframe(self.pose_graph, i)
            #     continue

            if j in self.pose_graph[i]:
                continue
            if self.to_add_pg_edge(kf_i, kf_j):
                print(f'Found loop closure! {i, j}')
                add_constraint(self.pose_graph, i, j)

        for kf in self.keyframes.values():
            kf.visible_gaussians = None

    @torch.no_grad()
    def to_insert_keyframe(
        self,
        previous_keyframe: Frame,
        new_frame: Frame,
    ):
        outputs: RasterizationOutput = self.splats(
            [new_frame.camera, previous_keyframe.camera],
            [new_frame.pose, previous_keyframe.pose],
            render_depth=True,
        )

        _photometric_error = (outputs.rgbs[0] - new_frame.img).square().mean()
        # TODO parameterize this
        # TODO this isn't kicking in
        # if photometric_error.item() > 0.15:
        #   return True

        new_frame.visible_gaussians = outputs.radii[0] > 0
        previous_keyframe.visible_gaussians = outputs.radii[1] > 0

        intersection = (
            torch.logical_and(
                new_frame.visible_gaussians, previous_keyframe.visible_gaussians
            )
            .sum()
            .detach()
            .item()
        )
        union = (
            torch.logical_or(
                new_frame.visible_gaussians, previous_keyframe.visible_gaussians
            )
            .sum()
            .detach()
            .item()
        )

        new_frame.visible_gaussians = None
        previous_keyframe.visible_gaussians = None

        _iou = intersection / union
        # if iou < self.conf.kf_cov:
        #   return True
        pose_difference = torch.linalg.inv(new_frame.pose()) @ previous_keyframe.pose()
        translation = pose_difference[:3, 3].pow(2.0).sum().pow(0.5).item()
        median_depth = outputs.depthmaps[outputs.alphas[..., 0] > 0.1].median()
        if translation > self.conf.kf_m * median_depth:
            return True

        cosine_sim = torch.nn.functional.cosine_similarity(
            new_frame.pose()[:3, 2], previous_keyframe.pose()[:3, 2], dim=0
        )
        if cosine_sim < self.conf.kf_cos:
            print(f"Added keyframe: rotation-{torch.acos(cosine_sim)*180/torch.pi}\n")
            return True
        return False

    def run(self):
        self.warp = None
        self.pause_map_optim = False

        if self.enable_viser_server:
            self.server = viser.ViserServer(verbose=False)
            self.viewer = Viewer(
                server=self.server,
                render_fn=self.viewer_render_fn,
                mode="training",
            )

        while True:
            if self.queue.empty():
                if self.pause_map_optim or len(self.keyframes) == 0:
                    time.sleep(0.03)
                    continue
                with self.splats_mutex:
                    self.optimize_map()
            match self.queue.get():
                case [FrontendMessage.ADD_REFINED_DEPTHMAP, _depth, _frame_idx]:
                    raise NotImplementedError()
                case [FrontendMessage.ADD_FRAME, frame]:
                    frame = deepcopy(frame)
                    if self.warp is None:
                        self.warp = Warp(
                            frame.camera.intrinsics,
                            frame.camera.height,
                            frame.camera.width,
                        )
                    if len(self.keyframes) == 0:
                        self.initialize(frame)
                        continue
                    last_keyframe = self.keyframes[sorted(self.keyframes.keys())[-1]]
                    if self.to_insert_keyframe(last_keyframe, frame):
                        self.pause_map_optim = False
                        with self.splats_mutex:
                            self.add_keyframe(frame)
                            self.optimize_map(1, prune=True, regularize=False)
                        if self.conf.enable_pgo:
                            with self.splats_mutex:
                                self.add_pgo_constraints()
                    if frame.index % 5 == 0:
                        with self.splats_mutex:
                            self.sync()
                case [FrontendMessage.REQUEST_INIT, frame]:
                    frame = deepcopy(frame)
                    self.pause_map_optim = False
                    self.initialize(frame)
                    with self.splats_mutex:
                        self.optimize_map(
                            self.conf.num_iters_initialization,
                            prune=True,
                            regularize=False,
                        )
                    self.sync()
                    continue
                case None:
                    print('Not running final optimization.')
                    # self.optimize_final()
                    break
                case message_from_frontend:
                    self.logger.warning(f"Unknown {message_from_frontend}")

        self.end_sync()
        print(self.pose_graph)
        self.backend_done_event.set()
