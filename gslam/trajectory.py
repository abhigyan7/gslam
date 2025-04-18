import math
import numpy as np
from typing import Tuple

from matplotlib import pyplot as plt

from .primitives import Frame
import torch
import pypose as pp

plt.switch_backend('agg')


def kabsch_umeyama(
    A: np.ndarray, B: np.ndarray
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    implementation from https://zpl.fi/aligning-point-patterns-with-kabsch-umeyama-algorithm/
    B' = t + c * R @ b gives b in A's frame
    """
    assert A.shape == B.shape
    n, m = A.shape

    EA = np.mean(A, axis=0)
    EB = np.mean(B, axis=0)
    VarA = np.mean(np.linalg.norm(A - EA, axis=1) ** 2)

    try:
        H = ((A - EA).T @ (B - EB)) / n
        U, D, VT = np.linalg.svd(H)
        d = np.sign(np.linalg.det(U) * np.linalg.det(VT))
        S = np.diag([1] * (m - 1) + [d])

        R = U @ S @ VT
        c = VarA / np.trace(np.diag(D) @ S)
        t = EA - c * R @ EB
    except np.linalg.LinAlgError as e:
        print(f'{e=}')
        R = np.eye(3)
        c = 1.0
        t = np.array([0, 0, 0], dtype=np.float32)

    return R, c, t


def average_translation_error(A: np.ndarray, B: np.ndarray) -> float:
    R, c, t = kabsch_umeyama(A, B)
    aligned = np.array([t + c * R @ b for b in B])
    error = aligned - A
    ate = np.mean(np.sqrt(np.sum(np.multiply(error, error), -1)))
    return ate


def plot_trajectory(trajectories: list, labels: list[str], ax, keyframe_indices=None):
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    for trajectory, label in zip(trajectories, labels):
        ax.plot(trajectory[..., 1], trajectory[..., 2], label=label)
        ax.scatter(trajectory[[0], 1], trajectory[[0], 2], marker='x')
        if keyframe_indices is not None and max(keyframe_indices) <= len(trajectory):
            ax.scatter(
                trajectory[keyframe_indices, 1],
                trajectory[keyframe_indices, 2],
                marker='o',
            )
    ax.set_aspect('equal')
    ax.legend()


def evaluate_trajectories(
    trajectories: dict[str, list[Frame]],
    keyframe_indices: list[int] = None,
) -> tuple[plt.Figure, dict]:
    fig, axes = plt.subplots(1, len(trajectories))
    fig.set_figwidth(5 * len(trajectories))
    fig.set_figheight(5)

    ates = dict()

    for traj_name, ax in zip(trajectories, axes):
        gt_Rts = np.array([f.gt_pose.cpu().numpy() for f in trajectories[traj_name]])
        if len(gt_Rts) < 2:
            continue
        estimated_Rts = np.array(
            [f.pose().cpu().numpy() for f in trajectories[traj_name]]
        )
        R, c, t = kabsch_umeyama(gt_Rts[:, :3, 3], estimated_Rts[:, :3, 3])
        estimated_ts = np.array([t + c * R @ b[:3, 3] for b in estimated_Rts])
        gt_ts = gt_Rts[..., :3, 3]

        plot_trajectory([gt_ts, estimated_ts], ['gt', traj_name], ax, keyframe_indices)
        ax.set_aspect('equal')
        ax.set_box_aspect(1)

        ates['ate_' + traj_name] = average_translation_error(gt_ts, estimated_ts)

    return fig, ates


class Trajectory(torch.nn.Module):
    def __init__(
        self,
        interval: float,
        starting_time: float,
        num_cps: int = 4000,
    ):
        super().__init__()
        self.cps_SO3 = pp.Parameter(
            pp.identity_SO3(num_cps, requires_grad=True, dtype=torch.double)
        )
        self.cps_R3 = torch.nn.Parameter(
            torch.zeros([num_cps, 3], requires_grad=True, dtype=torch.double)
        )
        self.interval: float = interval
        self.starting_time: float = starting_time
        # self.gravity_vector: torch.nn.Parameter = torch.nn.Parameter(
        #     torch.randn(
        #         [
        #             3,
        #         ],
        #         requires_grad=True,
        #     )
        # )
        # self.gravity_alignment = pp.Parameter(pp.identity_Sim3(requires_grad=True))
        self.cursor = 0
        # self.extend_to_time(starting_time+4*interval)

        # [-1, 0, 1, 2], pre-alloc'd and stored in device
        self.indices_4 = torch.nn.Buffer(
            torch.arange(-1, 3, device=self.cps_SO3.device, dtype=torch.long)
        )

    @torch.no_grad()
    def add_control_point(self, new_SO3: pp.SO3, new_R3: torch.Tensor):
        assert self.cursor < self.cps_SO3.shape[0]
        self.cps_SO3[self.cursor] = new_SO3
        self.cps_R3[self.cursor] = new_R3
        self.cursor += 1

    def support_end(self):
        return self.starting_time + self.interval * self.cursor

    @torch.no_grad()
    def extend_to_time(self, time: float):
        n_added = 0
        while self.support_end() < time:
            assert self.cursor < self.cps_SO3.shape[0]
            so3 = self.cps_SO3[self.cursor - 2].Inv() * self.cps_SO3[self.cursor - 1]
            self.cps_SO3[self.cursor] = self.cps_SO3[self.cursor - 1] * so3 * so3
            self.cps_R3[self.cursor] = self.cps_R3[self.cursor - 1] + 2 * (
                self.cps_R3[self.cursor - 1] - self.cps_R3[self.cursor - 2]
            )
            self.cursor += 1
            n_added += 1
        if n_added > 0:
            print(f'Added {n_added} CPs upto time {time}, {self.support_end()=}')
            return True
        return False

    def _parse_time(self, time: torch.Tensor):
        segment = math.floor((time - self.starting_time) / self.interval)
        if segment < 1:
            print(f'Too early: {time=}, {self.starting_time=}, {self.interval=}')
            segment = 1
        if segment > self.cursor - 2:
            print(f'Too far: {time=}, {self.support_end()=}, {self.interval=}')
            segment = self.cursor - 2
        segment_start = segment * self.interval + self.starting_time
        t = (time - segment_start) / self.interval
        return segment, t

    def _parse_time_torch(self, time: torch.Tensor):
        segment = time.sub(self.starting_time).div(self.interval).floor()
        segment = segment.clamp(1, self.cursor - 2)
        segment_start = (segment * self.interval + self.starting_time).detach()
        t = (time - segment_start) / self.interval
        return segment.long(), t

    def __len__(
        self,
    ):
        return self.cursor

    def forward(self, time: torch.Tensor):
        if not isinstance(time, torch.Tensor):
            time = torch.tensor(time, device=self.cps_R3.device, dtype=torch.double)
        segment, t = self._parse_time_torch(time)
        t2 = t.square()
        t3 = t * t2
        coeff_1 = (5.0 + 3 * t - 3 * t2 + t3) / 6.0
        coeff_2 = (1.0 + 3 * t + 3 * t2 - 2 * t3) / 6.0
        coeff_3 = t3 / 6.0

        cps_SO3 = self.cps_SO3[segment.reshape(-1, 1) + self.indices_4]
        diffs_so3 = (cps_SO3[..., :-1, :].Inv() @ cps_SO3[..., 1:, :]).Log()
        ret_SO3 = cps_SO3[..., 0, :]
        ret_SO3 = ret_SO3 * (diffs_so3[..., 0, :] * coeff_1.view(-1, 1)).Exp()
        ret_SO3 = ret_SO3 * (diffs_so3[..., 1, :] * coeff_2.view(-1, 1)).Exp()
        ret_SO3 = ret_SO3 * (diffs_so3[..., 2, :] * coeff_3.view(-1, 1)).Exp()

        cps_R3 = self.cps_R3[segment.reshape(-1, 1) + self.indices_4]
        diffs_R3 = cps_R3[..., 1:, :] - cps_R3[..., :-1, :]
        ret_R3 = cps_R3[..., 0, :] + (
            coeff_1.view(-1, 1) * diffs_R3[..., 0, :]
            + coeff_2.view(-1, 1) * diffs_R3[..., 1, :]
            + coeff_3.view(-1, 1) * diffs_R3[..., 2, :]
        )

        return ret_SO3, ret_R3

    def velocity(self, time: torch.Tensor, gravity: bool = False):
        if not isinstance(time, torch.Tensor):
            time = torch.tensor(time, device=self.cps_R3.device, dtype=torch.double)
        segment, t = self._parse_time_torch(time)
        t2 = t.square()
        coeff_1 = (3.0 - 6 * t + 3 * t2) / 6.0
        coeff_2 = (3 + 6 * t - 6 * t2) / 6.0
        coeff_3 = (3 * t2) / 6.0

        cps_R3 = self.cps_R3[segment.reshape(-1, 1) + self.indices_4]
        diffs_R3 = cps_R3[..., 1:, :] - cps_R3[..., :-1, :]
        ret_R3 = (
            coeff_1.view(-1, 1) * diffs_R3[..., 0, :]
            + coeff_2.view(-1, 1) * diffs_R3[..., 1, :]
            + coeff_3.view(-1, 1) * diffs_R3[..., 2, :]
        )
        return ret_R3

    def acceleration(self, time: torch.Tensor, gravity: bool = False):
        if not isinstance(time, torch.Tensor):
            time = torch.tensor(time, device=self.cps_R3.device, dtype=torch.double)
        segment, t = self._parse_time_torch(time)
        coeff_1 = -1 + t
        coeff_2 = 1 - 2 * t
        coeff_3 = t

        cps_R3 = self.cps_R3[segment.reshape(-1, 1) + self.indices_4]
        diffs_R3 = cps_R3[..., 1:, :] - cps_R3[..., :-1, :]
        ret_R3 = (
            coeff_1.view(-1, 1) * diffs_R3[..., 0, :]
            + coeff_2.view(-1, 1) * diffs_R3[..., 1, :]
            + coeff_3.view(-1, 1) * diffs_R3[..., 2, :]
        )

        SO3, _R3 = self.forward(time)
        ret_R3 = SO3 * ret_R3 * (1.0 / self.interval) ** 2 * 2.0
        if gravity:
            ret_R3 = ret_R3 + self.gravity_alignment * self.gravity_vector
        return ret_R3
