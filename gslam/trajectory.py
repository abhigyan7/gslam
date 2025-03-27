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
    def __init__(self, interval: float, starting_time: float, end_time: float = None):
        super().__init__()
        self.cps_SO3 = torch.nn.ParameterList()
        self.cps_R3 = torch.nn.ParameterList()
        self.interval: float = interval
        self.starting_time: float = starting_time
        self.end_time: float = self.starting_time if end_time is None else end_time
        self.gravity_vector: torch.nn.Parameter = torch.nn.Parameter(
            torch.randn(
                [
                    3,
                ],
                requires_grad=True,
            )
        )
        self.gravity_alignment = pp.identity_Sim3(requires_grad=True)

    def _parse_time(self, time: torch.Tensor):
        segment = math.floor((time - self.starting_time) / self.interval)
        segment = max(segment, 1)
        segment = min(segment, len(self) - 3)
        segment_start = segment * self.interval + self.starting_time
        t = (time - segment_start) / self.interval
        return segment, t

    def __len__(
        self,
    ):
        return len(self.cps_SO3)

    def forward(self, time: torch.Tensor):
        segment, t = self._parse_time(time)
        coeff_1 = (5.0 + 3 * t - 3 * t * t + t * t * t) / 6.0
        coeff_2 = (1.0 + 3 * t + 3 * t * t - 2 * t * t * t) / 6.0
        coeff_3 = (t * t * t) / 6.0

        cps_SO3 = pp.SO3(
            torch.stack([i.data for i in self.cps_SO3[segment - 1 : segment + 3]])
        )
        diffs_so3 = (cps_SO3[:-1].Inv() @ cps_SO3[1:]).Log()
        ret_SO3 = cps_SO3[0]
        ret_SO3 = ret_SO3 * (diffs_so3[0] * coeff_1).Exp()
        ret_SO3 = ret_SO3 * (diffs_so3[1] * coeff_2).Exp()
        ret_SO3 = ret_SO3 * (diffs_so3[2] * coeff_3).Exp()

        cps_R3 = self.cps_R3[segment - 1 : segment + 3]
        diffs_R3 = [j - i for (i, j) in zip(cps_R3[:-1], cps_R3[1:])]
        ret_R3 = cps_R3[0] + (
            coeff_1 * diffs_R3[0] + coeff_2 * diffs_R3[1] + coeff_3 * diffs_R3[2]
        )

        return ret_SO3, ret_R3

    def angular_velocity(self, time: torch.Tensor):
        segment, t = self._parse_time(time)
        dot_coeff_1 = (3.0 - 6 * t + 3 * t * t) / 6.0
        dot_coeff_2 = (3 + 6 * t - 6 * t * t) / 6.0
        dot_coeff_3 = (3 * t * t) / 6.0

        coeff_1 = (5.0 + 3 * t - 3 * t * t + t * t * t) / 6.0
        coeff_2 = (1.0 + 3 * t + 3 * t * t - 2 * t * t * t) / 6.0
        coeff_3 = (t * t * t) / 6.0

        cps_SO3 = pp.SO3(
            torch.stack([i.data for i in self.cps_SO3[segment - 1 : segment + 3]])
        )
        diffs_so3 = (cps_SO3[:-1].Inv() @ cps_SO3[1:]).Log()
        ret_se3 = (dot_coeff_1 * diffs_so3[0]).Exp() * (diffs_so3[0] * coeff_1)
        ret_se3 = (dot_coeff_2 * diffs_so3[1]).Exp() * ret_se3 + (
            diffs_so3[1] * coeff_2
        )
        ret_se3 = (dot_coeff_3 * diffs_so3[2]).Exp() * ret_se3 + (
            diffs_so3[2] * coeff_3
        )

        return ret_se3

    def velocity(self, time: torch.Tensor, gravity: bool = False):
        segment, t = self._parse_time(time)
        coeff_1 = (3.0 - 6 * t + 3 * t * t) / 6.0
        coeff_2 = (3 + 6 * t - 6 * t * t) / 6.0
        coeff_3 = (3 * t * t) / 6.0

        cps_R3 = self.cps_R3[segment - 1 : segment + 3]
        diffs_R3 = [j - i for (i, j) in zip(cps_R3[:-1], cps_R3[1:])]
        ret_R3 = coeff_1 * diffs_R3[0] + coeff_2 * diffs_R3[1] + coeff_3 * diffs_R3[2]
        return ret_R3

    def acceleration(self, time: torch.Tensor, gravity: bool = False):
        segment, t = self._parse_time(time)
        coeff_1 = -1 + t
        coeff_2 = 1 - 2 * t
        coeff_3 = t

        cps_R3 = self.cps_R3[segment - 1 : segment + 3]
        diffs_R3 = [j - i for (i, j) in zip(cps_R3[:-1], cps_R3[1:])]
        ret_R3 = coeff_1 * diffs_R3[0] + coeff_2 * diffs_R3[1] + coeff_3 * diffs_R3[2]
        if gravity:
            ret_R3 += self.gravity_alignment * self.gravity_vector
        return ret_R3
