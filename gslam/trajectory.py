import numpy as np
from typing import Tuple

from matplotlib import pyplot as plt

from .primitives import Frame


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
    ate = np.sqrt(np.mean(np.sum(np.multiply(error, error), -1)))
    return ate


def plot_trajectory(trajectories: list, labels: list[str], ax):
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    for trajectory, label in zip(trajectories, labels):
        ax.plot(trajectory[..., 1], trajectory[..., 2], label=label)
        ax.scatter(trajectory[[0], 1], trajectory[[0], 2], marker='x')
    ax.set_aspect('equal')
    ax.legend()


def evaluate_trajectories(
    trajectories: dict[str, list[Frame]],
) -> tuple[plt.Figure, dict]:
    fig, axes = plt.subplots(1, len(trajectories))
    fig.set_figwidth(5 * len(trajectories))
    fig.set_figheight(5)

    ates = dict()

    for traj_name, ax in zip(trajectories, axes):
        gt_Rts = np.array([f.gt_pose.cpu().numpy() for f in trajectories[traj_name]])
        estimated_Rts = np.array(
            [f.pose().cpu().numpy() for f in trajectories[traj_name]]
        )
        R, c, t = kabsch_umeyama(gt_Rts[:, :3, 3], estimated_Rts[:, :3, 3])
        estimated_ts = np.array([t + c * R @ b[:3, 3] for b in estimated_Rts])
        gt_ts = gt_Rts[..., :3, 3]

        plot_trajectory([gt_ts, estimated_ts], ['gt', traj_name], ax)
        ax.set_aspect('equal')
        ax.set_box_aspect(1)

        ates['ate_' + traj_name] = average_translation_error(gt_ts, estimated_ts)

    return fig, ates
