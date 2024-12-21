from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .utils import unvmap

identity_6d = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1]. Adapted from pytorch3d.
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


# https://github.com/nerfstudio-project/gsplat/blob/795161945b37747709d4da965b226a19fdf87d3f/examples/utils.py
class Pose(torch.nn.Module):
    def __init__(self, _pose: torch.Tensor = None, is_learnable: bool = True):
        super().__init__()

        self.is_learnable = is_learnable
        self.Rt = torch.nn.Buffer(_pose)
        if _pose is None:
            self.Rt = torch.nn.Buffer(torch.eye(4))

        self.pose_delta = torch.nn.Parameter(
            torch.tensor(
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                dtype=torch.float32,
                requires_grad=is_learnable,
            )
        )

    def set_random_pose_delta(self):
        torch.nn.init.normal_(self.embeds.weight)

    def forward(self) -> torch.Tensor:
        if not self.is_learnable:
            return self.Rt
        dx, drot = self.pose_delta[:3], self.pose_delta[3:]
        id = torch.tensor([1, 0, 0, 0, 1, 0], device=self.Rt.device)
        rot = unvmap(rotation_6d_to_matrix)(drot + id)
        transform = torch.eye(4, device=self.Rt.device)
        transform[..., :3, :3] = rot
        transform[..., :3, 3] = dx
        return torch.matmul(self.Rt, transform)

    def to_qt(
        self,
    ):
        pose = self()
        R = pose[:3, :3]
        t = pose[:3, 3]
        return unvmap(matrix_to_quaternion)(R), t


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    out = quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))
    return standardize_quaternion(out)


@dataclass
class Camera:
    intrinsics: torch.Tensor
    height: int
    width: int

    def to(self, device):
        self.intrinsics = self.intrinsics.to(device)
        return self


@dataclass
class Frame:
    img: torch.Tensor
    timestamp: float
    camera: Camera
    pose: Pose
    gt_pose: torch.Tensor

    def to(self, device):
        self.camera = self.camera.to(device)
        self.pose = self.pose.to(device)
        self.gt_pose = self.gt_pose.to(device)
        self.img = self.img.to(device)
        return self
