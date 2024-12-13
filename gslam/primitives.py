import torch
from dataclasses import dataclass
from typing import Optional


# https://github.com/nerfstudio-project/gsplat/blob/795161945b37747709d4da965b226a19fdf87d3f/examples/utils.py
class Pose(torch.nn.Module):

    def __init__(self, n: int = 1, is_learnable: bool = True):
        super().__init__()

        self.is_learnable = is_learnable
        self.embeds = torch.nn.Embedding(n, 9)
        self.register_buffer(
            'identity', torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
            )
        self.zero_init()

    def zero_init(self):
        torch.nn.init.zeros_(self.embeds.weight)

    def random_init(self):
        torch.nn.init.normal_(self.embeds.weight)

    def forward(self, camtoworlds: torch.Tensor = None, embed_ids: torch.Tensor = None) -> torch.Tensor:
        if not self.is_learnable:
            return camtoworlds
        if camtoworlds is None:
            camtoworlds = torch.eye(4).unsqueeze(0).cuda()
        if embed_ids is None:
            embed_ids = torch.tensor([0]).cuda()
        batch_shape = camtoworlds.shape[:-2]
        pose_deltas = self.embeds(embed_ids)
        dx, drot = pose_deltas[..., :3], pose_deltas[..., 3:]
        rot = rotation_6d_to_matrix(
            drot + self.identity.expand(*batch_shape, -1)
        )
        transform = torch.eye(4, device=pose_deltas.device).repeat((*batch_shape, 1, 1))
        transform[..., :3, :3] = rot
        transform[..., :3, 3] = dx
        return torch.matmul(camtoworlds, transform)


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
