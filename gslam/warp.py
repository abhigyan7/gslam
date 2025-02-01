#!/usr/bin/env python3

import torch
import torch.nn.functional as F


class Warp(torch.nn.Module):
    def __init__(self, K: torch.Tensor, H: int, W: int) -> None:
        super().__init__()

        self.H = H
        self.W = W
        K_inv = torch.linalg.inv(K)
        uu, vv = torch.meshgrid(
            torch.arange(W, device=K.device),
            torch.arange(H, device=K.device),
            indexing='ij',
        )
        # [H, W, 3]
        grid = torch.stack(
            [
                uu,
                vv,
                torch.ones((W, H), device=K.device),
            ],
            dim=-1,
        ).to(K.device)

        # backproject in a very ugly way
        unprojected = torch.matmul(K_inv, grid.unsqueeze(-1))

        self.register_buffer("K", K)
        self.register_buffer("unprojected", unprojected)  # N x 2, uv coordinate

    def forward(
        self,
        f1_pose: torch.Tensor,
        f2_pose: torch.Tensor,
        _K: torch.Tensor,
        c1: torch.Tensor,
        d1: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        K = self.K
        unprojected = self.unprojected
        T = f1_pose @ torch.linalg.inv(f2_pose)

        backprojected_points = d1.t().unsqueeze(-1).unsqueeze(-1) * unprojected
        backprojected_points_in_new_frame = T[:3, :3] @ backprojected_points

        backprojected_points_in_new_frame = (
            backprojected_points_in_new_frame.squeeze(-1) + T[:3, 3]
        )
        warped_points = torch.matmul(backprojected_points_in_new_frame, K.t())

        warps = warped_points[..., :2] / warped_points[..., -1].unsqueeze(-1)
        warps[..., 0] *= 2.0 / self.W
        warps[..., 1] *= 2.0 / self.H
        warps -= 1.0

        normalized_warps = warps.unsqueeze(0)

        normalized_warps = normalized_warps.permute((0, 2, 1, 3))
        result = F.grid_sample(
            c1.permute((2, 0, 1)).unsqueeze(0),
            normalized_warps,
            padding_mode='zeros',
            align_corners=False,
        )
        result = result.squeeze(0).permute((1, 2, 0))

        # filtering out points that are outside of the image bounds
        normalized_warps_v = normalized_warps[0, ..., 0]
        normalized_warps_h = normalized_warps[0, ..., 1]

        keep_mask = (
            (normalized_warps_v < 1.0)
            & (normalized_warps_h < 1.0)
            & (normalized_warps_v > -1.0)
            & (normalized_warps_h > -1.0)
        )

        return result, normalized_warps, keep_mask


def differentiable_warp(image, depth, warp, K):
    B, _, H, W = image.shape

    # Create pixel coordinates
    y, x = torch.meshgrid(
        torch.arange(H, device=image.device), torch.arange(W, device=image.device)
    )
    pixels = torch.stack((x, y, torch.ones_like(x)), dim=-1).float()  # (H, W, 3)
    pixels = pixels.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, W, 3)

    # Unproject to 3D
    depth = depth.permute(0, 2, 3, 1)  # (B, H, W, 1)
    cam_points = (torch.inverse(K).unsqueeze(1) @ pixels.unsqueeze(-1)).squeeze(
        -1
    )  # (B, H, W, 3)
    world_points = depth * cam_points  # (B, H, W, 3)

    # Transform points
    world_points = world_points.view(B, -1, 3).transpose(1, 2)  # (B, 3, H*W)
    world_points = torch.cat(
        [world_points, torch.ones_like(world_points[:, :1])], dim=1
    )  # (B, 4, H*W)
    cam_points = warp @ world_points  # (B, 4, H*W)
    cam_points = cam_points[:, :3] / cam_points[:, 3:4]  # (B, 3, H*W)

    # Project to 2D
    proj_points = K @ cam_points  # (B, 3, H*W)
    proj_points = proj_points[:, :2] / proj_points[:, 2:3]  # (B, 2, H*W)
    proj_points = proj_points.view(B, 2, H, W)  # (B, 2, H, W)

    # Normalize coordinates
    proj_points[:, 0] = (proj_points[:, 0] / (W - 1)) * 2 - 1
    proj_points[:, 1] = (proj_points[:, 1] / (H - 1)) * 2 - 1
    proj_points = proj_points.permute(0, 2, 3, 1)  # (B, H, W, 2)

    # Sample from image
    warped_image = F.grid_sample(
        image, proj_points, mode='bilinear', padding_mode='zeros', align_corners=True
    )

    return warped_image


def diffwarp_wrap(T1, T2, K, image, depth):
    T = T1 @ torch.linalg.inv(T2)
    image = image.permute((2, 0, 1)).unsqueeze(0)
    depth = depth.unsqueeze(0).unsqueeze(0)
    warp = T.unsqueeze(0)
    K = K.unsqueeze(0)
    return differentiable_warp(image, depth, warp, K), None, None


class DiffWarp(torch.nn.Module):
    def __init__(self, K: torch.Tensor, H: int, W: int) -> None:
        super().__init__()
        y, x = torch.meshgrid(
            torch.arange(H, device='cuda'), torch.arange(W, device='cuda')
        )
        pixels = torch.stack((x, y, torch.ones_like(x)), dim=-1).float()  # (H, W, 3)
        pixels = pixels.unsqueeze(0).expand(1, -1, -1, -1)  # (B, H, W, 3)
        self.register_buffer("pixels", pixels)  # N x 2, uv coordinate
        self.H = H
        self.W = W

    def forward(
        self,
        f1_pose: torch.Tensor,
        f2_pose: torch.Tensor,
        _K: torch.Tensor,
        c1: torch.Tensor,
        d1: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = 1
        H = self.H
        W = self.W
        T = f1_pose @ torch.linalg.inv(f2_pose)
        image = c1.permute((2, 0, 1)).unsqueeze(0)
        depth = d1.unsqueeze(0).unsqueeze(0)
        warp = T.unsqueeze(0)
        K = _K.unsqueeze(0)

        # Unproject to 3D
        depth = depth.permute(0, 2, 3, 1)  # (B, H, W, 1)
        cam_points = (
            torch.inverse(K).unsqueeze(1) @ self.pixels.unsqueeze(-1)
        ).squeeze(-1)  # (B, H, W, 3)
        world_points = depth * cam_points  # (B, H, W, 3)

        # Transform points
        world_points = world_points.view(B, -1, 3).transpose(1, 2)  # (B, 3, H*W)
        world_points = torch.cat(
            [world_points, torch.ones_like(world_points[:, :1])], dim=1
        )  # (B, 4, H*W)
        cam_points = warp @ world_points  # (B, 4, H*W)
        cam_points = cam_points[:, :3] / cam_points[:, 3:4]  # (B, 3, H*W)

        # Project to 2D
        proj_points = K @ cam_points  # (B, 3, H*W)
        proj_points = proj_points[:, :2] / proj_points[:, 2:3]  # (B, 2, H*W)
        proj_points = proj_points.view(B, 2, H, W)  # (B, 2, H, W)

        # Normalize coordinates
        proj_points[:, 0] = (proj_points[:, 0] / (W - 1)) * 2 - 1
        proj_points[:, 1] = (proj_points[:, 1] / (H - 1)) * 2 - 1
        proj_points = proj_points.permute(0, 2, 3, 1)  # (B, H, W, 2)

        # Sample from image
        warped_image = F.grid_sample(
            image,
            proj_points,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True,
        )

        return warped_image, None, None
