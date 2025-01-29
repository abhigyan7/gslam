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
