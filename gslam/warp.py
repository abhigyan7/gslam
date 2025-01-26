#!/usr/bin/env python3

import torch
import torch.nn.functional as F
from .primitives import Frame


def warp_jit(
    f1_pose: torch.Tensor,
    f2_pose: torch.Tensor,
    K: torch.Tensor,
    c1: torch.Tensor,  # H, W, 3
    d1: torch.Tensor,  # H, W
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    K_inv = torch.linalg.inv(K)
    T = f1_pose @ torch.linalg.inv(f2_pose)

    H, W = d1.shape

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
    backprojected_points = d1.t().unsqueeze(-1).unsqueeze(-1) * unprojected
    backprojected_points_in_new_frame = T[:3, :3] @ backprojected_points
    backprojected_points_in_new_frame = (
        backprojected_points_in_new_frame.squeeze(-1) + T[:3, 3]
    )
    warped_points = torch.matmul(
        K, backprojected_points_in_new_frame.unsqueeze(-1)
    ).squeeze(-1)
    warps = warped_points[..., :2] / warped_points[..., -1].unsqueeze(-1)

    normalized_warps = warps / torch.tensor([W, H], device=K.device).float()
    normalized_warps *= 2.0
    normalized_warps -= 1.0

    normalized_warps = normalized_warps.unsqueeze(0)

    normalized_warps = normalized_warps.permute((0, 2, 1, 3))
    # DONE sample old image as per warp
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

    # H, W, 3
    return result, normalized_warps, keep_mask


def warp(
    f1: Frame,
    f2: Frame,
    c1: torch.Tensor,  # H, W, 3
    d1: torch.Tensor,  # H, W
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    K = f1.camera.intrinsics
    K_inv = torch.linalg.inv(K)
    T = f1.pose() @ torch.linalg.inv(f2.pose())

    H, W = d1.shape

    # [H, W, 3]
    grid = torch.stack(
        [
            *torch.meshgrid(
                torch.arange(W, device=K.device),
                torch.arange(H, device=K.device),
                indexing='ij',
            ),
            torch.ones((W, H), device=K.device),
        ],
        dim=-1,
    ).to(d1.device)

    # backproject in a very ugly way
    unprojected = torch.matmul(K_inv, grid.unsqueeze(-1))
    backprojected_points = d1.t().unsqueeze(-1).unsqueeze(-1) * unprojected
    backprojected_points_in_new_frame = T[:3, :3] @ backprojected_points
    backprojected_points_in_new_frame = (
        backprojected_points_in_new_frame.squeeze(-1) + T[:3, 3]
    )
    warped_points = torch.matmul(
        K, backprojected_points_in_new_frame.unsqueeze(-1)
    ).squeeze(-1)
    warps = warped_points[..., :2] / warped_points[..., [-1]]

    normalized_warps = warps / torch.tensor([W, H], device=warps.device).float()
    normalized_warps *= 2.0
    normalized_warps -= 1.0

    normalized_warps = normalized_warps.unsqueeze(0)

    normalized_warps = normalized_warps.permute((0, 2, 1, 3))
    # DONE sample old image as per warp
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

    # H, W, 3
    return result, normalized_warps, keep_mask


def get_jit_warp(device):
    sample_inputs = (
        torch.randn((4, 4)).to(device),
        torch.randn((4, 4)).to(device),
        torch.randn((3, 3)).to(device),
        torch.randn((480, 640, 3)).to(device),
        torch.randn((480, 640)).to(device),
    )
    print(f'Jitting warp for {device}')
    # if device == 'cuda':
    #     return torch.compile(warp_jit, backend='cudagraphs')
    # else:
    #     return torch.compile(warp_jit, backend='onnxrt')
    return torch.jit.trace(warp_jit, sample_inputs)
