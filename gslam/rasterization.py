import math
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor
from typing_extensions import Literal

from gsplat.cuda._wrapper import (
    fully_fused_projection,
    isect_offset_encode,
    isect_tiles,
    rasterize_to_pixels,
)


def rasterization(
    means: Tensor,  # [N, 3]
    quats: Tensor,  # [N, 4]
    log_scales: Tensor,  # [N, 3]
    logit_opacities: Tensor,  # [N]
    logit_colors: Tensor,  # [(C,) N, D]
    viewmats: Tensor,  # [C, 4, 4]
    Ks: Tensor,  # [C, 3, 3]
    width: int,
    height: int,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    eps2d: float = 0.3,
    packed: bool = True,
    tile_size: int = 16,
    backgrounds: Optional[Tensor] = None,
    render_mode: Literal["RGB", "D", "ED", "RGB+D", "RGB+ED"] = "RGB",
    sparse_grad: bool = False,
    absgrad: bool = False,
    rasterize_mode: Literal["classic", "antialiased"] = "classic",
    channel_chunk: int = 32,
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole",
    covars: Optional[Tensor] = None,
    log_betas: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Dict]:
    """Rasterize a set of 3D Gaussians (N) to a batch of image planes (C).

    Args:
        means: The 3D centers of the Gaussians. [N, 3]
        quats: The quaternions of the Gaussians (wxyz convension). It's not required to be normalized. [N, 4]
        log_scales: The scales of the Gaussians. [N, 3]
        logit_opacities: The opacities of the Gaussians. [N]
        logit_colors: The colors of the Gaussians. [N, D].
        viewmats: The world-to-cam transformation of the cameras. [C, 4, 4]
        Ks: The camera intrinsics. [C, 3, 3]
        width: The width of the image.
        height: The height of the image.
        near_plane: The near plane for clipping. Default is 0.01.
        far_plane: The far plane for clipping. Default is 1e10.
        radius_clip: Gaussians with 2D radius smaller or equal than this value will be
            skipped. This is extremely helpful for speeding up large scale scenes.
            Default is 0.0.
        eps2d: An epsilon added to the egienvalues of projected 2D covariance matrices.
            This will prevents the projected GS to be too small. For example eps2d=0.3
            leads to minimal 3 pixel unit. Default is 0.3.
        packed: Whether to use packed mode which is more memory efficient but might or
            might not be as fast. Default is True.
        tile_size: The size of the tiles for rasterization. Default is 16.
            (Note: other values are not tested)
        backgrounds: The background colors. [C, D]. Default is None.
        render_mode: The rendering mode. Supported modes are "RGB", "D", "ED", "RGB+D",
            and "RGB+ED". "RGB" renders the colored image, "D" renders the accumulated depth, and
            "ED" renders the expected depth. Default is "RGB".
        sparse_grad: If true, the gradients for {means, quats, scales} will be stored in
            a COO sparse layout. This can be helpful for saving memory. Default is False.
        absgrad: If true, the absolute gradients of the projected 2D means
            will be computed during the backward pass, which could be accessed by
            `meta["means2d"].absgrad`. Default is False.
        rasterize_mode: The rasterization mode. Supported modes are "classic" and
            "antialiased". Default is "classic".
        channel_chunk: The number of channels to render in one go. Default is 32.
            If the required rendering channels are larger than this value, the rendering
            will be done looply in chunks.
        camera_model: The camera model to use. Supported models are "pinhole", "ortho",
            and "fisheye". Default is "pinhole".
        covars: Optional covariance matrices of the Gaussians. If provided, the `quats` and
            `scales` will be ignored. [N, 3, 3], Default is None.
        log_betas: Optional confidences (betas) for the Gaussians. [N,].

    Returns:
        A tuple:

        **render_colors**: The rendered colors. [C, height, width, D].
        **render_alphas**: The rendered alphas. [C, height, width, 1].
        **meta**: A dictionary of intermediate results of the rasterization.

    """
    meta = {}

    N = means.shape[0]
    C = viewmats.shape[0]
    assert means.shape == (N, 3), means.shape
    if covars is None:
        assert quats.shape == (N, 4), quats.shape
        assert log_scales.shape == (N, 3), log_scales.shape
    else:
        assert covars.shape == (N, 3, 3), covars.shape
        quats, scales = None, None
        # convert covars from 3x3 matrix to upper-triangular 6D vector
        tri_indices = ([0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2])
        covars = covars[..., tri_indices[0], tri_indices[1]]
    assert logit_opacities.shape == (N,), logit_opacities.shape
    assert viewmats.shape == (C, 4, 4), viewmats.shape
    assert Ks.shape == (C, 3, 3), Ks.shape
    assert render_mode in ["RGB", "D", "ED", "RGB+D", "RGB+ED"], render_mode

    # treat colors as post-activation values, should be in shape [N, D]
    assert (logit_colors.dim() == 2 and logit_colors.shape[0] == N) or (
        logit_colors.dim() == 3 and logit_colors.shape[:2] == (C, N)
    ), logit_colors.shape

    opacities = torch.sigmoid(logit_opacities)
    colors = torch.sigmoid(logit_colors)
    scales = torch.exp(log_scales)
    if log_betas is not None:
        betas = torch.exp(log_betas)

    # Project Gaussians to 2D. Directly pass in {quats, scales} is faster than precomputing covars.
    proj_results = fully_fused_projection(
        means,
        covars,
        quats,
        scales,
        viewmats,
        Ks,
        width,
        height,
        eps2d=eps2d,
        packed=packed,
        near_plane=near_plane,
        far_plane=far_plane,
        radius_clip=radius_clip,
        sparse_grad=sparse_grad,
        calc_compensations=(rasterize_mode == "antialiased"),
        camera_model=camera_model,
    )

    if packed:
        # The results are packed into shape [nnz, ...]. All elements are valid.
        (
            camera_ids,
            gaussian_ids,
            radii,
            means2d,
            depths,
            conics,
            compensations,
        ) = proj_results
        opacities = opacities[gaussian_ids]  # [nnz]
    else:
        # The results are with shape [C, N, ...]. Only the elements with radii > 0 are valid.
        radii, means2d, depths, conics, compensations = proj_results
        opacities = opacities.repeat(C, 1)  # [C, N]
        camera_ids, gaussian_ids = None, None

    if compensations is not None:
        opacities = opacities * compensations

    meta.update(
        {
            # global camera_ids
            "camera_ids": camera_ids,
            # local gaussian_ids
            "gaussian_ids": gaussian_ids,
            "radii": radii,
            "means2d": means2d,
            "depths": depths,
            "conics": conics,
            "opacities": opacities,
        }
    )

    # Turn colors into [C, N, D] or [nnz, D] to pass into rasterize_to_pixels()
    # Colors are post-activation values, with shape [N, D] or [C, N, D]
    if packed:
        if colors.dim() == 2:
            # Turn [N, D] into [nnz, D]
            colors = colors[gaussian_ids]
            # Turn [N] into [nnz]
            if log_betas is not None:
                betas = betas[gaussian_ids]
        else:
            # Turn [C, N, D] into [nnz, D]
            colors = colors[camera_ids, gaussian_ids]
            # Turn [C, N] into [nnz]
            if log_betas is not None:
                betas = betas[camera_ids, gaussian_ids]
    else:
        if colors.dim() == 2:
            # Turn [N, D] into [C, N, D]
            colors = colors.expand(C, -1, -1)
            # Turn [N] into [C, N]
            if log_betas is not None:
                betas = betas.expand(C, -1)
        else:
            # colors is already [C, N, D]
            pass

    # Rasterize to pixels
    if render_mode in ["RGB+D", "RGB+ED"]:
        colors = torch.cat((colors, depths[..., None]), dim=-1)
        if backgrounds is not None:
            backgrounds = torch.cat(
                [backgrounds, torch.zeros(C, 1, device=backgrounds.device)], dim=-1
            )
        depth_index = colors.shape[-1] - 1
    elif render_mode in ["D", "ED"]:
        colors = depths[..., None]
        if backgrounds is not None:
            backgrounds = torch.zeros(C, 1, device=backgrounds.device)
        depth_index = colors.shape[-1] - 1
    else:  # RGB
        pass

    if log_betas is not None:
        colors = torch.cat((colors, betas[..., None]), dim=-1)
        if backgrounds is not None:
            backgrounds = torch.cat(
                [backgrounds, torch.zeros(C, 1, device=backgrounds.device)], dim=-1
            )
        betas_index = colors.shape[-1] - 1

    # Identify intersecting tiles
    tile_width = math.ceil(width / float(tile_size))
    tile_height = math.ceil(height / float(tile_size))
    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        means2d,
        radii,
        depths,
        tile_size,
        tile_width,
        tile_height,
        packed=packed,
        n_cameras=C,
        camera_ids=camera_ids,
        gaussian_ids=gaussian_ids,
    )
    # print("rank", world_rank, "Before isect_offset_encode")
    isect_offsets = isect_offset_encode(isect_ids, C, tile_width, tile_height)

    meta.update(
        {
            "tile_width": tile_width,
            "tile_height": tile_height,
            "tiles_per_gauss": tiles_per_gauss,
            "isect_ids": isect_ids,
            "flatten_ids": flatten_ids,
            "isect_offsets": isect_offsets,
            "width": width,
            "height": height,
            "tile_size": tile_size,
            "n_cameras": C,
        }
    )

    # print("rank", world_rank, "Before rasterize_to_pixels")
    if colors.shape[-1] > channel_chunk:
        # slice into chunks
        n_chunks = (colors.shape[-1] + channel_chunk - 1) // channel_chunk
        render_colors, render_alphas = [], []
        for i in range(n_chunks):
            colors_chunk = colors[..., i * channel_chunk : (i + 1) * channel_chunk]
            backgrounds_chunk = (
                backgrounds[..., i * channel_chunk : (i + 1) * channel_chunk]
                if backgrounds is not None
                else None
            )
            render_colors_, render_alphas_ = rasterize_to_pixels(
                means2d,
                conics,
                colors_chunk,
                opacities,
                width,
                height,
                tile_size,
                isect_offsets,
                flatten_ids,
                backgrounds=backgrounds_chunk,
                packed=packed,
                absgrad=absgrad,
            )
            render_colors.append(render_colors_)
            render_alphas.append(render_alphas_)
        render_colors = torch.cat(render_colors, dim=-1)
        render_alphas = render_alphas[0]  # discard the rest
    else:
        render_colors, render_alphas = rasterize_to_pixels(
            means2d,
            conics,
            colors,
            opacities,
            width,
            height,
            tile_size,
            isect_offsets,
            flatten_ids,
            backgrounds=backgrounds,
            packed=packed,
            absgrad=absgrad,
        )
    if render_mode in ["ED", "RGB+ED", "D", "RGB+D"]:
        meta['depths'] = render_colors[..., depth_index]
    if render_mode in ["ED", "RGB+ED", "ED"]:
        # normalize the accumulated depth to get the expected depth
        meta['depths'] = (meta['depths'] / render_alphas.clamp(min=1e-10),)
    if log_betas is not None:
        meta['betas'] = render_colors[..., betas_index]
    if render_mode not in ['D', 'ED']:
        render_colors = render_colors[..., :3]
    else:
        render_colors = None

    return render_colors, render_alphas, meta
