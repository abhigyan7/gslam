#!/usr/bin/env python3

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch

from .primitives import Frame
from .rasterization import RasterizationOutput
from .utils import false_colormap, torch_to_pil


@torch.no_grad()
def log_frame(
    f: Frame,
    name: str = "/tracking/pose",
    outputs: RasterizationOutput = None,
    loss: float = None,
    tracking_time: float = None,
    is_tracking_frame: bool = False,
) -> None:
    q, t = f.pose.to_qt()
    q = np.roll(q.detach().cpu().numpy().reshape(-1), -1)
    t = t.detach().cpu().numpy().reshape(-1)
    rr.log(
        name,
        rr.Transform3D(
            rotation=rr.datatypes.Quaternion(xyzw=q),
            translation=t,
            from_parent=True,
        ),
    )

    rr.log(
        f"{name}/image",
        rr.Pinhole(
            resolution=[f.camera.width, f.camera.height],
            focal_length=[
                f.camera.intrinsics[0, 0].item(),
                f.camera.intrinsics[1, 1].item(),
            ],
            principal_point=[
                f.camera.intrinsics[0, 2].item(),
                f.camera.intrinsics[1, 2].item(),
            ],
        ),
    )
    if is_tracking_frame:
        rr.log(
            '/tracking/frame_index',
            rr.TextDocument(f"# {f.index}", media_type=rr.MediaType.MARKDOWN),
        )

    if outputs is not None and f.img is not None:
        rr.log(f"{name}/image", rr.Image(torch_to_pil(outputs.rgbs[0])).compress(90))
        rr.log(f"{name}/gt_image", rr.Image(torch_to_pil(f.img)).compress(90))

        errormap = (f.img - outputs.rgbs[0]).abs()  # H, W
        errormap = torch_to_pil(errormap)
        rr.log(f"{name}/errormap", rr.Image(errormap).compress(90))

        betas = false_colormap(outputs.betas[0].clip(max=2 * 2.7172))
        rr.log(f"{name}/uncertainty", rr.Image(betas).compress(70))

        depths = outputs.depthmaps[0]
        depths_min = depths[outputs.alphas[0, ..., 0] > 0.9].min().item()
        depths_max = depths[outputs.alphas[0, ..., 0] > 0.9].max().item()
        depths = false_colormap(outputs.depthmaps[0], near=depths_min, far=depths_max)
        rr.log(f"{name}/depth", rr.Image(depths).compress(70))

        alpha = false_colormap(outputs.alphas[0].squeeze(-1))
        rr.log(f"{name}/alpha", rr.Image(alpha).compress(70))

        betas = false_colormap(outputs.betas[0].log())
        rr.log(f"{name}/uncertainty", rr.Image(betas).compress(70))

    if loss is not None:
        rr.log('/tracking/loss', rr.Scalar(loss))

    if tracking_time is not None:
        tracking_time = min(30.0, tracking_time)
        rr.log('/tracking/fps', rr.Scalar(1.0 / tracking_time))


def get_blueprint() -> rrb.Blueprint:
    blueprint = rrb.Horizontal(
        rrb.Vertical(
            rrb.Spatial3DView(
                name='3D',
                origin='/tracking',
                contents=["$origin/**", "- /tracking/pc", "- /tracking/kf/**"],
            ),
            rrb.Horizontal(
                rrb.TimeSeriesView(name='tracking loss', origin="/tracking/loss"),
                rrb.TextDocumentView(
                    name='frame_index', origin='/tracking/frame_index'
                ),
                rrb.TimeSeriesView(name='ate', origin="/tracking/ate"),
                column_shares=[6, 1, 6],
            ),
            row_shares=[4, 1],
        ),
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial2DView(name="render", origin="/tracking/pose/image"),
                rrb.Spatial2DView(name="gt image", origin="/tracking/pose/gt_image"),
            ),
            rrb.Horizontal(
                rrb.Spatial2DView(name="errormap", origin="/tracking/pose/errormap"),
                rrb.Spatial2DView(
                    name="uncertainty", origin="/tracking/pose/uncertainty"
                ),
            ),
            rrb.Horizontal(
                rrb.Spatial2DView(name="alpha", origin="/tracking/pose/alpha"),
                rrb.Spatial2DView(name="depth", origin="/tracking/pose/depth"),
            ),
            rrb.TimeSeriesView(name='tracking fps', origin='/tracking/fps'),
        ),
        column_shares=[7, 3],
    )

    return rrb.Blueprint(blueprint, collapse_panels=True)


@torch.no_grad()
def log_splats(splats):
    modified_colors = splats.colors.detach().cpu().numpy()
    modified_opacities = splats.opacities.detach().cpu().numpy()
    modified_colors = 1 / (
        1
        + np.exp(
            -np.concatenate([modified_colors, modified_opacities[..., None]], axis=1)
        )
    )
    if splats.ages.max() != 0:
        modified_colors[
            splats.ages.cpu().numpy() == splats.ages.max().cpu().numpy()
        ] = np.array([[0, 1, 0, 1]])
    rr.log(
        '/tracking/pc',
        rr.Points3D(
            positions=splats.means.detach().cpu().numpy(),
            radii=torch.exp(splats.scales).min(dim=-1).values.detach().cpu().numpy()
            * 0.5,
            colors=modified_colors,
        ),
    )

    transparency = torch.sigmoid(splats.opacities)
    radii = splats.scales.exp() * transparency.unsqueeze(-1) * 2.0 + 0.004
    q = splats.quats.cpu().numpy()
    q = np.roll(q, -1, axis=1)
    rr.log(
        '/tracking/splats',
        rr.Ellipsoids3D(
            half_sizes=radii.cpu().numpy(),
            centers=splats.means.cpu().numpy(),
            quaternions=q,
            colors=modified_colors,
            fill_mode=rr.components.FillMode.Solid,
        ),
    )
