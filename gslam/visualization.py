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
    f: Frame, name: str = "/tracking/pose", outputs: RasterizationOutput = None
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

    if outputs is not None and f.img is not None:
        rr.log(f"{name}/image", rr.Image(torch_to_pil(outputs.rgbs[0])).compress(90))
        rr.log(f"{name}/gt_image", rr.Image(torch_to_pil(f.img)).compress(90))

        errormap = (f.img - outputs.rgbs[0]).abs()  # H, W
        errormap = torch_to_pil(errormap)
        rr.log(f"{name}/errormap", rr.Image(errormap).compress(90))

        betas = false_colormap(outputs.betas[0].log())
        rr.log(f"{name}/uncertainty", rr.Image(betas).compress(70))

        depths = outputs.depthmaps[0]
        depths_min = depths[outputs.betas[0] < 2.0].min().item()
        depths_max = depths[outputs.betas[0] < 2.0].max().item()
        depths = false_colormap(outputs.depthmaps[0], near=depths_min, far=depths_max)
        rr.log(f"{name}/depth", rr.Image(depths).compress(70))

        alpha = false_colormap(outputs.alphas[0].squeeze(-1))
        rr.log(f"{name}/alpha", rr.Image(alpha).compress(70))

        betas = false_colormap(outputs.betas[0].log())
        rr.log(f"{name}/uncertainty", rr.Image(betas).compress(70))


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
