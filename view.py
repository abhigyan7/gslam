#!/usr/bin/env python3

import argparse
import time
from typing import Tuple

import nerfview
import torch
import viser

from gslam.map import GaussianSplattingData
from gslam.primitives import Pose, Camera

torch.serialization.add_safe_globals([GaussianSplattingData, set])


def main(args):
    torch.manual_seed(42)
    device = torch.device("cuda")

    splats = torch.load(args.ckpt, map_location=device, weights_only=True)

    print("Number of Gaussians:", len(splats.means))

    # register and open viewer
    @torch.no_grad()
    def viewer_render_fn(camera_state: nerfview.CameraState, img_wh: Tuple[int, int]):
        width, height = img_wh
        c2w = camera_state.c2w
        K = camera_state.get_K(img_wh)
        c2w = torch.from_numpy(c2w).float().to(device)
        K = torch.from_numpy(K).float().to(device)
        viewmat = c2w.inverse()

        camera = Camera(K, height, width)
        pose = Pose(viewmat, False)

        outputs = splats([camera], [pose])
        render_rgbs = outputs.rgbs[0, ...].cpu().numpy()
        return render_rgbs

    server = viser.ViserServer(port=args.port, verbose=False)
    _ = nerfview.Viewer(
        server=server,
        render_fn=viewer_render_fn,
        mode="rendering",
    )
    print("Viewer running... Ctrl+C to exit.")
    time.sleep(100000)


if __name__ == "__main__":
    """
    # Use single GPU to view the scene
    python simple_viewer.py \
        --ckpt ckpt.pt \
        --port 8081
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="path to the .pt file")
    parser.add_argument(
        "--port", type=int, default=8080, help="port for the viewer server"
    )
    args = parser.parse_args()
    main(args)
