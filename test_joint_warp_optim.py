#!/usr/bin/env python3

from gslam.data import TumRGB
from gslam.warp import Warp
from gslam.primitives import Frame, Pose
from gslam.utils import false_colormap, torch_to_pil
import torch
import tqdm
import torch.nn.functional as F
import rerun as rr
import numpy as np

dataset = TumRGB('/mnt/data/datasets/rgbd_dataset_freiburg3_long_office_household/')

device = 'cuda'


def log_frame(f: Frame):
    i = f.index
    q, t = f.pose.to_qt()
    q = np.roll(q.detach().cpu().numpy().reshape(-1), -1)
    t = t.detach().cpu().numpy().reshape(-1)

    rr.log(
        f'/tracking/pose_{i}',
        rr.Transform3D(
            rotation=rr.datatypes.Quaternion(xyzw=q),
            translation=t,
            from_parent=True,
        ),
    )

    rr.log(
        f'/tracking/pose_{i}/camera',
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


def total_variation_loss(img: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    v_h = img[..., 1:, :] - img[..., :-1, :]
    v_w = img[..., :, 1:] - img[..., :, :-1]
    if mask is not None:
        v_h = v_h * mask[..., 1:, :]
        v_w = v_w * mask[..., :, 1:]
    tv_h = (v_h).pow(2).mean()
    tv_w = (v_w).pow(2).mean()
    return tv_h + tv_w


s_id = 200
kf1 = dataset[s_id].to(device)
depths = torch.ones(
    [kf1.camera.height, kf1.camera.width], device=device, requires_grad=True
).float()


kfids = range(s_id, s_id + 8, 1)
# kfs = [dataset[i].to(device) for i in [1]]
# kfs = [dataset[i].to(device) for i in [1, 2]]
# kfs = [dataset[i].to(device) for i in [1, 2, 3, 4]]
# kfs = [dataset[i].to(device) for i in [1,2,3,4,5,6,7,8]]
# kfs = [dataset[i].to(device) for i in [1,3,5,7,9,11,13,15]]

rr.init('gslam_jpt', recording_id='gslam_jpt')
rr.save('rr.rrd')
rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)


depths = torch.ones([kf1.camera.height, kf1.camera.width], device=device).float()
ddepths = torch.zeros(
    [kf1.camera.height, kf1.camera.width], device=device, requires_grad=True
).float()
params = [ddepths]

dm_optimizer = torch.optim.Adam(params)

warp = Warp(kf1.camera.intrinsics, kf1.camera.height, kf1.camera.width)

log_frame(kf1)

kfs = []
for kfid in kfids:
    kf = dataset[kfid].to(device)
    with torch.no_grad():
        if len(kfs) >= 1:
            kf.pose = Pose(kfs[-1].pose()).to(device)
    kfs.append(kf)
    pose_optimizer = torch.optim.Adam([kf.pose.dR, kf.pose.dt])

    for i in (pbar := tqdm.trange(100)):
        dm_optimizer.zero_grad()
        pose_optimizer.zero_grad()
        loss = 0
        result, _normalized_warps, keep_mask = warp(
            kf1.pose(),
            kf.pose(),
            kf1.img,
            ddepths + depths,
        )
        loss = loss + F.l1_loss(result[keep_mask], kf.img[keep_mask])
        log_frame(kf)
        pbar.set_description(f'Loss: {loss.item():.3f}')
        loss += 20.0 * total_variation_loss(ddepths + depths)
        # fix scale
        # loss += (torch.norm(kfs[0].pose()[:3, 3]) - 1.0).abs() * 0.0001
        loss.backward()
        pose_optimizer.step()
        dm_optimizer.step()
    false_colormap(ddepths.abs()).save(f'jpt/ddepth_{kfid}.png')
    torch_to_pil((kf.img - result), minmax_norm=True).save(
        f'jpt/residual_{kf.index}.png'
    )
    vis_depth = ddepths + depths
    vis_depth[ddepths.abs() < 0.1] = 0.0
    false_colormap(vis_depth).save(f'jpt/ddepth_vis_{kfid}.png')
    false_colormap(ddepths + depths).save(f'jpt/depth_{kfid}.png')
    pose_optimizer.zero_grad()
    dm_optimizer.zero_grad()

false_colormap(ddepths + depths).save('jpt/depth_final.png')
false_colormap(kf1.gt_depth).save('jpt/lt_depth.png')

with torch.no_grad():
    for kf in kfs:
        result, _normalized_warps, keep_mask = warp(
            kf1.pose(),
            kf.pose(),
            kf1.img,
            ddepths + depths,
        )
        torch_to_pil(result).save(f'jpt/{kf.index}_bhako.png')
        torch_to_pil(kf.img).save(f'jpt/{kf.index}_hunuparne.png')
        torch_to_pil((kf.img - result), minmax_norm=True).save(
            f'jpt/residual_{kf.index}.png'
        )
        torch_to_pil(keep_mask).save(f'jpt/mask_{kf.index}.png')
log_frame(kf1)

vis_depth = ddepths + depths
vis_depth[ddepths.abs() < 0.1] = 0.0
rr.log(f'/tracking/pose_{kfs[-1].index}/camera/depth', rr.DepthImage(vis_depth))
rr.log(f'/tracking/pose_{kf1.index}/camera/depth', rr.DepthImage(kf1.gt_depth))
rr.log(f'/tracking/pose_{kf1.index}/camera', rr.Image(kf.img))
rr.log(f'/tracking/pose_{kfs[-1].index}/camera', rr.Image(kfs[-1].img))
