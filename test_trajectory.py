#!/usr/bin/env python3

from gslam.trajectory import Trajectory
from gslam.data import TumRGB
from gslam.primitives import Frame
import pypose as pp
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import rerun as rr

# torch.set_default_device('cuda')


def plot_points(pts):
    pts = np.array(pts.detach())
    plt.scatter(pts[..., 0], pts[..., 1])


def log_pose(r: pp.SO3_type, t: torch.Tensor, is_kf=False, i: int = 0):
    q = np.roll(r.detach().cpu().numpy().reshape(-1), -1)
    t = t.detach().cpu().numpy().reshape(-1)
    name = f'/control_point_{i}' if is_kf else '/interpolated_point'
    rr.log(
        name,
        rr.Transform3D(
            rotation=rr.datatypes.Quaternion(xyzw=q),
            translation=t,
            from_parent=True,
        ),
        static=is_kf,
    )
    rr.log(
        name,
        rr.Pinhole(
            resolution=[640, 480],
            focal_length=[
                500,
                500,
            ],
            principal_point=[
                320,
                240,
            ],
        ),
        static=is_kf,
    )


traj = Trajectory(1.0, 0.0)

dataset = TumRGB('/mnt/data/datasets/rgbd_dataset_freiburg1_room', 2500)

traj.cps_SO3.append(pp.identity_SO3())
traj.cps_SO3.append(pp.randn_SO3())
traj.cps_SO3.append(pp.randn_SO3())
traj.cps_SO3.append(pp.randn_SO3())
traj.cps_SO3.append(pp.randn_SO3())

traj.cps_R3.append(torch.tensor([0, 0, 1]).double().requires_grad_(True))
traj.cps_R3.append(torch.tensor([1, 1, 1]).double().requires_grad_(True))
traj.cps_R3.append(torch.tensor([2, 2, 1]).double().requires_grad_(True))
traj.cps_R3.append(torch.tensor([3, 3, 1]).double().requires_grad_(True))
traj.cps_R3.append(torch.tensor([4, 4, 1]).double().requires_grad_(True))

# print(traj(-1.0))
# print(traj(0.0))
# print(traj(1.0))
# print(traj(2.0))
# print(traj(3.0))
# print(traj(4.0))
# print(traj(5.0))
# print(traj(6.0))

traj = Trajectory(0.13, 4.0)
N = 100
xyz = []

rr.init('spline', recording_id='spline', spawn=True)
starting_time = None
interval = None
end_time = None
for i in tqdm(range(0, len(dataset), 30)):
    frame = dataset[i]
    accel: np.ndarray = dataset.accel_frames[i]
    mag = np.power(np.power(accel, 2.0).sum(), 0.5).item() - 9.8
    rr.log(
        '/accel_gt',
        rr.Scalar(
            mag,
        ),
    )

    f: Frame = frame.to('cpu')
    pose = f.gt_pose
    tx = pose[:3, 3]
    R = pose[:3, :3]
    SO3 = pp.mat2SO3(R)
    traj.cps_SO3.append(SO3.requires_grad_(True))
    traj.cps_R3.append(tx.requires_grad_(True))
    log_pose(SO3, tx, True, i)
    xyz.append(tx)
    if starting_time is None:
        starting_time = f.timestamp
    elif interval is None:
        interval = f.timestamp - starting_time
    end_time = f.timestamp

print(traj)
print(starting_time)
print(interval)
traj.starting_time = starting_time
traj.interval = interval
xyz = torch.stack(xyz)

timestamps = np.linspace(starting_time, end_time, 3000)
interps = []
with torch.no_grad():
    for t in tqdm(timestamps):
        rot_q, translation = traj(t.item())
        vel = traj.velocity(t.item())
        mag = vel.square().sum().sqrt().item()
        rr.log(
            '/vel',
            rr.Scalar(
                mag,
            ),
        )
        accel = traj.acceleration(t.item())
        mag = accel.square().sum().sqrt().item()
        rr.log(
            '/accel',
            rr.Scalar(
                mag,
            ),
        )
        log_pose(rot_q, translation)
        interps.append(translation)

interps = torch.stack(interps)

plot_points(xyz)
plt.savefig('test.png')
plot_points(interps)
plt.savefig('test_interps.png')
