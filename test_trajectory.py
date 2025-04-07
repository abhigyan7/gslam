#!/usr/bin/env python3

from gslam.trajectory import Trajectory
from gslam.data import TumAsync, SensorTypes
from gslam.primitives import IMUFrame
import pypose as pp
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import rerun as rr
import time

torch.set_default_device('cuda')
# torch.autograd.set_detect_anomaly(True)


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


rr.init('spline', recording_id=f'spline_{int(time.time())}', spawn=True)

dataset = TumAsync('/mnt/data/datasets/rgbd_dataset_freiburg1_room')
traj = Trajectory(0.25, dataset.gt_timestamps[0])

timestamps = []
for gt_pose in dataset.poses[::10]:
    tx = torch.from_numpy(gt_pose[:3, 3]).requires_grad_(True).detach()
    R = gt_pose[:3, :3]
    SO3 = pp.mat2SO3(R).requires_grad_(True).detach()
    traj.add_control_point(SO3, tx)
timestamps.extend(dataset.gt_timestamps[::10])

optimizer = torch.optim.Adam(traj.parameters())
starting_time = None
end_time = None

for j in tqdm(range(100)):
    for i in tqdm(range(0, 1000, 10), leave=False):
        sensor_type, frame = dataset[i]
        if sensor_type == SensorTypes.IMU:
            frame: IMUFrame = frame
            gt_accel = frame.accel
            observed_accel = traj.acceleration(frame.timestamp, gravity=True)
            loss = (gt_accel - observed_accel).square().sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if starting_time is None:
                starting_time = frame.timestamp
            end_time = frame.timestamp

print(traj)

timestamps = np.linspace(
    traj.starting_time, traj.starting_time + traj.interval * len(traj.cps_SO3), 100
)
timestamps = np.linspace(starting_time, end_time, 100)
timestamps = np.array(timestamps)
interps = []
with torch.no_grad():
    for i, t in tqdm(enumerate(timestamps), leave=False):
        rot_q, translation = traj(t.item())
        vel = traj.velocity(t.item())
        mag = vel.square().sum().sqrt().item()
        rr.log(
            '/vel',
            rr.Scalar(
                mag,
            ),
        )
        accel = traj.acceleration(t.item(), gravity=True)
        mag = accel.square().sum().sqrt().item()
        rr.log(
            '/accel',
            rr.Scalar(
                mag,
            ),
        )
        mag = torch.tensor(dataset.accel_frames[i]).square().sum().sqrt().item()
        rr.log(
            '/accel_gt',
            rr.Scalar(
                mag,
            ),
        )
        log_pose(rot_q, translation)
        interps.append(translation)

interps = torch.stack(interps)

# plot_points(xyz)
# plt.savefig('test.png')
plot_points(interps.detach().cpu())
plt.savefig('test_interps.png')
