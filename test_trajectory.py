#!/usr/bin/env python3

from gslam.trajectory import Trajectory
from gslam.data import TumAsync, SensorTypes
from gslam.primitives import IMUFrame
from gslam.utils import create_batch
import pypose as pp
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import rerun as rr
import time

# rr = MagicMock()

device = 'cuda'
torch.set_default_device(device)
# torch.autograd.set_detect_anomaly(True)


def plot_points(pts):
    pts = np.array(pts.detach())
    plt.scatter(pts[..., 0], pts[..., 1])


def log_pose(r: pp.SO3_type, t: torch.Tensor, is_kf=False, i: int = 0):
    q = np.roll(r.detach().cpu().numpy().reshape(-1), -1)
    t = t.detach().cpu().numpy().reshape(-1)
    name = f'/control_point/{i}' if is_kf else '/interpolated_point'
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

dataset = TumAsync(
    '/mnt/data/datasets/rgbd_dataset_freiburg1_xyz', factors=(SensorTypes.IMU,)
)

# traj = Trajectory(0.25, dataset[0][1].timestamp-0.25)
# for frame_type, frame in dataset:
#     print(f'{traj.cursor=}, {traj.support_end()=}')
#     print(f'{frame_type=}, {frame.timestamp=}')
#     ret = traj.extend_to_time(frame.timestamp)
#     traj(frame.timestamp)
#     print()
#     if frame_type == SensorTypes.RGB:
#         pass
#     elif frame_type == SensorTypes.Depth:
#         pass
#     else:
#         pass
#     time.sleep(0.01)
#
# exit()

poses = dataset.poses[:1000:10]
timestamps = dataset.gt_timestamps[:1000:10]
interval = timestamps[2] - timestamps[1]
traj = Trajectory(interval, dataset.gt_timestamps[0])
for gt_pose in tqdm(poses):
    tx = torch.from_numpy(gt_pose[:3, 3]).requires_grad_(True).detach()
    R = gt_pose[:3, :3]
    SO3 = pp.mat2SO3(R).requires_grad_(True).detach()
    traj.add_control_point(SO3, tx)

optimizer = torch.optim.Adam(traj.parameters())

batch_size = 1000
for epoch in tqdm(range(10)):
    for batch in tqdm(range(0, len(dataset) - 2 * batch_size, batch_size), leave=False):
        frames = [dataset[i][1] for i in range(batch, batch + batch_size)]
        gt_accels = create_batch(frames, lambda x: x.accel)
        times = torch.tensor(
            [f.timestamp for f in frames], device=device, dtype=torch.double
        )
        observed_accels = traj.acceleration(times)
        loss = (gt_accels - observed_accels).square().mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

print(traj)
timestamps = np.linspace(timestamps[0].item(), timestamps[-1].item(), 1000)
timestamps = np.array(timestamps)

interps = []
with torch.no_grad():
    for idx, (frame_type, frame) in tqdm(enumerate(dataset), total=len(dataset)):
        if idx % 5 > 0:
            continue
        if frame.timestamp < (traj.starting_time + traj.interval):
            continue
        if frame.timestamp > (traj.support_end() - 2 * traj.interval):
            continue
        frame: IMUFrame = frame
        if frame_type == SensorTypes.RGB:
            continue
        if frame_type == SensorTypes.Depth:
            continue
        t = frame.timestamp
        t = torch.tensor([t.item()], device=device, dtype=torch.double)
        rot_q, translation = traj(t)
        log_pose(rot_q, translation)
        segment, _t = traj._parse_time(t.item())
        rr.log(
            '/segment',
            rr.Scalar(
                segment - 1,
            ),
        )
        vel = traj.velocity(t)
        mag = vel.square().sum().sqrt().item()
        rr.log(
            '/vel',
            rr.Scalar(
                mag,
            ),
        )
        accel = traj.acceleration(t)
        mag = accel.square().sum().sqrt().item()
        rr.log(
            '/accel',
            rr.Scalar(
                mag,
            ),
        )
        mag = torch.tensor(frame.accel).square().sum().sqrt().item()
        rr.log(
            '/accel_gt',
            rr.Scalar(
                mag,
            ),
        )
        interps.append(translation)

print(f'{traj.cursor=}')
for i in range(traj.cursor):
    rot_r = traj.cps_R3[i]
    rot_q = traj.cps_SO3[i]
    log_pose(rot_q, rot_r, is_kf=True, i=i)

for i in range(traj.cursor):
    rot_r = traj.cps_R3[i]
    rot_q = traj.cps_SO3[i]
    log_pose(rot_q, rot_r)
interps = torch.stack(interps)

# plot_points(xyz)
# plt.savefig('test.png')
plot_points(interps.detach().cpu())
plt.savefig('test_interps.png')
