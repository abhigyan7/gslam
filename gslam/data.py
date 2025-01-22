from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pyquaternion import Quaternion
from torch.multiprocessing import JoinableQueue, Process
from threading import Event

from .primitives import Camera, Frame, Pose


class TumRGB:
    def __init__(self, sequence_dir: Path, seq_len: int = -1):
        self.sequence_dir = Path(sequence_dir)

        rgb_frames = np.loadtxt(self.sequence_dir / "rgb.txt", np.str_)
        self.rgb_frame_timestamps = rgb_frames[:, 0].astype(np.float64)
        self.rgb_frame_filenames = rgb_frames[:, 1].astype(np.str_)

        depth_frames = np.loadtxt(self.sequence_dir / "depth.txt", np.str_)
        self.depth_frame_timestamps = depth_frames[:, 0].astype(np.float64)
        self.depth_frame_filenames = depth_frames[:, 1].astype(np.str_)

        self.num_frames = len(self.rgb_frame_filenames)

        ground_truth = np.loadtxt(self.sequence_dir / "groundtruth.txt", np.str_)

        gt_timestamps = ground_truth[:, 0].astype(np.float64)
        gt_poses = ground_truth[:, 1:].astype(np.float64)

        # associate nearest-in-time frame and gt pose
        # TODO see if interpolating pose is better
        nearest_timestamp_ids = np.abs(
            np.subtract.outer(
                self.rgb_frame_timestamps,
                gt_timestamps,
            )
        ).argmin(axis=1)  # argmin in rgb frames because at least one sequence
        # has more poses than rgb frames

        gt_translations = gt_poses[nearest_timestamp_ids][:, :3]
        gt_quaternions = gt_poses[nearest_timestamp_ids][:, 3:]
        gt_rotation_matrices = np.array(
            [Quaternion(*q).rotation_matrix for q in gt_quaternions]
        )

        self.poses = np.tile(np.eye(4, dtype=np.float64), [self.num_frames, 1, 1])
        self.poses[..., :3, :3] = gt_rotation_matrices
        self.poses[..., :3, 3] = gt_translations

        self.length = self.num_frames
        if seq_len > 0:
            self.length = min(self.num_frames, seq_len)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration
        rgb_filename = self.sequence_dir / self.rgb_frame_filenames[idx]
        image = np.asarray(Image.open(rgb_filename))
        image = np.float32(image) / 255.0
        image = torch.Tensor(image).cuda()
        height, width, _channels = image.shape

        depth_filename = self.sequence_dir / self.depth_frame_filenames[idx]
        depth_image = np.asarray(Image.open(depth_filename))
        depth_image = torch.Tensor(depth_image.copy()).cuda() / 5000.0

        gt_pose = torch.Tensor(self.poses[idx, ...])
        ts = self.rgb_frame_timestamps[idx]
        Ks = torch.FloatTensor(
            [
                [525.0, 0.0, 319.5],
                [0.0, 525.5, 239.5],
                [0.0, 0.0, 0.0],
            ]
        ).cuda()
        camera = Camera(Ks, height, width)
        frame = Frame(
            image,
            ts,
            camera,
            Pose(),
            gt_pose,
            gt_depth=depth_image,
            img_file=rgb_filename,
            index=idx,
        )
        return frame


class RGBSensorStream(Process):
    def __init__(self, dataset, queue, frontend_done_event):
        super().__init__()
        self.dataset = dataset
        self.queue = queue
        self.frontend_done_event: Event = frontend_done_event

    # @rr.shutdown_at_exit
    def run(self):
        for data in iter(self.dataset):
            while self.queue.qsize() > 10:
                # preventing choke
                continue
            self.queue.put(data)
        self.queue.put(None)
        self.frontend_done_event.wait()
        return


if __name__ == "__main__":
    td = TumRGB("/home/abhigyan/gslam/datasets/tum/rgbd_dataset_freiburg1_desk")
    print(td[0])
    print(td[1])

    queue = JoinableQueue()
    stream = RGBSensorStream(td, queue)
    stream.run()
    while not queue.empty():
        print(queue.get())
        queue.task_done()
    # stream.join()
