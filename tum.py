from pathlib import Path

import numpy as np
from PIL import Image
from pyquaternion import Quaternion


class TumRGB:
    def __init__(self, sequence_dir: Path):
        self.sequence_dir = Path(sequence_dir)

        rgb_frames = np.loadtxt(self.sequence_dir / "rgb.txt", np.str_)
        self.rgb_frame_timestamps = rgb_frames[:, 0].astype(np.float64)
        self.rgb_frame_filenames = rgb_frames[:, 1].astype(np.str_)

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

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx):
        filename = self.sequence_dir / self.rgb_frame_filenames[idx]
        image = np.asarray(Image.open(filename))
        pose = self.poses[idx, ...]
        return image, pose, self.rgb_frame_timestamps[idx]


if __name__ == "__main__":
    td = TumRGB("/home/abhigyan/gslam/datasets/tum/rgbd_dataset_freiburg1_desk")
    print(td[0])
    print(td[1])
