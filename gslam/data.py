from datetime import timedelta
from pathlib import Path
import tempfile
import time

import depthai as dai
import numpy as np
import torch
from PIL import Image
from pyquaternion import Quaternion
from torch.multiprocessing import JoinableQueue, Process, Event
from tqdm import tqdm
import cv2

from typing import assert_never

from enum import StrEnum, auto

from .primitives import Camera, Frame, PoseZhou as Pose, IMUFrame, DepthFrame

import os

tum_intrinsics_params = {
    "freiburg1": [517.3, 516.5, 318.6, 255.3, 0.2624, -0.9531, -0.0054, 0.0026, 1.1633],
    "freiburg2": [
        520.9,
        521.0,
        325.1,
        249.7,
        0.2312,
        -0.7849,
        -0.0033,
        -0.0001,
        0.9172,
    ],
    "freiburg3": [535.4, 539.2, 320.1, 247.6, 0, 0, 0, 0, 0],
}

oakd_intrinsics = [
    2285.444580078125,
    2283.459716796875,
    1939.57421875,
    1084.2498779296875,
    7.613864421844482,
    17.2153263092041,
    0.00012345814320724458,
    -0.00020271474204491824,
    -1.0174853801727295,
    7.641190528869629,
    19.72230339050293,
    1.2936210632324219,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
]


class SensorTypes(StrEnum):
    IMU = auto()
    RGB = auto()
    Depth = auto()


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
        # tum is xyzw, pyquaternion is wxyz
        gt_quaternions = np.roll(gt_quaternions, 1, axis=1)
        gt_rotation_matrices = np.array(
            [Quaternion(*q).rotation_matrix for q in gt_quaternions]
        )

        self.poses = np.tile(np.eye(4, dtype=np.float64), [self.num_frames, 1, 1])
        self.poses[..., :3, :3] = gt_rotation_matrices
        self.poses[..., :3, 3] = gt_translations

        self.length = self.num_frames
        if seq_len > 0:
            self.length = min(self.num_frames, seq_len)

        sequence_type = str(self.sequence_dir.parts[-1]).split('_')[2]
        intrinsics = tum_intrinsics_params[sequence_type]

        fx, fy, cx, cy, *d = intrinsics

        K = np.array(
            [
                [
                    fx,
                    0,
                    cx,
                ],
                [
                    0,
                    fy,
                    cy,
                ],
                [
                    0,
                    0,
                    1,
                ],
            ],
            dtype=np.float32,
        )

        self.Ks, self.roi = cv2.getOptimalNewCameraMatrix(
            K, np.array(d), (640, 480), 0, (640, 480)
        )
        self.undistort_map_x, self.undistort_map_y = cv2.initUndistortRectifyMap(
            K,
            np.array(d),
            None,
            self.Ks,
            (640, 480),
            cv2.CV_32FC1,
        )

        self.Ks = torch.tensor(self.Ks).cuda()
        # we need to hold on to this because once this is gc'd
        # python deletes the tmpdir
        self.tmpdir_object = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self.tmpdir_object.name)

        self.accel_frames = np.loadtxt(
            self.sequence_dir / "accelerometer.txt", np.double
        )
        self.accel_frames = self.accel_frames[..., 1:]

    def init(self):
        return

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration
        rgb_filename = self.sequence_dir / self.rgb_frame_filenames[idx]
        im = Image.open(rgb_filename)
        image = cv2.remap(
            np.array(im),
            self.undistort_map_x,
            self.undistort_map_y,
            cv2.INTER_LINEAR,
        )
        x, y, w, h = self.roi
        image = image[y : y + h, x : x + w]

        # save undistorted gt_img to tmp because we'll need it to evaluate reconstruction later
        gt_img_save_path = self.tmpdir / rgb_filename.parts[-1]
        cv2.imwrite(str(gt_img_save_path), image[..., ::-1])
        image = np.asarray(np.float32(image)) / 255.0
        image = torch.Tensor(image).cuda()
        height, width, _channels = image.shape

        depth_filename = self.sequence_dir / self.depth_frame_filenames[idx]
        depth_im = Image.open(depth_filename)
        depth_image = np.asarray(depth_im)
        depth_image = depth_image[y : y + h, x : x + w]
        depth_image = torch.Tensor(depth_image.copy()).cuda() / 5000.0

        gt_pose = torch.Tensor(self.poses[idx, ...])
        ts = self.rgb_frame_timestamps[idx]
        camera = Camera(self.Ks.clone(), height, width)
        frame = Frame(
            image,
            ts,
            camera,
            Pose(),
            gt_pose,
            gt_depth=depth_image,
            img_file=gt_img_save_path,
            index=idx,
        )
        return frame


class Replica:
    def __init__(self, sequence_dir: Path, seq_len: int = -1):
        self.sequence_dir = Path(sequence_dir)

        filenames = sorted(os.listdir(sequence_dir / 'results'))
        self.rgb_frame_filenames = [f for f in filenames if f.startswith('frame')]
        self.depth_frame_filenames = [f for f in filenames if f.startswith('depth')]

        self.num_frames = len(self.rgb_frame_filenames)
        self.length = self.num_frames
        if seq_len > 0:
            self.length = min(self.num_frames, seq_len)

        gt_poses = (
            np.loadtxt(self.sequence_dir / 'traj.txt', np.str_)
            .astype(np.float64)
            .reshape(-1, 4, 4)
        )

        # self.poses = np.linalg.inv(gt_poses)
        self.poses = gt_poses

        K = np.array(
            [[300, 0, 299.75], [0, 300, 169.75], [0, 0, 1]],
            dtype=np.float32,
        )

        self.Ks = torch.from_numpy(K).cuda()

    def init(self):
        return

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration
        rgb_filename = self.sequence_dir / ('results/' + self.rgb_frame_filenames[idx])
        image = Image.open(rgb_filename)
        image.thumbnail((600, 340), Image.Resampling.LANCZOS)
        image = np.asarray(np.float32(image)) / 255.0
        image = torch.Tensor(image).cuda()
        height, width, _channels = image.shape

        depth_filename = self.sequence_dir / (
            'results/' + self.depth_frame_filenames[idx]
        )
        depth_im = Image.open(depth_filename)
        depth_image = np.asarray(depth_im)
        depth_image = torch.Tensor(depth_image.copy()).cuda() / 5000.0

        gt_pose = torch.Tensor(self.poses[idx, ...])
        camera = Camera(self.Ks.clone(), height, width)
        frame = Frame(
            image,
            0.0,
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
        self.dataset.init()
        for data in iter(self.dataset):
            while self.queue.qsize() > 10:
                # preventing choke
                continue
            self.queue.put(data)
        self.queue.put(None)
        self.frontend_done_event.wait()
        return


RGB_SOCKET = dai.CameraBoardSocket.CAM_A
LEFT_SOCKET = dai.CameraBoardSocket.CAM_B
RIGHT_SOCKET = dai.CameraBoardSocket.CAM_C
ALIGN_SOCKET = LEFT_SOCKET

COLOR_RESOLUTION = dai.ColorCameraProperties.SensorResolution.THE_1080_P
LEFT_RIGHT_RESOLUTION = dai.MonoCameraProperties.SensorResolution.THE_400_P

ISP_SCALE = 3


class OakdSensor:
    def __init__(self, fps=30.0):
        self.pipeline = dai.Pipeline()
        self.camRgb = self.pipeline.create(dai.node.ColorCamera)
        self.left = self.pipeline.create(dai.node.MonoCamera)
        self.right = self.pipeline.create(dai.node.MonoCamera)
        self.stereo = self.pipeline.create(dai.node.StereoDepth)
        self.sync = self.pipeline.create(dai.node.Sync)
        self.out = self.pipeline.create(dai.node.XLinkOut)
        self.align = self.pipeline.create(dai.node.ImageAlign)

        self.left.setResolution(LEFT_RIGHT_RESOLUTION)
        self.left.setBoardSocket(LEFT_SOCKET)
        self.left.setFps(fps)

        self.right.setResolution(LEFT_RIGHT_RESOLUTION)
        self.right.setBoardSocket(RIGHT_SOCKET)
        self.right.setFps(fps)

        self.camRgb.setBoardSocket(RGB_SOCKET)
        self.camRgb.setResolution(COLOR_RESOLUTION)
        self.camRgb.setFps(fps)
        self.camRgb.setIspScale(1, ISP_SCALE)

        self.stereo.setDefaultProfilePreset(
            dai.node.StereoDepth.PresetMode.HIGH_DENSITY
        )
        self.stereo.setDepthAlign(dai.CameraBoardSocket.LEFT)

        self.out.setStreamName("out")

        self.sync.setSyncThreshold(timedelta(seconds=0.5 / fps))

        self.camRgb.isp.link(self.sync.inputs["rgb"])
        self.left.out.link(self.stereo.left)
        self.right.out.link(self.stereo.right)
        self.stereo.depth.link(self.align.input)
        self.align.outputAligned.link(self.sync.inputs["depth_aligned"])
        self.camRgb.isp.link(self.align.inputAlignTo)
        self.sync.out.link(self.out.input)

        self.idx = 0

    def start(self):
        self.device = dai.Device(self.pipeline)
        self.output_queue = self.device.getOutputQueue("out", 1, False)
        self.calibrationHandler = self.device.readCalibration()
        self.rgbDistortion = np.array(
            self.calibrationHandler.getDistortionCoefficients(RGB_SOCKET)
        )
        distortionModel = self.calibrationHandler.getDistortionModel(RGB_SOCKET)
        if distortionModel != dai.CameraModel.Perspective:
            raise RuntimeError(
                "Unsupported distortion model for RGB camera. This example supports only Perspective model."
            )
        for i in tqdm(range(30), desc='consuming frames'):
            _messageGroup = self.output_queue.get()
        print('started oakd sensor')

    def next(self):
        messageGroup = self.output_queue.get()
        assert isinstance(messageGroup, dai.MessageGroup)
        frameRgb = messageGroup["rgb"]
        assert isinstance(frameRgb, dai.ImgFrame)
        frameDepth = messageGroup["depth_aligned"]
        assert isinstance(frameDepth, dai.ImgFrame)

        cvFrame = frameRgb.getCvFrame()
        depth_image = frameDepth.getFrame()
        rgbIntrinsics = np.array(
            self.calibrationHandler.getCameraIntrinsics(
                RGB_SOCKET, int(cvFrame.shape[1]), int(cvFrame.shape[0])
            )
        )
        image = cv2.undistort(
            cvFrame,
            rgbIntrinsics,
            np.array(self.rgbDistortion),
        )
        image = np.asarray(np.float32(image)) / 255.0
        image = torch.Tensor(image).cuda()
        height, width, _channels = image.shape

        # milimeters to meters
        depth_image = torch.Tensor(np.float32(depth_image)).cuda() / 1000.0
        gt_pose = None
        ts = time.time()
        K = torch.Tensor(rgbIntrinsics).cuda()
        camera = Camera(K, height, width)
        gt_img_save_path = None
        frame = Frame(
            image,
            ts,
            camera,
            Pose(),
            gt_pose,
            gt_depth=depth_image,
            img_file=gt_img_save_path,
            index=self.idx,
        )
        self.idx += 1
        return frame


class TumAsync:
    def __init__(
        self,
        sequence_dir: Path,
        factors: tuple = (SensorTypes.RGB, SensorTypes.IMU, SensorTypes.Depth),
    ):
        self.sequence_dir = Path(sequence_dir)
        self.num_frames = 0

        rgb_frames_file = self.sequence_dir / "rgb.txt"
        if rgb_frames_file.exists() and rgb_frames_file.is_file():
            rgb_frames = np.loadtxt(rgb_frames_file, np.str_)
            self.rgb_frame_timestamps = rgb_frames[:, 0].astype(np.float64)
            self.rgb_frame_filenames = rgb_frames[:, 1].astype(np.str_)
            self.num_frames += len(rgb_frames)

        depth_frames_file = self.sequence_dir / "depth.txt"
        if depth_frames_file.exists() and depth_frames_file.is_file():
            depth_frames = np.loadtxt(depth_frames_file, np.str_)
            self.depth_frame_timestamps = depth_frames[:, 0].astype(np.float64)
            self.depth_frame_filenames = depth_frames[:, 1].astype(np.str_)
            self.num_frames += len(self.depth_frame_timestamps)

        accel_frames_file = self.sequence_dir / "accelerometer.txt"
        if accel_frames_file.exists() and accel_frames_file.is_file():
            accel_frames = np.loadtxt(accel_frames_file, np.double)
            self.accel_frames = accel_frames[..., 1:]
            mean_accel = self.accel_frames.mean(axis=0)
            self.accel_frames = self.accel_frames - mean_accel
            self.accel_timestamps = accel_frames[..., 0]
            self.num_frames += len(accel_frames)

        ground_truth_file = self.sequence_dir / "groundtruth.txt"
        ground_truth_frames = np.loadtxt(ground_truth_file, np.double)
        self.gt_timestamps = ground_truth_frames[:, 0].astype(np.float64)
        gt_poses = ground_truth_frames[:, 1:].astype(np.float64)

        gt_translations = gt_poses[:, :3]
        gt_quaternions = gt_poses[:, 3:]
        # tum is xyzw, pyquaternion is wxyz
        gt_quaternions = np.roll(gt_quaternions, 1, axis=1)
        gt_rotation_matrices = np.array(
            [Quaternion(*q).rotation_matrix for q in gt_quaternions]
        )

        self.poses = np.tile(np.eye(4, dtype=np.float64), [len(gt_translations), 1, 1])
        self.poses[..., :3, :3] = gt_rotation_matrices
        self.poses[..., :3, 3] = gt_translations

        sequence_type = str(self.sequence_dir.parts[-1]).split('_')[2]
        fx, fy, cx, cy, *d = tum_intrinsics_params[sequence_type]
        K = np.array(
            [
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )
        self.Ks, self.roi = cv2.getOptimalNewCameraMatrix(
            K, np.array(d), (640, 480), 0, (640, 480)
        )
        self.undistort_map_x, self.undistort_map_y = cv2.initUndistortRectifyMap(
            K,
            np.array(d),
            None,
            self.Ks,
            (640, 480),
            cv2.CV_32FC1,
        )
        self.Ks = torch.tensor(self.Ks).cuda()

        # we need to hold on to this because once this is gc'd
        # python deletes the tmpdir
        self.tmpdir_object = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self.tmpdir_object.name)

        # sort the frames by timestamp
        imu_timestamps = [
            (SensorTypes.IMU, idx, ts) for (idx, ts) in enumerate(self.accel_timestamps)
        ]
        rgb_timestamps = [
            (SensorTypes.RGB, idx, ts)
            for (idx, ts) in enumerate(self.rgb_frame_timestamps)
        ]
        depth_timestamps = [
            (SensorTypes.Depth, idx, ts)
            for (idx, ts) in enumerate(self.depth_frame_timestamps)
        ]

        all_timestamps = []
        if SensorTypes.IMU in factors:
            all_timestamps.extend(imu_timestamps)
        if SensorTypes.Depth in factors:
            all_timestamps.extend(depth_timestamps)
        if SensorTypes.RGB in factors:
            all_timestamps.extend(rgb_timestamps)

        self.sorted_timestamps = sorted(all_timestamps, key=lambda x: x[-1])

    def __len__(self):
        return len(self.sorted_timestamps)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration
        sensor_type, idx, ts = self.sorted_timestamps[idx]

        if sensor_type == SensorTypes.RGB:
            rgb_filename = self.sequence_dir / self.rgb_frame_filenames[idx]
            im = Image.open(rgb_filename)
            image = cv2.remap(
                np.array(im),
                self.undistort_map_x,
                self.undistort_map_y,
                cv2.INTER_LINEAR,
            )
            x, y, w, h = self.roi
            image = image[y : y + h, x : x + w]

            # save undistorted gt_img to tmp because we'll need it to evaluate reconstruction later
            gt_img_save_path = self.tmpdir / rgb_filename.parts[-1]
            cv2.imwrite(str(gt_img_save_path), image[..., ::-1])
            image = np.asarray(np.float32(image)) / 255.0
            image = torch.Tensor(image).cuda()
            height, width, _channels = image.shape

            gt_pose = torch.Tensor(self.poses[idx, ...])
            camera = Camera(self.Ks.clone(), height, width)
            frame = Frame(
                image,
                ts,
                camera,
                Pose(),
                gt_pose,
                gt_depth=None,
                img_file=gt_img_save_path,
                index=idx,
            )
            return (SensorTypes.RGB, frame)

        if sensor_type == SensorTypes.Depth:
            depth_filename = self.sequence_dir / self.depth_frame_filenames[idx]
            depth_im = Image.open(depth_filename)
            depth_image = np.asarray(depth_im)
            x, y, w, h = self.roi
            depth_image = depth_image[y : y + h, x : x + w]
            depth_image = torch.Tensor(depth_image.copy()).cuda() / 5000.0
            height, width = depth_image.shape

            gt_pose = torch.Tensor(self.poses[idx, ...])
            camera = Camera(self.Ks.clone(), height, width)
            frame = DepthFrame(
                depth_image,
                camera,
                ts,
                index=idx,
            )
            return (SensorTypes.Depth, frame)

        if sensor_type == SensorTypes.IMU:
            accel_val = self.accel_frames[idx]
            frame = IMUFrame(
                accel=torch.Tensor(accel_val).cuda(),
                gyro=None,
                timestamp=ts,
                index=idx,
            )
            return (SensorTypes.IMU, frame)

        assert_never(sensor_type)


class VideoCap:
    def __init__(self, video_file: Path, start: int = 0):
        self.video_file_path = Path(video_file)
        self.start = start

    def init(self):
        self.video_cap = cv2.VideoCapture(self.video_file_path)

        self.length = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(self.length)

        while self.start > 0:
            ret, _frame = self.video_cap.read()
            assert ret, f"video file too short for {self.start=} more frames"
            self.length -= 1
            self.start -= 1

        fx, fy, cx, cy, *d = oakd_intrinsics

        img_size = (3840, 2160)
        K = np.array(
            [
                [
                    fx,
                    0,
                    cx,
                ],
                [
                    0,
                    fy,
                    cy,
                ],
                [
                    0,
                    0,
                    1,
                ],
            ],
            dtype=np.float32,
        )

        self.Ks, self.roi = cv2.getOptimalNewCameraMatrix(
            K,
            np.array(d),
            img_size,
            0,
            img_size,
        )

        self.undistort_map_x, self.undistort_map_y = cv2.initUndistortRectifyMap(
            K,
            np.array(d),
            None,
            self.Ks,
            img_size,
            cv2.CV_32FC1,
        )

        self.Ks = torch.tensor(self.Ks).cuda()

    def __len__(self):
        return self.length

    @torch.no_grad()
    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration
        _ret, im = self.video_cap.read()
        img_size = (3840, 2160)
        im = cv2.resize(im, dsize=img_size)
        image = cv2.remap(
            np.array(im),
            self.undistort_map_x,
            self.undistort_map_y,
            cv2.INTER_LINEAR,
        )
        new_size = (480, 270)
        x, y, w, h = self.roi
        image = image[y : y + h, x : x + w]
        image = cv2.resize(image, dsize=new_size)
        image = np.asarray(np.float32(image)) / 255.0
        image = torch.Tensor(image).cuda()
        # height, width, _channels = image.shape
        # print(f'old {height=}, {width=}')
        # print(f'new height={new_size[1]}, width={new_size[0]}')
        camera = Camera(self.Ks.clone() / 8, new_size[1], new_size[0])
        frame = Frame(
            image,
            time.time(),
            camera,
            Pose(),
            None,
            gt_depth=None,
            img_file=None,
            index=idx,
        )
        return frame


if __name__ == "__main__":
    oakd_sensor = OakdSensor()
    done_event = Event()
    queue = JoinableQueue()
    stream = RGBSensorStream(oakd_sensor, queue, done_event)
    stream.run()
    while not queue.empty():
        print(queue.get())
        queue.task_done()
    # stream.join()
