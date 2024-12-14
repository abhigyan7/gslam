from torch import multiprocessing as mp

from gslam.backend import Backend
from gslam.data import RGBSensorStream, TumRGB
from gslam.frontend import Frontend, TrackingConfig
from gslam.map import MapConfig
from gslam.rasterization import RasterizerConfig


def main():
    tum_dataset = TumRGB('../datasets/tum/rgbd_dataset_freiburg1_desk')

    dataset_queue = mp.JoinableQueue()
    frontend_to_backend_queue = mp.JoinableQueue()
    backend_to_frontend_queue = mp.JoinableQueue()
    sensor_stream_process = RGBSensorStream(tum_dataset, dataset_queue)
    frontend_process = Frontend(
        TrackingConfig(),
        RasterizerConfig(),
        frontend_to_backend_queue,
        backend_to_frontend_queue,
        dataset_queue,
    )

    backend_process = Backend(
        MapConfig(),
        frontend_to_backend_queue,
        backend_to_frontend_queue,
    )

    sensor_stream_process.start()
    backend_process.start()
    frontend_process.start()

    frontend_process.join()


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
