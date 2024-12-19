from torch import multiprocessing as mp

from gslam.backend import Backend
from gslam.data import RGBSensorStream, TumRGB
from gslam.frontend import Frontend, TrackingConfig
from gslam.map import MapConfig
from gslam.rasterization import RasterizerConfig
from gsplat.strategy import MCMCStrategy

import sys
import gc


# @rr.shutdown_at_exit
def main(seq_len: int = -1):
    tum_dataset = TumRGB('../datasets/tum/rgbd_dataset_freiburg1_desk', seq_len)

    dataset_queue = mp.JoinableQueue()
    frontend_to_backend_queue = mp.JoinableQueue()
    backend_to_frontend_queue = mp.JoinableQueue()

    # rr.init('gslam', recording_id='gslam_1')
    # rr.save('runs/rr.rrd')

    frontend_done_event = mp.Event()
    backend_done_event = mp.Event()

    gc.collect()
    sensor_stream_process = RGBSensorStream(
        tum_dataset, dataset_queue, frontend_done_event
    )

    frontend_process = Frontend(
        TrackingConfig(),
        RasterizerConfig(),
        frontend_to_backend_queue,
        backend_to_frontend_queue,
        dataset_queue,
        frontend_done_event,
        backend_done_event,
    )

    backend_process = Backend(
        MapConfig(),
        frontend_to_backend_queue,
        backend_to_frontend_queue,
        backend_done_event,
        MCMCStrategy(verbose=True),
    )

    sensor_stream_process.start()
    backend_process.start()
    frontend_process.start()

    frontend_process.join()
    backend_process.join()
    sensor_stream_process.join()


if __name__ == '__main__':
    mp.set_start_method('spawn')
    seq_len = -1
    if len(sys.argv) > 1:
        seq_len = int(sys.argv[1])
    main(seq_len)
