from torch import multiprocessing as mp
import tyro

from gslam.backend import Backend, MapConfig
from gslam.data import RGBSensorStream, TumRGB, Replica
from gslam.frontend import Frontend, TrackingConfig

from dataclasses import dataclass, field
from datetime import datetime
import os
from pathlib import Path
import sys

from typing import Literal


@dataclass
class PipelineConfig:
    scene: Path
    m: MapConfig = field(default_factory=lambda: MapConfig())
    t: TrackingConfig = field(default_factory=lambda: TrackingConfig())
    seq_len: int = -1
    run_name: str = ''
    dataset: Literal["tum", "replica"] = "tum"


def main(conf: PipelineConfig):
    if conf.dataset == "tum":
        dataset = TumRGB(conf.scene, conf.seq_len)
    elif conf.dataset == "replica":
        dataset = Replica(conf.scene, conf.seq_len)

    dataset_queue = mp.Queue()
    frontend_to_backend_queue = mp.Queue()
    backend_to_frontend_queue = mp.Queue()

    frontend_done_event = mp.Event()
    backend_done_event = mp.Event()
    global_pause_event = mp.Event()

    sensor_stream_process = RGBSensorStream(dataset, dataset_queue, frontend_done_event)

    runs_dir = Path('runs')
    run_name = conf.run_name
    if run_name == '':
        run_name = datetime.now().strftime('%Y-%m-%d--%H-%M-%S')
    output_dir = runs_dir / run_name
    os.makedirs(output_dir, exist_ok=True)

    with open(output_dir / 'args.txt', 'w') as f:
        f.write(' '.join(sys.argv))

    frontend_process = Frontend(
        conf.t,
        frontend_to_backend_queue,
        backend_to_frontend_queue,
        dataset_queue,
        frontend_done_event,
        backend_done_event,
        output_dir,
        global_pause_event,
    )

    backend_process = Backend(
        conf.m,
        frontend_to_backend_queue,
        backend_to_frontend_queue,
        backend_done_event,
        global_pause_event,
    )

    sensor_stream_process.start()
    backend_process.start()
    frontend_process.start()

    frontend_process.join()
    backend_process.join()
    sensor_stream_process.join()


if __name__ == '__main__':
    mp.set_start_method('spawn')
    conf = tyro.cli(PipelineConfig)
    main(conf)
