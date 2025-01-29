#!/usr/bin/env python3

from gslam.warp import Warp
from gslam.data import TumRGB
import time
import torch

device = 'cuda'
dataset = TumRGB('/mnt/data/datasets/rgbd_dataset_freiburg1_desk', scale=1.0)
f1 = dataset[0].to(device)
f2 = dataset[10].to(device)

warp_module = Warp(f1.camera.intrinsics, f1.camera.height, f1.camera.width)

warp_fns = {
    'warp_module': warp_module,
    'warp_module_traced': torch.jit.trace_module(
        warp_module,
        {
            'forward': (
                f1.pose(),
                f2.pose(),
                f1.camera.intrinsics,
                f1.img,
                f1.gt_depth,
            ),
        },
        check_trace=False,
    ),
}

if device == 'cuda':
    warp_fns['warp_module_compiled'] = torch.compile(warp_module, backend='cudagraphs')


def profile_warn_fn(warp_fn_name, warp_fn, num_runs=50):
    for _ in range(num_runs):
        torch.compiler.cudagraph_mark_step_begin()
        _result, _normalized_warps, _keep_mask = warp_fn(
            f1.pose(),
            f2.pose(),
            f1.camera.intrinsics,
            f1.img,
            f1.gt_depth,
        )

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_runs):
        torch.compiler.cudagraph_mark_step_begin()
        _result, _normalized_warps, _keep_mask = warp_fn(
            f1.pose(),
            f2.pose(),
            f1.camera.intrinsics,
            f1.img,
            f1.gt_depth,
        )

    torch.cuda.synchronize()
    end = time.time()
    print(f'{warp_fn_name:<25} {num_runs / (end - start)} fps')


for warp_fn_name, warp_fn in warp_fns.items():
    profile_warn_fn(warp_fn_name, warp_fn)
