#!/usr/bin/env python3

from collections import defaultdict


def add_constraint(pose_graph: defaultdict[int, set], kf1: int, kf2: int):
    pose_graph[kf1].add(kf2)
    pose_graph[kf2].add(kf1)
    return pose_graph


def remove_keyframe(pose_graph: defaultdict[int, set], kf_id: int):
    del pose_graph[kf_id]
    for kf in pose_graph:
        pose_graph[kf].discard(kf_id)
    return pose_graph
