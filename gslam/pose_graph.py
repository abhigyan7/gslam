#!/usr/bin/env python3

from collections import defaultdict


def add_constraint(pose_graph: defaultdict[int, set], kf1: int, kf2: int):
    pose_graph[kf1].add(kf2)
    pose_graph[kf2].add(kf1)
    return pose_graph
