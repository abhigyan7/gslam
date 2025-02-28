#!/usr/bin/env python3

from collections import defaultdict

import torch
import pypose as pp

from .primitives import Frame


def add_constraint(pose_graph: defaultdict[int, set], kf1: int, kf2: int):
    pose_graph[kf1].add(kf2)
    pose_graph[kf2].add(kf1)
    return pose_graph


def remove_keyframe(pose_graph: defaultdict[int, set], kf_id: int):
    del pose_graph[kf_id]
    for kf in pose_graph:
        pose_graph[kf].discard(kf_id)
    return pose_graph


class PoseGraphEdge:
    def __init__(
        self,
        idx1: int,
        idx2: int,
        transform: pp.LieTensor,
        uncertainty: torch.Tensor = None,
    ):
        self.idx1 = idx1
        self.idx2 = idx2
        self.T = transform
        self.uncertainty = uncertainty


class PoseGraph(torch.nn.Module):
    def __init__(self, bspline_order: int = 2):
        super().__init__()
        self.bspline_order = bspline_order
        self.coeffs = None
        self.dot_coeffs = None
        self.dotdot_coeffs = None
        self.knots = []  # make this a parameterlist
        self.keyframes = dict()
        self.timestamps: dict[int, float] = []

    def forward(self, t) -> pp.LieTensor:
        # TODO implement bspline interpolation
        return None

    def dot(self, t) -> pp.LieTensor:
        # TODO implement bspline derivative
        return None

    def dotdot(self, t) -> pp.LieTensor:
        # TODO implement bspline double derivative
        return None

    def add_edge(self, edge: PoseGraphEdge):
        idx1, idx2 = sorted(edge.idx1, edge.idx2)
        edge.idx1 = idx1
        edge.idx2 = idx2
        self.edges[(idx1, idx2)] = edge
        return

    def add_keyframe(self, kf: Frame):
        self.keyframes[kf.idx] = kf
        self.knots[kf.idx] = kf.pose

    def pose_graph_residual(self) -> torch.Tensor:
        # TODO implement the pypose example thing
        return None
