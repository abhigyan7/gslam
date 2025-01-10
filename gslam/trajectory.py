import numpy as np
from typing import Tuple


def kabsch_umeyama(
    A: np.ndarray, B: np.ndarray
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    implementation from https://zpl.fi/aligning-point-patterns-with-kabsch-umeyama-algorithm/
    B' = t + c * R @ b gives b in A's frame
    """
    assert A.shape == B.shape
    n, m = A.shape

    EA = np.mean(A, axis=0)
    EB = np.mean(B, axis=0)
    VarA = np.mean(np.linalg.norm(A - EA, axis=1) ** 2)

    H = ((A - EA).T @ (B - EB)) / n
    U, D, VT = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U) * np.linalg.det(VT))
    S = np.diag([1] * (m - 1) + [d])

    R = U @ S @ VT
    c = VarA / np.trace(np.diag(D) @ S)
    t = EA - c * R @ EB

    return R, c, t


def average_translation_error(A: np.ndarray, B: np.ndarray) -> float:
    R, c, t = kabsch_umeyama(A, B)
    aligned = np.array([t + c * R @ b for b in B])
    error = aligned - A
    ate = np.sqrt(np.mean(np.sum(np.multiply(error, error), -1)))
    return ate
