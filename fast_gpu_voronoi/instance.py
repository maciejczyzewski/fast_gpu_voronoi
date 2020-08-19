import numpy as np
import numba as nb
from numba import njit, prange
from .bench import bench
from .debug import save
from pprint import pprint

@njit(parallel=True, fastmath=True)
def fill_mat2d(M : np.ndarray, pts, ids) -> np.ndarray:
    for i in prange(0, len(ids)):
        x, y = pts[i]
        M[x, y] = [ids[i], x, y]
    return M

class EmptyInstanceExpections(Exception):
    pass

class Instance:
    shape = None
    points, seeds = [], []

    mat2d = np.array([])  # format(seed, pos_x, pos_y) | [x, y, 3]

    # FIXME: sort points? by odleglosc pomiedzy soba (dist sort)
    def __init__(self, points=None, seeds=None, shape=None):
        if points is None:
            raise EmptyInstanceExpections()
        self.points = np.array(points)  # FIXME: convert to numpy?
        if seeds is None:
            # FIXME: what if len(points) is wrong type?
            self.seeds = np.arange(
                1, len(self.points) + 1, 1).astype(np.uint32)
        else:
            self.seeds = np.array(seeds)
        if shape is None:
            shape = (self.points[:, 0].max() + 1, self.points[:, 1].max() + 1)
        self.shape = shape

        # self.reset()

        print(
            f"-------> SHAPE={self.shape} SEEDS={len(self.seeds)}")

        # save(self.mat2d[:, :, 0])

    def reset(self):
        self.mat2d = np.zeros((self.shape[0], self.shape[1], 3), dtype=np.uint32)
        self.mat2d = fill_mat2d(self.mat2d, self.points, self.seeds)
