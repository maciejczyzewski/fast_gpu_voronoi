import os
import math
import pyopencl as cl
import random
from contextlib import contextmanager
from pprint import pprint
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numba as nb
import numpy as np
from numba import njit, prange
from PIL import Image
from tqdm import tqdm

from . import __version__
from .debug import save
from .gen import gen_uniform
from .bench import bench
from .instance import Instance

try:
    import sys
    import IPython.core.ultratb

    sys.excepthook = IPython.core.ultratb.ColorTB()
except BaseException:
    pass

def run():
    import pathlib
    path = pathlib.Path().absolute()
    print(path)

    print("\n=== [START DEVELOP] ===\n")

    SHAPE = (512, 512)
    DENSITY = 0.01  # 0.01

    with bench("gen_uniform"):
        points, seeds = gen_uniform(shape=SHAPE, num=SHAPE[0] * SHAPE[1] *
                                    DENSITY)
        #points, seeds = gen_uniform(shape=SHAPE, num=300)

    x = Instance(points=points, shape=SHAPE)
    print(x.seeds)
