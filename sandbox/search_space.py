# PYOPENCL_CTX='0:1' poetry run python3 sandbox/search_space.py

import faulthandler
faulthandler.enable()

import os
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '0'

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

from fast_gpu_voronoi import __version__
from fast_gpu_voronoi.debug import save
from fast_gpu_voronoi.gen import gen_uniform
from fast_gpu_voronoi.bench import bench
from fast_gpu_voronoi.instance import Instance

try:
    import sys
    import IPython.core.ultratb

    sys.excepthook = IPython.core.ultratb.ColorTB()
except BaseException:
    pass

################################################################################

# FIXME: `warmup` --> aby szybciej dzialalo
# FIXME: scale? --> possibility to scale? matrix?
# FIXME: batch mode?
# https://gist.github.com/hellerbarde/2843375
# home of automl graphics research // 3d mesh / evrything faster
# Airspeed Velocity
# Mayavi 3d: https://docs.enthought.com/mayavi/mayavi/auto/examples.html

################################################################################

from fast_gpu_voronoi.ref import Algorithm_Brutforce, Algorithm_JFA, Algorithm_JFA_Fusion

from fast_gpu_voronoi import MEMORY
mf = cl.mem_flags
oo = 16776832

def load_prg(name):
    name = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "", name)
    with open(name) as f:
        prg = cl.Program(MEMORY.ctx, f.read()).build()
    return prg

def load_prg_from_str(txt):
    return cl.Program(MEMORY.ctx, txt).build()

# FIXME: ????? send estimation of number of seeds?
# -----------------------> ???????????????????????


def step_function_1(shape):
    steps = []
    for factor in range(1, +oo, 1):
        f = math.ceil(max(shape) / (2**(factor)))
        steps.append(f)
        if f <= 1:
            break
    return steps


def step_function_2(shape):
    steps = []
    for factor in range(1, +oo, 1):
        f = math.ceil(max(shape) / (3**(factor)))
        steps.append(f)
        if f <= 1:
            break
    return steps


def step_function_3(shape):
    steps = []
    for factor in range(1, +oo, 1):
        f = math.ceil(max(shape) / (3**(factor)))
        steps.append(f)
        if f <= 1 or factor >= 3:
            break
    # steps += [1]
    ##########################################
    # FIXME: ((ML HERE)) FROM DENSITY?????????
    ##########################################
    steps = [2, 2, 4, 2, 1]  # FOR DENSE?!
    return steps


def build_kernel(config, template="jfa_template.cl"):
    name = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "", template)
    with open(name) as f:
        XXX = f.read()

    if config['step_extended']:
        # FIXME: code generation?
        v1_i, v1_j = 5, 5
        v1_pos = "{-step, -step/2, 0, step/2, step}"
    else:
        v1_i, v1_j = 3, 3
        v1_pos = "{-step, 0, step}"

    XXX = XXX.replace('@{step_extended[pos]}', str(v1_pos))
    XXX = XXX.replace('@{step_extended[i]}', str(v1_i))
    XXX = XXX.replace('@{step_extended[j]}', str(v1_j))

    return XXX

# config = {
# 'step_extended': True,
# 'step_function': step_function_1,
# }

# code = build_kernel(config, template="jfa_template.cl")
# print(code)
# sys.exit()


class Algorithm_Template():
    name = "TEMPLATE"

    def __init__(self, config=None):
        self.config = config
        pprint(self.config)
        code = build_kernel(self.config, template="jfa_template.cl")
        self.kernel = load_prg_from_str(code)
        self.kernel_noise = load_prg("noise.cl")

    def do(self, x: Instance):
        T1_all = timer()
        mat2d_flatten = x.mat2d.flatten().astype(np.uint32)

        T_SUM = 0

        # === INPUT ===
        mat2d_in = cl.Buffer(MEMORY.ctx, mf.COPY_HOST_PTR | mf.READ_WRITE,
                             hostbuf=mat2d_flatten)

        if self.config["noise"]:
            points_in = cl.Buffer(MEMORY.ctx, mf.COPY_HOST_PTR |
                                  mf.READ_ONLY, hostbuf=x.points)
            seeds_in = cl.Buffer(MEMORY.ctx, mf.COPY_HOST_PTR |
                                 mf.READ_ONLY, hostbuf=x.seeds)

            # === RUN ===
            T1 = timer()
            event = self.kernel_noise.fn(MEMORY.queue, (x.shape[0], x.shape[1], 1), None,
                                 points_in,
                                 seeds_in,
                                 mat2d_in,
                                 np.int32(x.shape[0]),
                                 np.int32(x.shape[1]),
                                 np.int32(x.points.shape[0]),
                                 )
            event.wait()
            T2 = timer()
            T_SUM += T2-T1

            # DEBUG -----------------
            # mat2d_new = np.empty_like(mat2d_flatten)
            # cl.enqueue_copy(MEMORY.queue, mat2d_new, mat2d_in)
            # x.mat2d = mat2d_new.reshape(x.shape[0], x.shape[1], 3)
            # save(x.mat2d[:, :, 0])
            # -----------------------

        # === OUTPUT ===
        mat2d_out = cl.Buffer(MEMORY.ctx, mf.READ_WRITE, mat2d_flatten.nbytes)

        # === RUN ===
        steps = self.config["step_function"](x.shape)

        for step in steps:
            T1 = timer()
            event = self.kernel.fn(MEMORY.queue, (x.shape[0], x.shape[1], 1), None,
                           mat2d_in,
                           mat2d_out,
                           np.int32(x.shape[0]),
                           np.int32(x.shape[1]),
                           np.int32(step)
                           )
            event.wait()
            T2 = timer()
            T_SUM += T2-T1
            mat2d_in, mat2d_out = mat2d_out, mat2d_in
        mat2d_in, mat2d_out = mat2d_out, mat2d_in

        mat2d_new = np.empty_like(mat2d_flatten)
        event = cl.enqueue_copy(MEMORY.queue, mat2d_new, mat2d_out)
        event.wait()
        x.mat2d = mat2d_new.reshape(x.shape[0], x.shape[1], 3)

        # save(x.mat2d[:, :, 0])
        T2_all = timer()
        return x, T_SUM, T2_all-T1_all

################################################################################

import gc

def fn_loss(m1, m2):
    return 1 - np.count_nonzero((m1-m2) == 0)/(m1.shape[0]*m1.shape[1])


model_brutforce = Algorithm_Brutforce()
def valid(x, model, n=10):
    global model_brutforce
    loss_arr, time_arr, time_all_arr = [], [], []
    
    for _ in range(n):
        # brutforce
        # print("\033[95m BEF \033[0m")
        x.reset()
        m1, t1, t1_all = model_brutforce.do(x)
        m1 = m1.mat2d[:, :, 0]
        # algorithm
        x.reset()
        m2, t2, t2_all = model.do(x)
        m2 = m2.mat2d[:, :, 0]
        # error
        loss = fn_loss(m1, m2)
        # print("\033[95m AFF \033[0m")
        # append
        loss_arr.append(loss)
        time_arr.append(t2)
        time_all_arr.append(t2_all)
    
    shift = int(n*(0.1))
    loss_arr = sorted(loss_arr)[shift:-shift]
    time_arr = sorted(time_arr)[shift:-shift]
    time_all_arr = sorted(time_all_arr)[shift:-shift]
    loss = round(sum(loss_arr)/len(loss_arr)*100, 6)
    time = round(sum(time_arr)/len(time_arr), 6)
    time_all = round(sum(time_all_arr)/len(time_all_arr), 6)
    print(f"\033[91m [[ {model.name.ljust(20)} ]] =========> loss: {loss:8}% | time: {time:6}\033[0m | \033[94m time_all: {time_all:6}\033[0m")
    
    gc.collect()
    return loss, time


if __name__ == "__main__":
    """
    import pathlib
    path = pathlib.Path().absolute()
    print(path)
    """

    print("\n=== [START] ===\n")

    model_x = Algorithm_Template({
        'noise':         True,
        'step_extended': False,
        'step_function': step_function_2,
    })
    model_0 = Algorithm_Brutforce()
    model_1 = Algorithm_JFA()
    model_2 = Algorithm_JFA_Fusion()

    #######################################################
    # PLOT??????????????/ MATRIX2d

    def use_num(shape, x): return x
    def use_density(shape, x): return int(shape[0] * shape[1] * x)

    #shape_arr = [(16, 16), (32, 32), (64, 64),
    #             (128, 128), (256, 256), (512, 512)]
    shape_arr = [(128, 128), (512, 512)]
    case_arr = [
        #{gen_uniform: [use_num, 1]},
        #{gen_uniform: [use_num, 2]},
        {gen_uniform: [use_num, 3]},
        #{gen_uniform: [use_density, 0.0001]},
        #{gen_uniform: [use_density, 0.001]},
        {gen_uniform: [use_density, 0.01]},
        {gen_uniform: [use_density, 0.1]},
    ]

    def test_model(model, shape_arr, case_arr):
        for shape in shape_arr:
            for case in case_arr:
                print("\n")
                func, params = list(case.items())[0]
                func_anchor, arg = params
                num = max(1, func_anchor(shape, arg))
                points, seeds = gen_uniform(shape=shape, num=num)
                x = Instance(points=points, seeds=seeds, shape=shape)
                valid(x, model)
                del x, points, seeds
                gc.collect()

    test_model(model_1, shape_arr, case_arr)
    test_model(model_2, shape_arr, case_arr)
    
    #####################################################################
    sys.exit()

    SHAPE = (512, 512)
    DENSITY = 0.01  # 0.01

    with bench("gen_uniform"):
        points, seeds = gen_uniform(shape=SHAPE, num=SHAPE[0] * SHAPE[1] *
                                    DENSITY)
        #points, seeds = gen_uniform(shape=SHAPE, num=300)

    x = Instance(points=points, shape=SHAPE)
    print(x.seeds)

    # FIXME: probalistic programming
    model_x = Algorithm_Template({
        'noise':         True,
        'step_extended': False,
        'step_function': step_function_3,
    })
    valid(x, model_x)

    # model_0 = Algorithm_Brutforce()
    model_1 = Algorithm_JFA()
    model_2 = Algorithm_JFA_Fusion()

    # valid(x, model_0)
    valid(x, model_1)
    valid(x, model_2)
