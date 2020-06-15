from fast_gpu_voronoi import TestInstance
from fast_gpu_voronoi.jfa import JFA_test, Brute
from fast_gpu_voronoi.utils import do_sample

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from collections import defaultdict
from tqdm import tqdm

options = [["method", "square", "circle", "random_circle", "regular_circle"], ["step_size", 2, 3, 4, 5], ["step_as_power", True, False], ["noise", True, False]]

I = None
Iexact = None

err = None
ti = None

def arr():
    return []
total_err = defaultdict(arr)
total_time = defaultdict(arr)

err_loc = "err.pickle"
time_loc = "time.pickle"

def computeError(a, b):
    return np.count_nonzero(a!=b)/a.size

def rec(d, dict):
    if d == len(options):
        alg = dict.__str__()
        t = I.run(**dict)
        ti[alg]+= t
        err[alg]+= computeError(I.M, Iexact.M)
    else:
        for i in range(1, len(options[d])):
            dict[options[d][0]] = options[d][i]
            rec(d+1, dict)

def test(x, y, n):
    global I, Iexact, err, ti
    _, _, pts, _ = do_sample(x, y, n)
    I = TestInstance(alg=JFA_test, x=x, y=y, pts=pts)
    Iexact = TestInstance(alg=Brute, x=x, y=y, pts=pts)
    Iexact.run()
    # rec(0, {})
    # rec(4, {'method': 'square', 'step_size': 2, 'step_as_power': True, 'noise': False})
    # rec(4, {'method': 'square', 'step_size': 2, 'step_as_power': True, 'noise': True})
    # # rec(4, {'method': 'circle', 'step_size': 3, 'step_as_power': True, 'noise': False})
    # # rec(4, {'method': 'circle', 'step_size': 3, 'step_as_power': True, 'noise': True})
    # # rec(4, {'method': 'regular_circle', 'step_size': 3, 'step_as_power': True, 'noise': False})
    # # rec(4, {'method': 'regular_circle', 'step_size': 3, 'step_as_power': True, 'noise': True})
    # rec(4, {'method': 'circle', 'step_size': 2, 'step_as_power': True, 'noise': False})
    # rec(4, {'method': 'circle', 'step_size': 2, 'step_as_power': True, 'noise': True})
    # # rec(4, {'method': 'circle', 'step_size': 2, 'step_as_power': True, 'noise': False})
    # # rec(4, {'method': 'circle', 'step_size': 2, 'step_as_power': True, 'noise': True})
    # rec(4, {'method': 'regular_circle', 'step_size': 2, 'step_as_power': True, 'noise': False})
    # rec(4, {'method': 'regular_circle', 'step_size': 2, 'step_as_power': True, 'noise': True})
    
    rec(4, {'method': 'square', 'step_size': 4, 'step_as_power': True, 'noise': False})
    rec(4, {'method': 'square', 'step_size': 4, 'step_as_power': True, 'noise': True})
    rec(4, {'method': 'circle', 'step_size': 4, 'step_as_power': True, 'noise': False})
    rec(4, {'method': 'circle', 'step_size': 4, 'step_as_power': True, 'noise': True})
    rec(4, {'method': 'regular_circle', 'step_size': 4, 'step_as_power': True, 'noise': False})
    rec(4, {'method': 'regular_circle', 'step_size': 4, 'step_as_power': True, 'noise': True})


def append_dict_to_dict(d, total):
    for key, value in d.items():
        total[key].append(value/numberOfTests)

def draw_dict(total, ax):
    for key, value in total.items():
        ax.plot(seeds, value, label=key)

seeds = [8, 64, 256, 1024]
numberOfTests = 100

if False and os.path.isfile(err_loc) and os.path.isfile(time_loc):
    with open(err_loc, "rb") as f:
        total_err = pickle.load(f)
    with open(time_loc, "rb") as f:
        total_time = pickle.load(f)
else:
    for seed in seeds:
        err = defaultdict(lambda: 0)
        ti = defaultdict(lambda: 0)
        for t in tqdm(range(numberOfTests)):
            test(100, 100, seed)
        append_dict_to_dict(err, total_err)
        append_dict_to_dict(ti, total_time)
    with open(err_loc, "wb") as f:
        pickle.dump(total_err, f)
    with open(time_loc, "wb") as f:
        pickle.dump(total_time, f)

fig = plt.figure(figsize=(20, 10))
ax = fig.subplots(1, 2)
ax[0].set_title("Error")
ax[1].set_title("Time")
draw_dict(total_err, ax[0])
draw_dict(total_time, ax[1])
plt.legend()
plt.savefig("res.pdf")