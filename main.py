import os
import gc
import sys
import math
import subprocess
import numpy as np
import numba as nb
import pyopencl as cl
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from pprint import pprint
from timeit import default_timer as timer

from skopt import gp_minimize, forest_minimize
from skopt.utils import use_named_args
from skopt.plots import plot_objective, plot_evaluations, plot_convergence, plot_regret
from skopt.space import Categorical, Integer, Real

try:
    import sys
    import IPython.core.ultratb

    sys.excepthook = IPython.core.ultratb.ColorTB()
except BaseException:
    pass

print("[MAIN]")

oo = 16776832
mf = cl.mem_flags

class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)

def call(cmd):
    print(
        f"\x1b[7;37;40m{cmd}\x1b[0m\n" + subprocess.check_output(
            cmd, stderr=subprocess.STDOUT, shell=True).decode("utf-8"),
        end="",
    )

np.random.seed(+oo)

# FIXME: seed everything (but how OpenCl?)

################################################################################

class Config:
    pass

DEBUG = {
    "iter": 0,
    "name": "debug"
}

SESSION = {
    "ctx": None,
    "queue": None,
}

# FIXME: niech sam znajduje najlepsze GPU
SESSION["ctx"] = cl.create_some_context()
SESSION["queue"] = cl.CommandQueue(SESSION["ctx"])

call("rm -f __*.png")

################################################################################

# FIXME: wiecej roznych przykladow
# np. https://github.com/bartwronski/PoissonSamplingGenerator/blob/master/poisson.py
@nb.njit(parallel=True, fastmath=True)
def gen_uniform(shape, num=0):
    """losowe punkty na 2d plaszczyznie"""
    dhash = nb.typed.Dict.empty(
        key_type=nb.types.int64, value_type=nb.types.b1,)
    pts = np.empty((int(num) + 1, 2), np.int32)
    C, E, i = max(shape), int(num) + 1, 0
    while E > 0:
        for _ in range(E):
            x = np.random.randint(0, shape[0])
            y = np.random.randint(0, shape[1])
            checksum = x * C + y
            if checksum not in dhash:
                pts[i] = x, y
                dhash[checksum] = True
                i += 1
                E -= 1
    return pts[0: int(num)], np.arange(1, int(num) + 1, 1).astype(np.uint32)

@nb.njit(parallel=True, fastmath=True)
def mat2d_to_rgb(img, colors=256):
    dcolors = []
    for i in range(0, colors):
        h = i * 2147483647 + 131071
        dcolors.append([h % 255, (h % 8191) % 255, (h % 524287) % 255])
    img = np.expand_dims(img, axis=2)
    img = np.concatenate((img, np.zeros((img.shape[0], img.shape[1], 2))), axis=2)
    for x in nb.prange(0, img.shape[0]):
        for y in nb.prange(0, img.shape[1]):
            img[x, y] = dcolors[int(img[x, y][0]) % colors]
    return img

@nb.njit(parallel=True, fastmath=True)
def fill_mat2d(M : np.ndarray, pts, ids) -> np.ndarray:
    for i in nb.prange(0, len(ids)):
        x, y = pts[i]
        M[x, y] = [ids[i], x, y]
    return M

def to_mat2d(sample):
    mat2d = np.zeros((sample["shape"][0], sample["shape"][1], 3), dtype=np.uint32)
    return fill_mat2d(mat2d, sample["points"], sample["seeds"])

def to_scipy():
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Voronoi.html
    pass # FIXME: use GPU for this step?
    return Bunch({
        "vertices": None,
        "regions": None,
        "ridge_vertices": None,
        "ridge_points": None
    })

def create_instance(func, shape=None, num=None):
    points, seeds = func(shape, num=num)
    return {
        "shape": shape,
        "points": points,
        "seeds": seeds,
    }

################################################################################

def save(M, prefix=""):
    print("======== \033[92mSAVE {}\033[m ========".format(DEBUG["iter"]))

    DEBUG["iter"] += 1
    if prefix != "":
        DEBUG["name"] += "_" + prefix
    name = "__{}_{}.png".format(DEBUG["iter"], DEBUG["name"])

    _M_color = mat2d_to_rgb(M)
    im = Image.fromarray(np.uint8(_M_color), "RGB")
    im = im.resize((512, 512), Image.NEAREST)  # FIXME: option?
    im.save(name, "PNG")

# FIXME: add plot and simple table?

################################################################################

def placebo(x):
    return x

def load_prg(name):
    name = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "", name)
    with open(name) as f:
        prg = cl.Program(SESSION["ctx"], f.read()).build()
    return prg

def load_prg_from_str(code):
    return cl.Program(SESSION["ctx"], code).build()

def copy_to_host(ptr, out, shape=None):
    mat2d_new = np.empty_like(ptr)
    event = cl.enqueue_copy(SESSION["queue"], mat2d_new, out)
    event.wait()
    return mat2d_new.reshape(shape[0], shape[1], 3)

# FIXME: add modifiers like `noise kernel`

def step_function_default(shape, num=None):
    steps = []
    for factor in range(1, +oo, 1):
        f = math.ceil(max(shape) / (2**(factor)))
        steps.append(f)
        if f <= 1:
            break
    return steps

class Vorotron():
    name = "VOROTRON"
    
    config = {
        "brutforce": True,
        "step_function": step_function_default,
        "anchor_type": placebo,
        "noise": False,
    }

    def __init__(self, config=None):
        if config:
            self.config = dict(list(self.config.items()) + list(config.items())) 
        self.kernel_brutforce = load_prg("algo_brutforce.cl")
        self.kernel_noise = load_prg("algo_noise.cl")
        pprint(self.config)

        # FIXME: apply
        if not self.config["brutforce"]:
            code = open("algo_template.cl", 'r').read()
            for key in ["anchor_type"]:
                code = self.config[key](code, self.config)
            self.kernel = load_prg_from_str(code)

    def do(self, x):
        if "__mat2d" not in x:
            x["__mat2d"] = to_mat2d(x)

        mat2d_flatten = x["__mat2d"].flatten().astype(np.uint32)

        T1_all, T_SUM = timer(), 0
        
        # === INPUT ===
        if self.config["brutforce"]:
            points_in = cl.Buffer(SESSION["ctx"], mf.COPY_HOST_PTR |
                                  mf.READ_ONLY, hostbuf=x["points"])
            seeds_in = cl.Buffer(SESSION["ctx"], mf.COPY_HOST_PTR |
                                 mf.READ_ONLY, hostbuf=x["seeds"])
        else:
            mat2d_in = cl.Buffer(SESSION["ctx"], mf.COPY_HOST_PTR | mf.READ_WRITE,
                                 hostbuf=mat2d_flatten)

        # === OUTPUT ===
        mat2d_out = cl.Buffer(SESSION["ctx"], mf.READ_WRITE, mat2d_flatten.nbytes)


        # === NOISE ===
        if self.config["noise"]:
            points_in = cl.Buffer(SESSION["ctx"], mf.COPY_HOST_PTR |
                                  mf.READ_ONLY, hostbuf=x["points"])
            seeds_in = cl.Buffer(SESSION["ctx"], mf.COPY_HOST_PTR |
                                 mf.READ_ONLY, hostbuf=x["seeds"])

            # === RUN ===
            T1 = timer()
            event = self.kernel_noise.fn(
                                 SESSION["queue"],
                                 (x["shape"][0], x["shape"][1], 1), None,
                                 points_in,
                                 seeds_in,
                                 mat2d_in,
                                 np.int32(x["shape"][0]),
                                 np.int32(x["shape"][1]),
                                 np.int32(x["points"].shape[0]),
                                 )
            event.wait()
            T2 = timer()
            T_SUM += T2-T1

            # DEBUG -----------------
            # x["__mat2d"] = copy_to_host(mat2d_flatten, mat2d_in, x["shape"])
            # save(x["__mat2d"][:, :, 0])
            # -----------------------

        # === RUN ===
        if self.config["brutforce"]:
            T1 = timer()
            event = self.kernel_brutforce.fn(
                           SESSION["queue"],
                           (x["shape"][0], x["shape"][1], 1),
                           None, # default?
                           points_in,
                           seeds_in,
                           mat2d_out,
                           np.int32(x["shape"][0]),
                           np.int32(x["shape"][1]),
                           np.int32(x["points"].shape[0]),
                           )
            event.wait()
            T2 = timer()
            T_SUM = T2-T1
        else:
            steps = self.config["step_function"](x["shape"], x["points"].shape[0])

            for step in steps:
                T1 = timer()
                event = self.kernel.fn(
                               SESSION["queue"],
                               (x["shape"][0], x["shape"][1], 1),
                               None,
                               mat2d_in,
                               mat2d_out,
                               np.int32(x["shape"][0]),
                               np.int32(x["shape"][1]),
                               np.int32(step)
                               )
                event.wait()
                T2 = timer()
                T_SUM += T2-T1
                mat2d_in, mat2d_out = mat2d_out, mat2d_in
            mat2d_in, mat2d_out = mat2d_out, mat2d_in

        T2_all = timer()
        x["__mat2d"] = copy_to_host(mat2d_flatten, mat2d_out, x["shape"])
        return x, T_SUM, T2_all-T1_all

################################################################################

def use_num(shape, x): return x
def use_density(shape, x): return int(shape[0] * shape[1] * x)

def fn_loss(m1, m2):
    return 1 - np.count_nonzero((m1-m2) == 0)/(m1.shape[0]*m1.shape[1])

def valid(model1, model2, x, n=4):
    loss_arr, time_1_arr, time_2_arr = [], [], []

    # FIXME: jesli nie ma duzych roznic przerwij
    for _ in range(n):
        # brutforce
        if "__mat2d" in x:
            del x["__mat2d"]
        m1, t1, t1_all = model1.do(x)
        m1 = m1["__mat2d"][:, :, 0]
        # algorithm
        if "__mat2d" in x:
            del x["__mat2d"]
        m2, t2, t2_all = model2.do(x)
        m2 = m2["__mat2d"][:, :, 0]
        # error
        loss = fn_loss(m1, m2)
        # append
        loss_arr.append(loss)
        time_1_arr.append(t1)
        time_2_arr.append(t2)

    # FIXME: tylko debug
    save(m2)

    def avgarr(arr):
        shift = max(int(len(arr)*(0.1)), 1)
        arr = sorted(arr)[shift:-shift]
        return round(sum(arr)/len(arr), 6)

    loss = avgarr(loss_arr)*100
    time_1 = avgarr(time_1_arr)
    time_2 = avgarr(time_2_arr)
    
    loss_diff = round(loss, 6)
    time_diff = round(time_1 / time_2, 6)
    print(f"\033[91m [[ {model2.name.ljust(20)} ]] =========> loss: {loss_diff:8}% | \033[94m time: {time_diff:6}\033[0m")

    gc.collect()
    return loss_diff, time_diff

def optimize(space, domain, n_calls=10):
    @use_named_args(space)
    def score(**params):
        print("======= EXPERIMENT =======")
        # FIXME: dla roznych domen rozne wyniki?
        final_score = do_compirason(params, domain)
        print("==========================")
        return -final_score

    return gp_minimize(func=score, dimensions=space, n_calls=n_calls)

def do_compirason(config=None, domain=None):
    config["brutforce"] = False
    model = Vorotron(config)
    model_brutforce = Vorotron()

    loss_arr = []

    for shape in domain["shapes"]:
        for case in domain["cases"]:
            print("\n")
            func, params = list(case.items())[0]
            func_anchor, arg = params
            num = max(1, func_anchor(shape, arg))

            sample = create_instance(func, shape=shape, num=num)

            a, b = valid(model_brutforce, model, sample)
            # FIXME: wzor jakis na {a,b}?
            loss_arr.append(b/(1+a))
            print(f"shape={shape} | a={a} b={b}")

    score = sum(loss_arr)/len(loss_arr)
    print(f"----> SCORE={score}")

    # FIXME: sami musimy sobie zapisywac jakie parametry jaki daly wynik
    return score

################################################################################

def do_test_gen():
    for _ in tqdm(range(10)):
        sample = create_instance(gen_uniform, shape=(100, 100), num=50)    
        M = to_mat2d(sample)
        save(M[:, :, 0])

def do_test_simple():
    config = {
        "brutforce": False,
        "step_function": step_function_default,
        "noise": True, 
    }
    sample = create_instance(gen_uniform, shape=(32, 32), num=8)
    algo = Vorotron(config)
    X, T1, T2 = algo.do(sample)
    save(X["__mat2d"][:, :, 0])

TEST_DOMAIN = {
    "shapes":
        [(128, 128), (512, 512)],
    "cases":
        [
            {gen_uniform: [use_num, 1]},
            {gen_uniform: [use_num, 2]},
            {gen_uniform: [use_density, 0.01]},
            {gen_uniform: [use_density, 0.05]},
        ]
}

def do_test_error():
    # FIXME: plot?
    model = Vorotron({"brutforce": False})
    model_brutforce = Vorotron()

    for shape in TEST_DOMAIN["shapes"]:
        for case in TEST_DOMAIN["cases"]:
            print("\n")
            func, params = list(case.items())[0]
            func_anchor, arg = params
            num = max(1, func_anchor(shape, arg))
        
            sample = create_instance(func, shape=shape, num=num)

            a, b = valid(model_brutforce, model, sample)
            print(f"shape={shape} | a={a} b={b}")

################################################################################

def step_function_star(shape, num):
    def LogStar(n):
        if n <= 1:
            return 0
        if 1 < n and n <= 8:
            return 1
        if 8 < n and n <= 64:
            return 2
        if 64 < n and n <= 1024:
            return 3
        if 1024 < n and n <= 65536:
            return 4
        if 65536 < n:
            return 5
    n = LogStar(num)
    steps, i = [], 0 # FIXME: enum
    for factor in range(1, +oo, 1):
        i += 1
        f = math.ceil(max(shape) / (3**(factor)))
        steps.append(f)
        if f <= 1 or i == n + 1:
            break
    return steps

def step_function_factor3(shape, num=None):
    steps = []
    for factor in range(1, +oo, 1):
        f = math.ceil(max(shape) / (3**(factor)))
        steps.append(f)
        if f <= 1:
            break
    return steps

# FIXME: modifiers here? to code? recurse?

def mod_anchor_type__square(code, config=None):
    if config["anchor_double"]:
        code = code.replace("#{ANCHOR_TYPE}", """
        int pos[] = {-step, -step/2, 0, step/2, step};
        for(int i = 0; i < 5; i++)
            for(int j = 0; j < 5; j++) {
                int nx = x + pos[i];
                int ny = y + pos[j];
        """)
    else:
        code = code.replace("#{ANCHOR_TYPE}", """
        int pos[] = {-step, 0, step};
        for(int i = 0; i < 3; i++)
            for(int j = 0; j < 3; j++) {
                int nx = x + pos[i];
                int ny = y + pos[j];
        """)
    return code

# FIXME: 12 jako parametr
def mod_anchor_type__circle(code, config=None):
    anchor_num = config["anchor_num"]
    if config["anchor_double"]:
        code = code.replace("#{ANCHOR_TYPE}", """
        for(int j = 0; j < 2; j++)
        for(int i = 0; i < #{ANCHOR_NUM}; i++) {
            int A = (step * cos((float) ((6.28/#{ANCHOR_NUM}) * i) ));
            int B = (step * sin((float) ((6.28/#{ANCHOR_NUM}) * i) ));
            if (j == 0) {
               A /= 2;
               B /= 2;
            }
            int nx = x+A;
            int ny = y+B;
        """.replace("#{ANCHOR_NUM}", str(anchor_num//2)))
    else:
        code = code.replace("#{ANCHOR_TYPE}", """
        for(int i = 0; i < #{ANCHOR_NUM}; i++) {
            int A = (step * cos((float) ((6.28/#{ANCHOR_NUM}) * i) ));
            int B = (step * sin((float) ((6.28/#{ANCHOR_NUM}) * i) ));
            int nx = x+A;
            int ny = y+B;
        """.replace("#{ANCHOR_NUM}", str(anchor_num)))
    return code

SPACE = [
    #Real(0, 1, name='a'),
    #Real(0.25, 1, name='b'),
    #Real(0, 1, name='c'),

    Integer(6, 12+6, name='anchor_num'),
    Categorical([False, True], name='noise'),
    Categorical([False, True], name='anchor_double'),
    Categorical([step_function_default,
                 step_function_star,
                 step_function_factor3],
                name='step_function'),
    Categorical([mod_anchor_type__square,
                 mod_anchor_type__circle],
                name='anchor_type'),
]

# FIXME: WYKRESY BO NIC NIE WIDAC KOTELY!!!!!!!!!!!!!!!!!!!!

DOMAIN = {
    "shapes":
        [(128, 128), (512, 512)],
    "cases":
        [
            {gen_uniform: [use_num, 1]},
            {gen_uniform: [use_num, 2]},
            {gen_uniform: [use_density, 0.001]},
            {gen_uniform: [use_density, 0.01]},
            {gen_uniform: [use_density, 0.05]},
            {gen_uniform: [use_density, 0.1]},
        ]
}

################################################################################

if __name__ == "__main__":
    # do_test_gen()
    # do_test_simple()
    # sys.exit()
    # do_test_search()
    
    opt_result = optimize(
        SPACE,
        DOMAIN,
        n_calls=10
    )

    _ = plot_objective(opt_result, n_points=10)
    # FIXME: nabrawic te marginy
    plt.savefig('raport.png')

    print("ok")
