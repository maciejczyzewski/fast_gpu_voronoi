# PYOPENCL_CTX='0:1'

import os
import gc
import sys
import math
import json
import subprocess
import numpy as np
import numba as nb
import pyopencl as cl
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from pprint import pprint, pformat
from timeit import default_timer as timer

from skopt import gp_minimize, forest_minimize, dummy_minimize
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
    IS_SPECIAL = False

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
SESSION["queue"] = cl.CommandQueue(SESSION["ctx"],
    properties=cl.command_queue_properties.PROFILING_ENABLE) # FIXME?

if os.path.isdir("results/"):
    print("Wait! Path `results/` already exists ;-)")
    sys.exit()

call("rm -f __*.png")
call("mkdir -p results/")
call("mkdir -p results/log/")
call("mkdir -p results/code/")

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

def save(M, prefix=None):
    print("======== \033[92mSAVE {}\033[m ========".format(DEBUG["iter"]))

    DEBUG["iter"] += 1
    name = DEBUG["name"]
    if prefix != "" and prefix is not None:
        name += "_" + prefix
    name = "__{}_{}.png".format(DEBUG["iter"], name)

    _M_color = mat2d_to_rgb(M)
    im = Image.fromarray(np.uint8(_M_color), "RGB")
    # im = im.resize((512, 512), Image.NEAREST)  # FIXME: option?
    im.save(f"img/{name}", "PNG")

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

def step_function_default(shape, num=None, config=None):
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
        "bruteforce": True,
        "step_function": step_function_default,
        "anchor_type": placebo,
        "noise": False,
        "anchor_distance_ratio": 1/2,
        "anchor_number_ratio": 1/2,
    }

    def __init__(self, config=None):
        if config:
            self.config = dict(list(self.config.items()) + list(config.items())) 
        self.kernel_bruteforce = load_prg("algo_bruteforce.cl")
        self.kernel_noise = load_prg("algo_noise.cl")
        pprint(self.config)

        # FIXME: apply
        if not self.config["bruteforce"]:
            code = open("algo_template.cl", 'r').read()
            for key in ["anchor_type"]:
                code = self.config[key](code, self.config)
            self.kernel = load_prg_from_str(code)
            self.code = code

    def do(self, x):
        if "__mat2d" not in x:
            x["__mat2d"] = to_mat2d(x)

        mat2d_flatten = x["__mat2d"].flatten().astype(np.uint32)

        T_ALL_1, T_CPU, T_GPU = timer(), 0, 0
        
        # === INPUT ===
        if self.config["bruteforce"]:
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
                                 (x["shape"][1], x["shape"][0], 1), None,
                                 points_in,
                                 seeds_in,
                                 mat2d_in,
                                 np.int32(x["shape"][0]),
                                 np.int32(x["shape"][1]),
                                 np.int32(x["points"].shape[0]),
                                 )
            event.wait()
            T_GPU += event.profile.end-event.profile.start
            T2 = timer()
            T_CPU += T2-T1

            # DEBUG -----------------
            # x["__mat2d"] = copy_to_host(mat2d_flatten, mat2d_in, x["shape"])
            # save(x["__mat2d"][:, :, 0])
            # -----------------------

        # === RUN ===
        if self.config["bruteforce"]:
            T1 = timer()
            event = self.kernel_bruteforce.fn(
                           SESSION["queue"],
                           (x["shape"][1], x["shape"][0], 1),
                           None, # default?
                           points_in,
                           seeds_in,
                           mat2d_out,
                           np.int32(x["shape"][0]),
                           np.int32(x["shape"][1]),
                           np.int32(x["points"].shape[0]),
                           )
            event.wait()
            T_GPU += event.profile.end-event.profile.start
            T2 = timer()
            T_CPU = T2-T1
        else:
            steps = self.config["step_function"](x["shape"],
                                                 x["points"].shape[0],
                                                 config=self.config)
            print(steps)

            for step in steps:
                T1 = timer()
                event = self.kernel.fn(
                               SESSION["queue"],
                               (x["shape"][1], x["shape"][0], 1),
                               None,
                               mat2d_in,
                               mat2d_out,
                               np.int32(x["shape"][0]),
                               np.int32(x["shape"][1]),
                               np.int32(step)
                               )
                event.wait()
                T_GPU += event.profile.end-event.profile.start
                T2 = timer()
                T_CPU += T2-T1
                mat2d_in, mat2d_out = mat2d_out, mat2d_in
            mat2d_in, mat2d_out = mat2d_out, mat2d_in

        T_ALL_2 = timer()
        x["__mat2d"] = copy_to_host(mat2d_flatten, mat2d_out, x["shape"])
        # return x, T_SUM, T2_all-T1_all
        return x, T_GPU, T_ALL_2-T_ALL_1, T_CPU

################################################################################

def use_num(shape, x): return x
def use_density(shape, x): return int(shape[0] * shape[1] * x)

def fn_loss(m1, m2):
    return 1 - np.count_nonzero((m1-m2) == 0)/(m1.shape[0]*m1.shape[1])

# FIXME: n=10 IS STABLE!!!!!!!!!!!!!!!!!!!!!! but okay
def valid(model1, model2, x, n=5):
    loss_arr, time_1_arr, time_2_arr = [], [], []

    # FIXME: jesli nie ma duzych roznic przerwij
    for _ in range(n):
        # bruteforce
        if "__mat2d" in x:
            del x["__mat2d"]
        m1, t1, _, _ = model1.do(x)
        m1 = m1["__mat2d"][:, :, 0]
        # algorithm
        if "__mat2d" in x:
            del x["__mat2d"]
        m2, t2, _, _ = model2.do(x)
        m2 = m2["__mat2d"][:, :, 0]
        save(m2)
        # error
        loss = fn_loss(m1, m2)
        # append
        loss_arr.append(loss)
        time_1_arr.append(t1)
        time_2_arr.append(t2)

    # FIXME: tylko debug
    # save(m1, prefix="gt")
    # save(m2, prefix="jfa")

    def avgarr(arr):
        shift = max(int(len(arr)*(0.1)), 1)
        arr = sorted(arr)[shift:-shift]
        return round(sum(arr)/len(arr), 6)

    loss = avgarr(loss_arr)*100
    time_1 = avgarr(time_1_arr)
    time_2 = avgarr(time_2_arr)
    
    loss_diff = round(loss, 6)
    time_diff = round(time_1 / time_2, 6)
    print(f"\033[91m [[ {model2.name.ljust(20)} ]] =========> loss: {loss_diff:8}% | \033[94m time: {time_1/1e9:6} : {time_2/1e9:6}\033[0m")

    gc.collect()
    return loss_diff, time_diff

def human_algo_name(config):
    from fractions import Fraction
    max_denominator = 10 # FIXME: for bigger fractions?
    anchor_type = config["anchor_type"].__name__.split("__")[1].title()
    if config["anchor_double"]:
        anchor_double = "Dual"
        if config['anchor_distance_ratio'] != 0.5:
            anchor_double += "(" + \
                str(Fraction(config['anchor_distance_ratio']) \
                    .limit_denominator(max_denominator)) \
            + ")"
    else:
        anchor_double = ""
    if anchor_type == "Circle":
        anchor_num = str(config["anchor_num"])
        if config['anchor_number_ratio'] != 0.5:
            anchor_num += "(" + \
                str(Fraction(config['anchor_number_ratio']) \
                    .limit_denominator(max_denominator)) \
            + ")"
    else:
        anchor_num = ""
    if config["noise"]:
        noise = "+Noise"
    else:
        noise = ""
    step_function = config["step_function"].__name__.replace("step_function_", "").title()

    ############## SPECIAL
    if "Special" in step_function:
        special = "(" + \
        f"{round(config['A'], 2)}/" + \
        f"{round(config['B'], 2)}/" + \
        f"{round(config['C'], 2)}/" + \
        f"{round(config['D'], 2)}/" + \
        f"{round(config['X'], 2)})"
    else:
        special = ""
    ######################

    return f"{anchor_type}{anchor_num}{anchor_double}|{step_function}{special}{noise}"


# FIXME: progress bar?
def optimize(model_ref, space, domain, n_calls=10):
    ALGOMAP = {}

    def score_from_config(config):
       # FIXME: if same HUMAN NAME----> continue
        
        print("======= EXPERIMENT =======")
        # FIXME: dla roznych domen rozne wyniki?

        config["bruteforce"] = False
        
        name = human_algo_name(config)
        if name in ALGOMAP:
            return ALGOMAP[name]

        model = Vorotron(config)

        score, log = do_compirason(model, model_ref, domain)
        # log["name"] = str(score) # FIXME: human-name        
        # FIXME: human-name        

        path_log = f"results/log/{score}.json"
        path_code = f"results/code/{score}.cl"

        if name == "Square|Default":
            print("=============================> SPECIAL (JFA)")
            path_log = f"results/jfa.json"
            path_code = f"results/jfa.cl"
            name = "JFA (original)"

        log["name"] = f"({int(score)}) " + name

        # FIXME: plot need to import DOMAIN-------->?

        # FIXME: -----> log[name] --> [generate here catchy name]
        pprint(log)
        print(f"[[[[[[[[[[[ \033[96m {log['name']} \033[0m ]]]]]]]]]]]")
        # FIXME: save to log/ then plot.py
        # FIXME: 3d ---> wykres?
        # FIXME: sorted [score] along all axis?
        log_cl = json.dumps(log)
        text_file = open(path_log, "w")
        text_file.write(log_cl)
        text_file.close()

        config_cl = pformat(config, indent=4)
        result_cl = "/*\n" + config_cl + "\n*/\n\n" + model.code
        text_file = open(path_code, "w")
        text_file.write(result_cl)
        text_file.close()

        print("==========================")
        ALGOMAP[name] = -score
        return -score

    @use_named_args(space)
    def score(**config):
        
        print("======= EXPERIMENT =======")
        # FIXME: dla roznych domen rozne wyniki?

        config["bruteforce"] = False
        return score_from_config(config)

    def save_domain(domain):
        domain_vec = []
        for shape in domain["shapes"]:
            for case in domain["cases"]:
                print("\n")
                func, params = list(case.items())[0]
                func_anchor, arg = params
                num = max(1, func_anchor(shape, arg))
                domain_vec.append({"shape": shape, "num": num})

        pprint(domain_vec)

        path_domain = "results/domain.json"
        domain_cl = json.dumps(domain_vec)
        text_file = open(path_domain, "w")
        text_file.write(domain_cl)
        text_file.close()

    save_domain(domain)

    score_from_config({
        'anchor_double': False,
        'anchor_num': None,
        'anchor_type': mod_anchor_type__square,
        'bruteforce': False,
        'noise': False,
        'step_function': step_function_default,
    })

    # [[ gp_minimize vs forest_minimize | dummy_minimize ]]
    if Config.IS_SPECIAL:
        return gp_minimize(func=score, dimensions=space,
                           n_calls=n_calls, random_state=0)
    else:
        return forest_minimize(func=score, dimensions=space,
                           n_calls=n_calls, random_state=0)

def fn_metric(a, b): # b/(1+a)
    # FIXME: max aby w pewnych domenach mogly funkcjonowac
    return max(0, (b * (100-a**1.5)))

def do_compirason(model, model_ref, domain=None): 
    loss_arr = []

    #FIMXE init with keys from domain/shapes
    log = {
        'loss': [],
        'time': [],
        'score': []
    }

    for shape in domain["shapes"]:
        for case in domain["cases"]:
            print("\n")
            func, params = list(case.items())[0]
            func_anchor, arg = params
            num = max(1, func_anchor(shape, arg))

            sample = create_instance(func, shape=shape, num=num)

            a, b = valid(model_ref, model, sample)
            # FIXME: wzor jakis na {a,b}?
            # FIXME: DO FUNKCJI JAKIES
            score = fn_metric(a, b)
            loss_arr.append(score)
            print(f"shape={shape} | a={a} b={b} -> score={score}")
            
            log["loss"].append(a)
            log["time"].append(b)
            log["score"].append(score)

    # FIXME: BETTER SCORE FUNCTION? multiple domains?
    # ------------------> PYTORCH ]]]]]]]]]]]]]]]]]]]
    score = sum(loss_arr)/len(loss_arr)
    print(f"----> SCORE=\033[92m {score} \033[0m")

    return score, log

################################################################################

def do_test_gen():
    for _ in tqdm(range(10)):
        sample = create_instance(gen_uniform, shape=(100, 100), num=50)    
        M = to_mat2d(sample)
        save(M[:, :, 0])

def do_test_simple():
    config_star =  {
        'anchor_double': False,
        'anchor_num': 12,
        'anchor_type': mod_anchor_type__circle,
        'bruteforce': False,
        'noise': True,
        'step_function': step_function_star
    }
    sample = create_instance(gen_uniform, shape=(32, 32), num=8)
    algo = Vorotron(config_star)
    X, T1, T2 = algo.do(sample)
    save(X["__mat2d"][:, :, 0])

TEST_DOMAIN = {
    "shapes":
        [(128, 128), (1024, 1024)],
    "cases":
        [
            {gen_uniform: [use_num, 1]},
            {gen_uniform: [use_num, 2]},
            {gen_uniform: [use_num, 4]},
            {gen_uniform: [use_density, 0.001]},
            {gen_uniform: [use_density, 0.005]},
            {gen_uniform: [use_density, 0.01]},
            # {gen_uniform: [use_density, 0.05]},
        ]
}

def do_test_error():
    # FIXME: plot?
    config_0015 = {
        'anchor_double': False,
        'anchor_num': 7,
        'anchor_type': mod_anchor_type__circle,
        'bruteforce': False,
        'noise': False,
        'step_function': step_function_star
    }
    config_0190 = {
        'anchor_double': True,
        'anchor_num': 13,
        'anchor_type': mod_anchor_type__circle,
        'bruteforce': False,
        'noise': True,
        'step_function': step_function_star
    }
    config_0195 = {
        'anchor_double': False,
        'anchor_num': 18,
        'anchor_type': mod_anchor_type__square,
        'bruteforce': False,
        'noise': True,
        'step_function': step_function_factor3
    }
    config_star = {
        'anchor_double': False,
        'anchor_num': 12,
        'anchor_type': mod_anchor_type__circle,
        'bruteforce': False,
        'noise': True,
        'step_function': step_function_star,
    }
    config_jfa = {
        'anchor_double': False,
        'anchor_num': 9,
        'anchor_type': mod_anchor_type__square,
        'bruteforce': False,
        'noise': True,
        'step_function': step_function_default,
    }
    model = Vorotron(config_jfa)
    model_ref = MODEL_BRUTEFORCE

    for shape in TEST_DOMAIN["shapes"]:
        for case in TEST_DOMAIN["cases"]:
            print("\n")
            func, params = list(case.items())[0]
            func_anchor, arg = params
            num = max(1, func_anchor(shape, arg))
        
            sample = create_instance(func, shape=shape, num=num)

            a, b = valid(model_ref, model, sample)
            print(f"shape={shape} | a={a} b={b}")

################################################################################

def step_function_star(shape, num, config=None):
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
    return steps + [1] # + [3, 2, 1]

# FIXME: pozwala polepszych jednak oslabia mocno inne parametry
# ----->               wymaga innego podejscia w przeszukiwaniu? (rundy?)
#           lub        uproszczenie / redukcja do najlepszych przypadkow
def step_function_special(shape, num=None, config=None):
    A = config["A"] #1.2 # <1, 2>
    B = config["B"] #0   # <0, 1>
    C = config["C"] #0.5 # <0, 1> FIXME: ile maksymalnie bedzie?
    D = config["D"] #1.5 # <1, 2>
    X = config["X"] #0.6 # <0.2, 1>

    steps = []
    q = num/(shape[0]*shape[1])
    qm = ((shape[0]+shape[1])/2) * q**(1/2)
    print(f"q={q} --> qm={qm}")
    # qm <-----> max(shape)
    # jakis x?
    S = B*qm + (1-B)*(max(shape)/2)
    St = math.log2(S)
    print(f"====> S={S} | {St}")

    print()
    for i in range(1, int(X*St*2), 1):
        f = round(1/(D**(i**A) + i%max(1, int(C*St))), 4)
        fm = f * S
        ffm = int(fm)
        print(f"--------> {i} f={f:10} fm={fm} | {ffm}")
        #f = math.ceil(max(shape) / (2**(i)))
        if ffm >= 1:
            steps.append(ffm)

    print()
    if len(steps) == 0:
        return [1]
    return steps

# https://www.youtube.com/watch?v=SnIAxVx7ZUs
def step_function_special__old(shape, num, config=None):
    a, b, c = config["a"], config["b"], config["c"]
    area = shape[0] * shape[1]
    q = area/(num+1)
    steps = []
    print(f"===========> {shape} {num} {area} q={q}")
    for factor in range(1, int(2*math.log2(max(shape))), 1):
        f = math.ceil(max(shape) / (3**(factor)))
        y = a*(1/q)*factor**3 - b*(1/q)*4*f**2 + c*q
        v = (y)**(1/3)
        print(factor, f, v)
        if y < 0:
            break
        steps.append(int(v))
    print("----> CUR=", steps)
    return steps

def step_function_factor3(shape, num=None, config=None):
    steps = []
    for factor in range(1, +oo, 1):
        f = math.ceil(max(shape) / (3**(factor)))
        steps.append(f)
        if f <= 1:
            break
    return steps

# FIXME: modifiers here? to code? recurse?

# FIXME: how to define here multiple anchors?
def mod_anchor_type__square(code, config=None):
    anchor_distance_ratio = config["anchor_distance_ratio"]
    if config["anchor_double"]:
        code = code.replace("#{ANCHOR_TYPE}", """
        int pos[] = {-step, -step*(#{ANCHOR_DISTANCE_RATIO}), 0, step*(#{ANCHOR_DISTANCE_RATIO}), step};
        for(int i = 0; i < 5; i++)
            for(int j = 0; j < 5; j++) {
                int nx = x + pos[i];
                int ny = y + pos[j];
        """).replace("#{ANCHOR_DISTANCE_RATIO}", str(anchor_distance_ratio))
    else:
        code = code.replace("#{ANCHOR_TYPE}", """
        int pos[] = {-step, 0, step};
        for(int i = 0; i < 3; i++)
            for(int j = 0; j < 3; j++) {
                int nx = x + pos[i];
                int ny = y + pos[j];
        """)
    return code

def mod_anchor_type__circle(code, config=None):
    anchor_num = config["anchor_num"]
    anchor_distance_ratio = config["anchor_distance_ratio"]
    anchor_number_ratio = config["anchor_number_ratio"]
    if config["anchor_double"]:
        code = code.replace("#{ANCHOR_TYPE}", """
        for(int j = -1; j < 1; j++)
        for(int i = 0; i < #{ANCHOR_NUM} * (
                ( 1 + j) * (1 - #{ANCHOR_NUMBER_RATIO}) 
                +
                (-1 * j) * (#{ANCHOR_NUMBER_RATIO})
            ); i++) {
            int A = (step * cos((float) ((6.28/#{ANCHOR_NUM}) * i) ));
            int B = (step * sin((float) ((6.28/#{ANCHOR_NUM}) * i) ));
            if (j == 0) {
               A *= #{ANCHOR_DISTANCE_RATIO};
               B *= #{ANCHOR_DISTANCE_RATIO};
            }
            int nx = x+A;
            int ny = y+B;
        """).replace("#{ANCHOR_NUM}", str(anchor_num//2)) \
            .replace("#{ANCHOR_DISTANCE_RATIO}", str(anchor_distance_ratio)) \
            .replace("#{ANCHOR_NUMBER_RATIO}", str(anchor_number_ratio))
    else:
        code = code.replace("#{ANCHOR_TYPE}", """
        for(int i = 0; i < #{ANCHOR_NUM}; i++) {
            int A = (step * cos((float) ((6.28/#{ANCHOR_NUM}) * i) ));
            int B = (step * sin((float) ((6.28/#{ANCHOR_NUM}) * i) ));
            int nx = x+A;
            int ny = y+B;
        """).replace("#{ANCHOR_NUM}", str(anchor_num))
    return code

MODEL_BRUTEFORCE = Vorotron({
    'brutforce': True
})
MODEL_JFA = Vorotron({
    'anchor_double': False,
    'anchor_num': None,
    'anchor_type': mod_anchor_type__square,
    'bruteforce': False,
    'noise': False,
    'step_function': step_function_default,
})
MODEL_JFA_STAR = Vorotron({
    'anchor_double': False,
    'anchor_num': 12,
    'anchor_type': mod_anchor_type__circle,
    'bruteforce': False,
    'noise': True,
    'step_function': step_function_star
})


# FIXME: =====================
# aby odpalic z step_function special ZMIEN forest_minimize na gp_minimize
#                                           !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

SPACE = [
    #Real(0, 1, name='a'),
    #Real(0.25, 1, name='b'),
    #Real(0, 1, name='c'),
    
    Categorical([step_function_default,
                  step_function_star,
                  step_function_factor3,
                 step_function_special],
                 name='step_function'),

    # FIXME: B nic nie robi? zawsze 1
    Real(1.65, 1.73, name='A'), #1.2 # <1, 2>
    Real(0.99,    1,    name='B'), #0   # <0, 1>
    Real(0.8,  1,    name='C'), #0.5 # <0, 1> FIXME: ile maksymalnie bedzie?
    Real(1.2,  1.25, name='D'), #1.5 # <1, 2>
    Real(0.65, 0.95, name='X'), #0.6 # <0.2, 1>
 
    #############3
    # Categorical([step_function_default,
    #              step_function_star,
    #              step_function_factor3],
    #             name='step_function'),
    ##############

    Integer(6, 12+6, name='anchor_num'),
    Categorical([False, True], name='noise'),
    Categorical([False, True], name='anchor_double'), 

    # BEST=================================
    #Categorical([step_function_default,
    #             step_function_star,
    #             step_function_factor3],
    #            name='step_function'),
    # =====================================
    
    # FIXME: distance more than 1?????? FOR FUN??????????
    Categorical([1/2, 1/3, 2/3, 1/4, 3/4], name='anchor_distance_ratio'),
    Categorical([1/2, 1/3, 2/3, 1/4, 3/4], name='anchor_number_ratio'),
    Categorical([mod_anchor_type__square,
                 mod_anchor_type__circle],
                name='anchor_type'),
]

if Config.IS_SPECIAL:
    SPACE += [
        Real(1,   2, name='A'), #1.2 # <1, 2>
        Real(0,   1, name='B'), #0   # <0, 1>
        Real(0,   1, name='C'), #0.5 # <0, 1> FIXME: ile maksymalnie bedzie?
        Real(1,   2, name='D'), #1.5 # <1, 2>
        Real(0.2, 1, name='X'), #0.6 # <0.2, 1>
    ]

    SPACE[0] = Categorical([step_function_special],
                    name='step_function')

# FIXME: WYKRESY BO NIC NIE WIDAC KOTELY!!!!!!!!!!!!!!!!!!!!

DOMAIN = {
    "shapes":
        [(64, 64), (128, 128), (256, 256), (512, 512), (768, 768)],
    "cases":
        [
            {gen_uniform: [use_num, 1]},
            {gen_uniform: [use_num, 2]},
            {gen_uniform: [use_density, 0.0001]},
            {gen_uniform: [use_density, 0.001]},
            {gen_uniform: [use_density, 0.01]},
            {gen_uniform: [use_density, 0.02]},
            {gen_uniform: [use_density, 0.03]},
            {gen_uniform: [use_density, 0.04]},
            {gen_uniform: [use_density, 0.05]},
            #{gen_uniform: [use_density, 0.1]},
        ]
}

# FIXME: znalesc na malych caly zakres parametrow?????????????????

DOMAIN_FAST = {
    "shapes":
         [(256, 256), (384, 384), (512, 512)],
    #    [(384, 384)], # best to DEBUG JFAStar
    #    [(32, 32), (128, 128)], # best to DEBUG optimizer
    #    [(32, 32), (64, 64), (128, 128), (256, 256)], # (128, 128) + (512, 512)
    "cases":
        [
            {gen_uniform: [use_num, 1]},
            {gen_uniform: [use_density, 0.001]},
            {gen_uniform: [use_density, 0.01]},
            {gen_uniform: [use_density, 0.03]},
            {gen_uniform: [use_density, 0.05]},
            #{gen_uniform: [use_density, 0.1]},
        ]
}

################################################################################

if __name__ == "__main__":
    # do_test_gen()
    #do_test_simple()
    #sys.exit()
    # do_test_error()
    # sys.exit()

    # FIXME: kolejna generacja zastepuje `MODEL_BRUTEFORCE`
    opt_result = optimize(
        MODEL_BRUTEFORCE,
        SPACE,
        DOMAIN_FAST, # DOMAIN vs DOMAIN_FAST
        n_calls=100, # (20*60) 100 vs 10*60
    )

    # FIXME: cos tu nie gra // naprawic?
    # moze tak ---> https://scikit-optimize.github.io/stable/auto_examples/plots/partial-dependence-plot-2D.html#sphx-glr-auto-examples-plots-partial-dependence-plot-2d-py
    _ = plot_objective(opt_result, n_points=40)
    # FIXME: nabrawic te marginy
    plt.savefig('figures/raport.png')

    print("ok")
