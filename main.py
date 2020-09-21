# [music]       "AC DC - 1976 - High Voltage"
# https://www.youtube.com/watch?v=78oIolvjIsw
# [music] Queens of the Stone Age - ACOUSTIC COMPILATION
# https://www.youtube.com/watch?v=qaRkqu7PJpQ

# FIXME: automatic PYOPENCL_CTX='0:1'

# !pip3 install pyopencl
# !pip3 install scikit-optimize
# !pip3 install --upgrade numba

import os
import gc
import sys
import math
import json
import subprocess
import collections
import numpy as np
import numba as nb
import pyopencl as cl
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from datetime import datetime
from pprint import pprint, pformat
from timeit import default_timer as timer
from scipy.stats.mstats import gmean, variation

from skopt import gp_minimize, forest_minimize, gbrt_minimize, dummy_minimize
from skopt.utils import use_named_args
from skopt.plots import plot_objective, plot_evaluations, plot_convergence, plot_regret
from skopt.space import Categorical, Integer, Real

# FIXME: kazdy przyklad wygenerowany i trzymany na pozniej?????????????
#                          ????????????????????????????????????????????

# VERSJA 1:
#        a) dla kazdej domeny szuka najlepszego algorytmu i zapisuje 
#               (folder odpowiednia nazwa)
#        b) dla wszystkich domen szuka najlepszego
#        c) robi to w krokach bo ma ROZNE spacy to ziterowania
#                             + optimizery

#  0) porzadki i czysty konfig
#  1) w turach optymalizacja - Special
#       -----------> DRZEWO ZALEZNOSCI /
#                        czyli co mozna z czym ratio/num ze squeare
#  2) wiecej przypadkow / gestosci?
#  3) lepsza funkcja scory / srednia? geometryczna?

try:
    import sys
    import IPython.core.ultratb

    sys.excepthook = IPython.core.ultratb.ColorTB()
except BaseException:
    pass

try:
    import google.colab
    """
    function ClickConnect(){
      console.log("Working");
      document
        .querySelector("#top-toolbar > colab-connect-button")
        .shadowRoot
        .querySelector("#connect")
        .click()
    }
    setInterval(ClickConnect,60000)
    """
    IN_COLAB = True
except BaseException:
    IN_COLAB = False

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

################################################################################

class Config:
    N_CALLS = 100 # FIXME: [2500] ustaw czas a nie ilosc iteracji
    OPTIMIZER = forest_minimize # gbrt_minimize #forest_minimize # "auto"
    DOMAIN = "DOMAIN_SPEC_MEDIUM_HIGH" #"DOMAIN_FAST" # "DOMAIN_JFASTAR"
    
    IS_MULTI_SPACE = True # 4 przypadki --> FIXME: opisac w pracy
    IS_ONLY_WORKING = True

    # XXX: old params / not working
    IS_SPECIAL_ONLY = False
    IS_CIRCLE_ONLY = False

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

if IN_COLAB:
    from google.colab import drive
    COLAB_OUTPUT = "/content/drive/My Drive/KAGGLE_OUTPUT/"
    drive.mount("/content/drive", force_remount=True)
    call(f"mkdir -p '{COLAB_OUTPUT}'")

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
    im = im.resize((512, 512), Image.NEAREST)  # FIXME: option?
    im.save(f"results/{name}", "PNG")

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

def copy_to_host(ptrs, outs, shape=None):
    mat2d_new = []
    for ptr, out in zip(ptrs, outs):
        h_new = np.empty_like(ptr)
        event = cl.enqueue_copy(SESSION["queue"], h_new, out)
        mat2d_new.append(h_new)
    event.wait()
    return np.stack(mat2d_new, axis=-1).reshape(shape[0], shape[1], 3)

# FIXME: add modifiers like `noise kernel`

def mod_step_function__default(shape, num=None, config=None):
    steps = []
    for factor in range(1, +oo, 1):
        f = math.ceil(max(shape) / (2**(factor)))
        steps.append(f)
        if f <= 1:
            break
    return steps

def mod_step_function__factor2(shape, num=None, config=None):
    steps = []
    f = 1
    for factor in range(1, +oo, 1):
        steps.append(f)
        f*=2
        if f > max(shape):
            break
    return list(reversed(steps))

class Vorotron():
    name = "VOROTRON"
    
    config = {
        "bruteforce": True,
        "step_function": mod_step_function__default,
        "anchor_type": placebo,
        "noise": "none",
        "anchor_distance_ratio": 1/2,
        "anchor_number_ratio": 1/2,
    }

    def __init__(self, config=None):
        if config:
            self.config = dict(list(self.config.items()) + list(config.items())) 
        self.kernel_bruteforce = load_prg("algo_bruteforce.cl")
        self.kernel_noise = load_prg("algo_noise.cl")
        self.kernel_lnoise = load_prg("algo_local_noise.cl")
        pprint(self.config)

        if not self.config["bruteforce"]:
            code = open("algo_template.cl", 'r').read()
            for key in ["anchor_type"]:
                code = self.config[key](code, self.config)
            self.kernel = load_prg_from_str(code)
            self.code = code

    def do(self, x):
        if "__mat2d" not in x:
            x["__mat2d"] = to_mat2d(x)

        h_id_in = x["__mat2d"][:,:,0].flatten().astype(np.uint32)
        h_x_in = x["__mat2d"][:,:,1].flatten().astype(np.uint32)
        h_y_in = x["__mat2d"][:,:,2].flatten().astype(np.uint32)

        T_ALL_1, T_CPU, T_GPU = timer(), 0, 0
        
        # === INPUT ===
        if self.config["bruteforce"]:
            points_in = cl.Buffer(SESSION["ctx"], mf.COPY_HOST_PTR |
                                  mf.READ_ONLY, hostbuf=x["points"])
            seeds_in = cl.Buffer(SESSION["ctx"], mf.COPY_HOST_PTR |
                                 mf.READ_ONLY, hostbuf=x["seeds"])
        else:
            d_id_in = cl.Buffer(SESSION["ctx"], mf.COPY_HOST_PTR | mf.READ_WRITE,
                             hostbuf=h_id_in)
            d_x_in = cl.Buffer(SESSION["ctx"], mf.COPY_HOST_PTR | mf.READ_WRITE,
                             hostbuf=h_x_in)
            d_y_in = cl.Buffer(SESSION["ctx"], mf.COPY_HOST_PTR | mf.READ_WRITE,
                             hostbuf=h_y_in)

        # === OUTPUT ===
        d_id_out = cl.Buffer(SESSION["ctx"], mf.READ_WRITE, h_id_in.nbytes)
        d_x_out = cl.Buffer(SESSION["ctx"], mf.READ_WRITE, h_x_in.nbytes)
        d_y_out = cl.Buffer(SESSION["ctx"], mf.READ_WRITE, h_y_in.nbytes)

        # if self.config["noise"] == "noise" or self.config["noise"] == "lnoise":
        # if self.config["noise"] == "lnoise":
        #     x["__mat2d"] = copy_to_host((h_id_in, h_x_in, h_y_in), (d_id_in,
        #                                                             d_x_in,
        #                                                             d_y_in), x["shape"])
        #     save(x["__mat2d"][:, :, 0], prefix="input")

        # === NOISE ===
        if self.config["noise"] == "noise":
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
                                 d_id_in, d_x_in, d_y_in,
                                 np.int32(x["shape"][0]),
                                 np.int32(x["shape"][1]),
                                 np.int32(x["points"].shape[0]),
                                 )
            event.wait()
            T_GPU += event.profile.end-event.profile.start
            T2 = timer()
            T_CPU += T2-T1

            # # DEBUG -----------------
            # x["__mat2d"] = copy_to_host((h_id_in, h_x_in, h_y_in), (d_id_in,
            #                                                         d_x_in,
            #                                                         d_y_in), x["shape"])
            # save(x["__mat2d"][:, :, 0], prefix="noise")
            # # -----------------------

        elif self.config["noise"] == "lnoise":
            points_in = cl.Buffer(SESSION["ctx"], mf.COPY_HOST_PTR |
                                  mf.READ_ONLY, hostbuf=x["points"])
            seeds_in = cl.Buffer(SESSION["ctx"], mf.COPY_HOST_PTR |
                                 mf.READ_ONLY, hostbuf=x["seeds"])

            # === RUN ===
            T1 = timer()
            event = self.kernel_lnoise.fn(
                                 SESSION["queue"],
                                 (x["shape"][1], x["shape"][0], 1), None,
                                 points_in,
                                 seeds_in,
                                 d_id_in, d_x_in, d_y_in,
                                 np.int32(x["shape"][0]),
                                 np.int32(x["shape"][1]),
                                 np.int32(x["points"].shape[0]),
                                 )
            event.wait()
            T_GPU += event.profile.end-event.profile.start
            T2 = timer()
            T_CPU += T2-T1

            # # DEBUG -----------------
            # x["__mat2d"] = copy_to_host((h_id_in, h_x_in, h_y_in), (d_id_in,
            #                                                         d_x_in,
            #                                                         d_y_in), x["shape"])
            # save(x["__mat2d"][:, :, 0], prefix="lnoise")
            # # -----------------------

        # === RUN ===
        if self.config["bruteforce"]:
            T1 = timer()
            event = self.kernel_bruteforce.fn(
                           SESSION["queue"],
                           (x["shape"][1], x["shape"][0], 1),
                           None, # default?
                           points_in,
                           seeds_in,
                           d_id_out, d_x_out, d_y_out,
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
            # print(steps)

            for step in steps:
                T1 = timer()
                event = self.kernel.fn(
                               SESSION["queue"],
                               (x["shape"][1], x["shape"][0], 1),
                               None,
                               d_id_in, d_x_in, d_y_in,
                               d_id_out, d_x_out, d_y_out,
                               np.int32(x["shape"][0]),
                               np.int32(x["shape"][1]),
                               np.int32(step)
                               )
                event.wait()
                T_GPU += event.profile.end-event.profile.start
                T2 = timer()
                T_CPU += T2-T1
                d_id_in, d_id_out = d_id_out, d_id_in
                d_x_in, d_x_out = d_x_out, d_x_in
                d_y_in, d_y_out = d_y_out, d_y_in
            d_id_in, d_id_out = d_id_out, d_id_in
            d_x_in, d_x_out = d_x_out, d_x_in
            d_y_in, d_y_out = d_y_out, d_y_in

        T_ALL_2 = timer()
        x["__mat2d"] = copy_to_host((h_id_in, h_x_in, h_y_in), (d_id_out, d_x_out, d_y_out), x["shape"])
        return x, T_GPU, T_ALL_2-T_ALL_1, T_CPU

################################################################################

def use_num(shape, x): return x
def use_density(shape, x): return int(shape[0] * shape[1] * x)

def fn_loss(m1, m2):
    return 1 - np.count_nonzero((m1-m2) == 0)/(m1.shape[0]*m1.shape[1])

def valid(model1, model2, x, n=3):
    loss_arr, time_1_arr, time_2_arr = [], [], []

    # FIXME: jesli nie ma duzych roznic przerwij - wczesniej?
    def do_single():
        # bruteforce
        if "__mat2d" in x:
            del x["__mat2d"]
        m1, t1, _, _ = model1.do(x)
        m1 = m1["__mat2d"][:, :, 0]
        #save(m1, prefix="brute")
        # algorithm
        if "__mat2d" in x:
            del x["__mat2d"]
        m2, t2, _, _ = model2.do(x)
        m2 = m2["__mat2d"][:, :, 0]
        #save(m2) # FIXME: debug
        # error
        loss = fn_loss(m1, m2)
        # append
        loss_arr.append(loss)
        time_1_arr.append(t1)
        time_2_arr.append(t2)

    for _ in range(n):
        do_single() # n=3

    if max([variation(time_1_arr), variation(time_2_arr)]) > 0.05:
        do_single() # +1

    # FIXME: debug
    # save(m1, prefix="gt")
    # save(m2, prefix="jfa")
 
    def avgarr(arr):    
        shift = max(int(len(arr)*(0.1)), 1)
        arr = sorted(arr)[shift:-shift]
        # XXX: return round(sum(arr)/len(arr), 6)
        return round(gmean(arr), 6)

    loss = avgarr(loss_arr)*100
    time_1 = avgarr(time_1_arr)
    time_2 = avgarr(time_2_arr)
    
    loss_diff = round(loss, 6)
    time_diff = round(time_1 / time_2, 6)
    print(f"\033[91m [[ {model2.name.ljust(20)} ]] =========> loss: {loss_diff:8}% \n\t| \033[94m time: {time_1/1e9:6} : {time_2/1e9:6}\033[0m")

    gc.collect()
    return loss_diff, time_diff

def evaluate(model, x, n=5):
    loss_arr, time_arr = [], []

    def do_single():
        if "__mat2d" in x:
            del x["__mat2d"]
        m, t, _, _ = model.do(x)
        m = m["__mat2d"][:, :, 0]
        #save(m1)
        loss = fn_loss(x["optimal_solution"], m)
        # append
        loss_arr.append(loss)
        time_arr.append(t)

    for _ in range(n):
        do_single() # n=3

    if variation(time_arr) > 0.05:
        do_single() # +1
 
    def avgarr(arr):    
        shift = max(int(len(arr)*(0.1)), 1)
        arr = sorted(arr)[shift:-shift]
        # XXX: return round(sum(arr)/len(arr), 6)
        return round(gmean(arr), 6)

    avg_loss = avgarr(loss_arr)*100
    avg_time = avgarr(time_arr)
    
    print(f"\033[91m [[ {model.name.ljust(20)} ]] =========> loss: {avg_loss:8}% \n\t| \033[94m time: {avg_time/1e9:6}\033[0m")

    gc.collect()
    return avg_loss, avg_time

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
    if config["noise"] == "noise":
        noise = "+Noise"
    elif config["noise"] == "lnoise":
        noise = "+LNoise"
    else:
        noise = ""
    step_function = config["step_function"].__name__.replace("mod_step_function__", "").title()

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

    return f"{anchor_type}{anchor_num}{anchor_double}-{step_function}{special}{noise}"

def save_domain(domain):
    domain_vec = []
    domain_generated = []
    for shape in domain["shapes"]:
        for case in domain["cases"]:
            print("\n")
            func, params = list(case.items())[0]
            func_anchor, arg = params
            num = max(1, func_anchor(shape, arg))
            domain_vec.append({"shape": shape, "num": num})

            sample = create_instance(func, shape=shape, num=num)
            domain_generated.append(sample)

    pprint(domain_vec)

    path_domain = "results/domain.json"
    domain_cl = json.dumps(domain_vec)
    text_file = open(path_domain, "w")
    text_file.write(domain_cl)
    text_file.close()

    return domain_generated

i_calls, pbar = 0, None # FIXME: progress bar? tqdm?
best_name, best_score = "?", 0
def optimize(model_ref, space, domain_generated, n_calls=10, prefix=None):
    global i_calls, pbar
    pbar = tqdm(total=n_calls)
    ALGOMAP = {}

    if not prefix:
        prefix = "all"

    def add_optimal_solutions(domain_generated, model):
        for i, sample in enumerate(domain_generated):
            if "__mat2d" in sample:
                del sample["__mat2d"]
            m, _, _, _ = model.do(sample)
            m = m["__mat2d"][:, :, 0]
            sample["optimal_solution"] = m

    def add_reference_time(domain_generated, model):
        for i, sample in enumerate(domain_generated):
            if "__mat2d" in sample:
                del sample["__mat2d"]
            _, t = evaluate(model, sample)
            sample["reference_time"] = t

    def score_from_config(config):
        global i_calls, best_name, best_score
        print("======= EXPERIMENT =======")

        name = human_algo_name(config)
        if name == "Square-Default":
            name = "JFA (original)"
        if name in ALGOMAP:
            print(f"----> {name} already saved")
            return ALGOMAP[name]

        model = Vorotron(config)
        model.name = name
        score, log = do_comparison(model, model_ref, domain_generated)

        path_log = f"results/log/{score}.json"
        path_code = f"results/code/{score}.cl"

        if name == "JFA (original)":
            print("=============================> SPECIAL (JFA)")
            path_log = f"results/jfa.json"
            path_code = f"results/jfa.cl"
            call(f"touch results/jfa={score}")

        log["name"] = f"({int(score)}) " + name
        print(f"[[[[[[[[[[[ \033[96m {log['name']} \033[0m ]]]]]]]]]]]")
        
        log_cl = json.dumps(log)
        text_file = open(path_log, "w")
        text_file.write(log_cl)
        text_file.close()

        config_cl = pformat(config, indent=4)
        result_cl = "/*\n" + config_cl + "\n*/\n\n" + model.code
        text_file = open(path_code, "w")
        text_file.write(result_cl)
        text_file.close()

        print(f"========================== \033[94m {i_calls}/{n_calls} \033[0m")
        if score <= best_score:
            best_name, best_score = log["name"], score
        ALGOMAP[name] = score
        return score

    @use_named_args(space)
    def score(**config):
        global i_calls, pbar
        print("======= EXPERIMENT =======")
        # FIXME: dla roznych domen rozne wyniki?
        i_calls += 1
        config["bruteforce"] = False
        score = score_from_config(config)
        pbar.update(1)
        ########### SECURITY ##############
        if IN_COLAB and i_calls % 100 == 0 and i_calls > 1:
            path_date = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
            call(f"cp -r results/ '{COLAB_OUTPUT}/{Config.DOMAIN}|{path_date}({prefix}|{i_calls}|{Config.N_CALLS})'")
        return score

    jfa_config = {
        'anchor_double': False,
        'anchor_num': None,
        'anchor_type': mod_anchor_type__square,
        'bruteforce': False,
        'noise': "none",
        'step_function': mod_step_function__default,
    }

    add_optimal_solutions(domain_generated, model_ref)
    add_reference_time(domain_generated, Vorotron(jfa_config))

    score_from_config(jfa_config)

    # sys.exit()

    if isinstance(Config.OPTIMIZER, collections.Callable):
        print(f"-------> {Config.OPTIMIZER}")
        obj = Config.OPTIMIZER(func=score, dimensions=space,
                                n_calls=n_calls, random_state=0)
    else:
        if Config.IS_SPECIAL_ONLY:
            # faster but lower results?
            obj = gp_minimize(func=score, dimensions=space,
                               n_calls=n_calls, random_state=0)
        else:
            obj = forest_minimize(func=score, dimensions=space,
                               n_calls=n_calls, random_state=0)

    pbar.close()
    return obj

def fn_metric(loss, speedup): # b/(1+a)
    # return (math.sqrt(b) * (100-a**1.85))
    return loss/speedup

    # FIXME: score is still wrong?
    ######################################
    # return (math.sqrt(b) * (100-a**2)) #
    ######################################
    # FIXME: return max(0, (math.sqrt(b) * (100-a**2)))
    # return max(0, (b * (100-a**1.5)))

# FIXME: if first k is totally wrong resign???
def do_comparison(model, model_ref, domain_generated=None, k = 0.25): 
    loss_arr = []

    log = {
        'loss': [],
        'time': [],
        'score': []
    }

    def calc_metric(arr):
        _score = 0
        if Config.IS_ONLY_WORKING:
            _score = sum(arr)/len(arr)
            #score = gmean(loss_arr) # dla choc jednego zera -> score=0
        else:
            _score = math.sqrt(sum(arr)/len(arr))
        return _score

    for i, sample in enumerate(domain_generated):
        if i/len(domain_generated) >= k and calc_metric(loss_arr) < 0:
            print("\n")
            a, b, score = 0, 100, 0
        else:
            # a, b = valid(model_ref, model, sample)
            a, b = evaluate(model, sample)
            b/= sample["reference_time"]
            score = fn_metric(a, b)
            if Config.IS_ONLY_WORKING:
                loss_arr.append(score)
            else:
                loss_arr.append(max(0, score)**2)
            print(f"shape={sample['shape']} | a={a} b={b} -> score={score}")
        
        log["loss"].append(a)
        log["time"].append(b)
        log["score"].append(score)

    score = calc_metric(loss_arr)
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
        'noise': "noise",
        'step_function': mod_step_function__star
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
    config_0015 = {
        'anchor_double': False,
        'anchor_num': 7,
        'anchor_type': mod_anchor_type__circle,
        'bruteforce': False,
        'noise': "none",
        'step_function': mod_step_function__star
    }
    config_0190 = {
        'anchor_double': True,
        'anchor_num': 13,
        'anchor_type': mod_anchor_type__circle,
        'bruteforce': False,
        'noise': "noise",
        'step_function': mod_step_function__star
    }
    config_0195 = {
        'anchor_double': False,
        'anchor_num': 18,
        'anchor_type': mod_anchor_type__square,
        'bruteforce': False,
        'noise': "noise",
        'step_function': mod_step_function__factor3
    }
    config_star = {
        'anchor_double': False,
        'anchor_num': 12,
        'anchor_type': mod_anchor_type__circle,
        'bruteforce': False,
        'noise': "noise",
        'step_function': mod_step_function__star,
    }
    config_jfa = {
        'anchor_double': False,
        'anchor_num': 9,
        'anchor_type': mod_anchor_type__square,
        'bruteforce': False,
        'noise': "noise",
        'step_function': mod_step_function__default,
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

def mod_step_function__star(shape, num, config=None):
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
    steps, i = [], 0
    for factor in range(1, +oo, 1):
        i += 1
        f = math.ceil(max(shape) / (3**(factor)))
        steps.append(f)
        if f <= 1 or i == n + 1:
            break
    return steps + [1]

# FIXME: pozwala polepszych jednak oslabia mocno inne parametry
# ----->               wymaga innego podejscia w przeszukiwaniu? (rundy?)
#           lub        uproszczenie / redukcja do najlepszych przypadkow
def mod_step_function__special(shape, num=None, config=None):
    A = config["A"] #1.2 # <1, 2>
    B = config["B"] #0   # <0, 1>
    C = config["C"] #0.5 # <0, 1> FIXME: ile maksymalnie bedzie?
    D = config["D"] #1.5 # <1, 2>
    X = config["X"] #0.6 # <0.2, 1>

    steps = []
    q = num/(shape[0]*shape[1])
    qm = ((shape[0]+shape[1])/2) * q**(1/2)
    # print(f"q={q} --> qm={qm}")
    S = B*qm + (1-B)*(max(shape)/2)
    St = math.log2(S)
    # print(f"====> S={S} | {St}")

    # print()
    for i in range(1, int(X*St*2), 1):
        f = round(1/(D**(i**A) + i%max(1, int(C*St))), 4)
        fm = f * S
        ffm = int(fm)
        # print(f"--------> {i} f={f:10} fm={fm} | {ffm}")
        if ffm >= 1:
            steps.append(ffm)

    # print()
    if len(steps) == 0:
        return [1]
    return steps

# https://www.youtube.com/watch?v=SnIAxVx7ZUs
# https://www.youtube.com/watch?v=fWvKvOViM3g
# https://www.youtube.com/watch?v=Tx9zMFodNtA
# https://www.youtube.com/watch?v=ckKi01gqW7A
def mod_step_function__special__old(shape, num, config=None):
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

def mod_step_function__factor3(shape, num=None, config=None):
    steps = []
    for factor in range(1, +oo, 1):
        f = math.ceil(max(shape) / (3**(factor)))
        steps.append(f)
        if f <= 1:
            break
    return steps

def mod_step_function__limited_factors(shape, num=None, config=None):
    steps = []
    factor = 2**config["factor"] #<.5; 2>
    num_steps = config["num_steps"] #{1,2,3,...,16}
    step = 1
    for i in range(num_steps):
        steps.append(step)
        step*=factor
    return list(reversed(steps))

# FIXME: nie dziala `anchor_number_ration` oraz `anchor_num`
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

# [[[BEST FOR 384x384]]]
# Real(1.65, 1.73, name='A'), # <1,   2>
# Real(0.99,    1, name='B'), # <0,   1>
# Real(0.8,     1, name='C'), # <0,   1>
# Real(1.2,  1.25, name='D'), # <1,   2>
# Real(0.65, 0.95, name='X'), # <0.2, 1>

# [[[BEST FOR 256x256]]]
# Real(1.40,  1.5, name='A'), # <1,   2>
# Real(0.99,    1, name='B'), # <0,   1>
# Real(0.8,     1, name='C'), # <0,   1>
# Real(1.2,   1.3, name='D'), # <1,   2>
# Real(0.65,    1, name='X'), # <0.2, 1>

# [[[BEST FOR 512x512]]
# Real(1.46,  1.5, name='A'), # <1,   2>
# Real(0.99,    1, name='B'), # <0,   1>
# Real(0.95,    1, name='C'), # <0,   1>
# Real(1.26, 1.33, name='D'), # <1,   2>
# Real(0.71, 0.90, name='X'), # <0.2, 1>

################################################################################

SPACE_SQUARE_NORMAL = [
    Categorical([mod_step_function__default,
                 mod_step_function__star,
                 mod_step_function__factor3],
                name='step_function'),

    Categorical(["none", "noise", "lnoise"], name='noise'),
    Categorical([False, True], name='anchor_double'),

    Categorical([1/2, 1/3, 2/3, 1/4, 3/4], name='anchor_distance_ratio'),

    Categorical([mod_anchor_type__square],
                name='anchor_type'),
]

SPACE_SQUARE_SPECIAL = [
    Categorical([mod_step_function__limited_factors],
                name='step_function'),

    Real(0.5,   2, name='factor'), # <1,   2>
    Integer(1, 16, name='num_steps'), # {1,2,3,...16}

    Categorical(["none", "noise", "lnoise"], name='noise'),
    Categorical([False, True], name='anchor_double'),

    Categorical([1/2, 1/3, 2/3, 1/4, 3/4], name='anchor_distance_ratio'),

    Categorical([mod_anchor_type__square],
                name='anchor_type'),
]

SPACE_CIRCLE_NORMAL = [
    Categorical([mod_step_function__default,
                 mod_step_function__star,
                 mod_step_function__factor3],
                name='step_function'),

    # FIXME: Integer(4, 12+6, name='anchor_num'),
    Integer(9, 16, name='anchor_num'),
    Categorical(["none", "noise", "lnoise"], name='noise'),
    Categorical([False, True], name='anchor_double'), 
    
    Categorical([1/2, 1/3, 2/3, 1/4, 3/4], name='anchor_distance_ratio'),
    Categorical([1/2, 1/3, 2/3, 1/4, 3/4], name='anchor_number_ratio'),

    Categorical([mod_anchor_type__circle],
                name='anchor_type'),
]

SPACE_CIRCLE_SPECIAL = [
    Categorical([mod_step_function__limited_factors],
                name='step_function'),
    Real(0.5,   2, name='factor'), # <1,   2>
    Integer(1, 16, name='num_steps'), # {1,2,3,...16}

    # FIXME: Integer(4, 12+6, name='anchor_num'),
    Integer(9, 16, name='anchor_num'),
    Categorical(["none", "noise", "lnoise"], name='noise'),
    Categorical([False, True], name='anchor_double'),

    Categorical([1/2, 1/3, 2/3, 1/4, 3/4], name='anchor_distance_ratio'),
    Categorical([1/2, 1/3, 2/3, 1/4, 3/4], name='anchor_number_ratio'),

    Categorical([mod_anchor_type__circle],
                name='anchor_type'),
]

SPACE_ALL = {
    "square_normal": SPACE_SQUARE_NORMAL,
    "square_special": SPACE_SQUARE_SPECIAL,
    "circle_normal": SPACE_CIRCLE_NORMAL,
    "circle_special": SPACE_CIRCLE_SPECIAL
}

################################################################################

SPACE = [
    Categorical([mod_step_function__default,
                 mod_step_function__star,
                 mod_step_function__factor3,
                 mod_step_function__special],
                name='step_function'),

    Real(1,   2, name='A'), # <1,   2>
    Real(0,   1, name='B'), # <0,   1>
    Real(0,   1, name='C'), # <0,   1>
    Real(1,   2, name='D'), # <1,   2>
    Real(0.2, 1, name='X'), # <0.2, 1>

    Integer(4, 12+6, name='anchor_num'),
    Categorical(["none", "noise", "lnoise"], name='noise'),
    Categorical([False, True], name='anchor_double'), 
    
    Categorical([1/2, 1/3, 2/3, 1/4, 3/4], name='anchor_distance_ratio'),
    Categorical([1/2, 1/3, 2/3, 1/4, 3/4], name='anchor_number_ratio'),

    Categorical([mod_anchor_type__square,
                 mod_anchor_type__circle],
                name='anchor_type'),
]

if Config.IS_SPECIAL_ONLY:
    SPACE[1:1+5] = [
        Real(1,   2, name='A'), # <1,   2>
        Real(0,   1, name='B'), # <0,   1>
        Real(0,   1, name='C'), # <0,   1>
        Real(1,   2, name='D'), # <1,   2>
        Real(0.2, 1, name='X'), # <0.2, 1>
    ]

    SPACE[0] = Categorical([mod_step_function__special],
                    name='step_function')

if Config.IS_CIRCLE_ONLY:
    SPACE[-1] = Categorical([mod_anchor_type__circle],
                    name='anchor_type')

################################################################################

SHAPES_SMALL = [(32, 32), (64, 64), (96, 96), (128, 128)]
SHAPES_MEDIUM = [(256, 256), (320, 320), (384, 384), (448, 448)]
SHAPES_LARGE = [(512, 512), (768, 768), (1024, 1024), (1536, 1536)]

CASES_LOW = [
    {gen_uniform: [use_num, 1]},
    {gen_uniform: [use_num, 3]},
    {gen_uniform: [use_density, 0.00005]},
    {gen_uniform: [use_density, 0.0001]},
    {gen_uniform: [use_density, 0.0002]},
    {gen_uniform: [use_density, 0.0003]},
    {gen_uniform: [use_density, 0.0004]},
    {gen_uniform: [use_density, 0.0005]},
    {gen_uniform: [use_density, 0.001]},
]

CASES_LOW_FAST = [ # XXX
    {gen_uniform: [use_num, 1]},
    {gen_uniform: [use_num, 3]},
    {gen_uniform: [use_density, 0.0001]},
    {gen_uniform: [use_density, 0.0003]},
    {gen_uniform: [use_density, 0.001]},
]

CASES_HIGH = [
    {gen_uniform: [use_density, 0.01]},
    {gen_uniform: [use_density, 0.02]},
    {gen_uniform: [use_density, 0.03]},
    {gen_uniform: [use_density, 0.04]},
    {gen_uniform: [use_density, 0.05]},
    {gen_uniform: [use_density, 0.1]},
]

CASES_HIGH_FAST = [ # XXX
    {gen_uniform: [use_density, 0.01]},
    {gen_uniform: [use_density, 0.03]},
    {gen_uniform: [use_density, 0.05]},
    {gen_uniform: [use_density, 0.1]},
]

################################################################################

# FIXME: dodac jako `param` jak shapes czy cases:
#                  IS_ONLY_WORKING ---> ???

DOMAIN_JFASTAR = {
    "shapes": SHAPES_SMALL + SHAPES_MEDIUM + SHAPES_LARGE,
    "cases": CASES_LOW_FAST + CASES_HIGH_FAST
}

### SMALL ###

DOMAIN_SPEC_SMALL_LOW = {
    "shapes": SHAPES_SMALL,
    "cases": CASES_LOW
}

DOMAIN_SPEC_SMALL_HIGH = {
    "shapes": SHAPES_SMALL,
    "cases": CASES_HIGH
}

### MEDIUM ###

DOMAIN_SPEC_MEDIUM_LOW = {
    "shapes": SHAPES_MEDIUM,
    "cases": CASES_LOW
}

DOMAIN_SPEC_MEDIUM_HIGH = {
    "shapes": SHAPES_MEDIUM,
    "cases": CASES_HIGH
}

### LARGE ###

DOMAIN_SPEC_LARGE_LOW = {
    "shapes": SHAPES_LARGE,
    "cases": CASES_LOW
}

DOMAIN_SPEC_LARGE_HIGH = {
    "shapes": SHAPES_LARGE,
    "cases": CASES_HIGH
}

### ??? ###

DOMAIN_SPEC_LOW = {
    "shapes": SHAPES_SMALL + SHAPES_MEDIUM + SHAPES_LARGE,
    "cases": CASES_LOW_FAST
}

DOMAIN_SPEC_HIGH = {
    "shapes": SHAPES_SMALL + SHAPES_MEDIUM + SHAPES_LARGE,
    "cases": CASES_HIGH_FAST
}

################################################################################

# FIXME: remove small cases? IS_ONLY_WORKING -> false?
# co gdy dla malych slabo dziala? IS_ONLY_WORKING
#        zrobic ramke/window na szukanie scoru

DOMAIN_DEBUG = {
    "shapes":
        [(32, 32)],
    "cases":
        [
            {gen_uniform: [use_num, 3]},
            {gen_uniform: [use_density, 0.01]},
            {gen_uniform: [use_density, 0.05]},
        ]
}

################################################################################

if __name__ == "__main__":
    # do_test_gen()
    # do_test_simple()
    # sys.exit()
    # do_test_error()
    # sys.exit()

    # FIXME: experiment name!!!!!!!!!!!!

    domain = globals()[Config.DOMAIN]
    domain_generated = save_domain(domain)

    if Config.IS_MULTI_SPACE:
        for name, SPACE_i in SPACE_ALL.items():
            opt_result = optimize(
                MODEL_BRUTEFORCE,
                SPACE_i,
                domain_generated, # DOMAIN vs DOMAIN_FAST
                n_calls=Config.N_CALLS, # BEST:[100] /// (20*60) 100 vs 10*60
                prefix=name
            )
    else:
        opt_result = optimize(
            MODEL_BRUTEFORCE,
            SPACE,
            domain_generated, # DOMAIN vs DOMAIN_FAST
            n_calls=Config.N_CALLS, # BEST:[100] /// (20*60) 100 vs 10*60
            prefix=None
        )

    print(f"\033[92m {best_name} \033[0m")
    
    if IN_COLAB:
        path_date = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
        call(f"cp -r results/ '{COLAB_OUTPUT}/{Config.DOMAIN}|{path_date}'")
    print("end")

    # FIXME: inline table? just to see results? (from plot.py)

    # _ = plot_objective(opt_result, n_points=40)
    # FIXME: nabrawic te marginy
    # moze tak ---> https://scikit-optimize.github.io/stable/auto_examples/plots/partial-dependence-plot-2D.html#sphx-glr-auto-examples-plots-partial-dependence-plot-2d-py
    # plt.savefig('figures/raport.png')

    # print("ok")
