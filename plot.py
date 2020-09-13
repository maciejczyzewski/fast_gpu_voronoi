import matplotlib.pyplot as plt
import numpy as np
import unidecode
import json, csv
import re

from glob import glob
from pprint import pprint

###############################
# [IDEAS]
# FIXME: 3d
# FIXME: sorted by score
# FIXME: time/loss (power line)
# FIXME: [[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[generacja tabelek]]]]]]]]
# FIXME: case to ax[x]?????????????
###############################

def slugify(text):
    text = unidecode.unidecode(text).lower()
    text = re.sub(r"[\W_]+", "-", text)
    if text[-1] == "-":
        return text[0:-1]
    return text


class figure:
    def __init__(self, name=None, prefix=None):
        self.name = name
        self.prefix = prefix

    def __enter__(self):
        print("--- FIGURE ---")
        print(f"`{self.name}`")
        plt.cla()
        plt.title(self.name)

    def __exit__(self, x, y, z):
        print("--- SAVE ---")
        figure_prefix = "figure-"
        if self.prefix is not None:
            figure_prefix += f"{str(self.prefix)}-"
        fig.savefig(f"figures/{figure_prefix}{slugify(self.name)}.pdf")


# (1) better style
plt.style.use(["science", "ieee"])
plt.rcParams.update({"text.usetex": True})

fig, ax = plt.subplots()
ax.autoscale(tight=True)

def read_file(path=None, x_name=None, y_name=None):
    X, Y = [], []
    with open(path, "r") as jsonfile:
        json_txt = jsonfile.read()
        log = json.loads(json_txt)
        for i in range(len(log["score"])):
            if x_name is not None:
                X.append(float(log[x_name][i]))
            else:
                X.append(i)
            Y.append(float(log[y_name][i]))
    return X, Y, log

def apply_SOTA(x_name=None, y_name=None, sort=False):
    # for [JFA]
    path = "results/jfa.json"
    x, y, log = read_file(path, x_name=x_name, y_name=y_name)
    if sort:
        y = sorted(y)
    plt.plot(x, y, label=log["name"], color="black", linestyle='solid')
    ax.fill_between(x, y, 0, facecolor='black', alpha=0.2)

    # for [BRUTFORCE]
    if y_name == "score":
        plt.plot(x, [100]*len(x), label="brutforce", color="red", linestyle='solid')
        ax.fill_between(x, [100]*len(x), 0, facecolor='red', alpha=0.2)
    if y_name == "loss":
        plt.plot(x, [0]*len(x), label="brutforce", color="red", linestyle='solid')
        ax.fill_between(x, [0]*len(x), 0, facecolor='red', alpha=0.2)
    if y_name == "time":
        plt.plot(x, [1]*len(x), label="brutforce", color="red", linestyle='solid')
        ax.fill_between(x, [1]*len(x), 0, facecolor='red', alpha=0.2)
    ax.grid(linestyle="--", linewidth=0.5, color='.25', zorder=-10, alpha=0.1)

def globlog():
    vec = []
    for path in glob("results/log/*.json"):
        x = float(path.replace(".json", "").replace("results/log/", ""))
        vec.append([x, path])
    return sorted(vec)[::-1]

### FIGURE (1): underfitting ###

# FIXME:
# - u gory: segmenty z np. 64x64, 128x128
# - na dole: jakie `seeds` jakie `rho`

def apply_domain__classic(x=None, with_shape=True):
    path = "results/domain.json"
    with open(path, "r") as jsonfile:
        json_txt = jsonfile.read()
        domain = json.loads(json_txt)
    pprint(domain)
    domain_str = []
    for d in domain:
        q = round(d['num']/(d['shape'][0]*d['shape'][1]), 4)
        if with_shape:
            domain_str.append(f"{d['shape'][0]}x{d['shape'][1]}\n$\\Omega$={d['num']}\n$\\rho$={q}")
        else:
            domain_str.append(f"$\\Omega$={d['num']}\n$\\rho$={q}")
    plt.xticks(x, domain_str, rotation='horizontal', fontsize=2)

def apply_domain__groups(x=None):
    path = "results/domain.json"
    with open(path, "r") as jsonfile:
        json_txt = jsonfile.read()
        domain = json.loads(json_txt)
    pprint(domain)
    
    groups = []
    cur_start, cur_span, cur_shape, first = -1, 0, "?x?", True
    for i, d in enumerate(domain):
        shape = f"{d['shape'][0]}x{d['shape'][1]}"
        if cur_span != 0 and shape != cur_shape:
            if first is not True:
                groups.append((cur_shape, (x[cur_start], x[cur_start+cur_span])))
            cur_start, cur_span, cur_shape = i, 0, shape
            if first is True:
                cur_start -= 1
                cur_span += 1
            first = False
        else:
            cur_span += 1
    groups.append((cur_shape, (x[cur_start], x[cur_start+cur_span])))
    
    pprint(groups)

    for name, xspan in groups:
        annotate_group(name, xspan)

    ax.tick_params(axis="x", bottom=True, top=True, labelbottom=False, labeltop=True)
    domain_str = []
    for d in domain:
        q = round(d['num']/(d['shape'][0]*d['shape'][1]), 4)
        domain_str.append(f"$\\Omega$={d['num']}\n$\\rho$={q}")
    plt.xticks(x, domain_str, rotation='horizontal', fontsize=2)

def apply_domain(x=None):
    # FIXME: dodac umiejetnosc pomijania niektorych? jak jest gesto?
    # apply_domain__classic(x)
    apply_domain__groups(x)
    # plt.xticks(x, domain_str, rotation='horizontal', fontsize=2)
    # import matplotlib.ticker as mticker
    # myLocator = mticker.MultipleLocator(2)
    # ax.xaxis.set_major_locator(myLocator)
    #ax.tick_params(axis="x", bottom=True, top=True, labelbottom=True, labeltop=True)
    #plt.setp([tick.label1 for tick in ax.xaxis.get_major_ticks()], rotation='horizontal', fontsize=2)
    #plt.setp([tick.label1 for tick in ax.xaxis.get_major_ticks()], rotation='horizontal', fontsize=2)

def annotate_group(name, xspan, ax=None):
    """Annotates a span of the x-axis"""
    def annotate(ax, name, left, right, y, pad):
        arrow = ax.annotate(name,
                xy=(left, y), xycoords='data',
                xytext=(right, y-pad), textcoords='data',
                annotation_clip=False, verticalalignment='top',
                horizontalalignment='center', linespacing=2.0,
                arrowprops=dict(arrowstyle='-', shrinkA=0, shrinkB=0,
                        connectionstyle='angle,angleB=90,angleA=0,rad=5')
                )
        return arrow
    if ax is None:
        ax = plt.gca()
    ymin = ax.get_ylim()[0]
    ypad = 0.01 * np.ptp(ax.get_ylim())
    xcenter = np.mean(xspan)
    left_arrow = annotate(ax, name, xspan[0], xcenter, ymin, ypad)
    right_arrow = annotate(ax, name, xspan[1], xcenter, ymin, ypad)
    return left_arrow, right_arrow

COLORS = ['g', 'b', 'm', 'orange']

################################################################################
# FIGURES
################################################################################

with figure("time", prefix=1):
    # FIXME: gray -----> alpha change on i?
    for i, (_, path) in enumerate(globlog()):
        x, y, log = read_file(path, x_name=None, y_name="time")
        if i <= 3:
            plt.plot(x, y, label=log["name"], color=COLORS[i], linestyle='--')
        else:
            plt.plot(x, y, label=log["name"], color="gray", alpha=0.1, linestyle='solid')
        if i >= 11:
            break

    apply_SOTA(x_name=None, y_name="time", sort=False)
    apply_domain(x)

    plt.ylabel("time")
    plt.xlabel("case")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

with figure("loss", prefix=2):
    for i, (_, path) in enumerate(globlog()):
        x, y, log = read_file(path, x_name=None, y_name="loss")
        if i <= 3:
            plt.plot(x, y, label=log["name"], color=COLORS[i], linestyle='--')
        else:
            plt.plot(x, y, label=log["name"], color="gray", alpha=0.1, linestyle='solid')
        if i >= 11:
            break

    apply_SOTA(x_name=None, y_name="loss", sort=False)
    apply_domain(x)

    plt.ylabel("loss")
    plt.xlabel("case")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

with figure("score", prefix=3):
    for i, (_, path) in enumerate(globlog()):
        x, y, log = read_file(path, x_name=None, y_name="score")
        if i <= 3:
            plt.plot(x, y, label=log["name"], color=COLORS[i], linestyle='--')
        else:
            plt.plot(x, y, label=log["name"], color="gray", alpha=0.1, linestyle='solid')
        if i >= 11:
            break

    apply_SOTA(x_name=None, y_name="score", sort=False)
    apply_domain(x)

    plt.ylabel("score")
    plt.xlabel("case")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

with figure("power", prefix=4):
    # FIXME: color just top 3 --> brutforce/jfa/special!
    for i, (_, path) in enumerate(globlog()):
        x, y, log = read_file(path, x_name=None, y_name="score")
        y = sorted(y)
        if i <= 3:
            plt.plot(x, y, label=log["name"], color=COLORS[i], linestyle='--')
        else:
            plt.plot(x, y, label=log["name"], color="gray", alpha=0.1, linestyle='solid')
        if i >= 11:
            break

    apply_SOTA(x_name=None, y_name="score", sort=True)    
    # apply_domain(x) ???

    plt.ylabel("score")
    plt.xlabel("case (unordered)")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))


# FIXME: 3d wykres?

