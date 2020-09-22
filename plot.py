import re
import os
import pylab
import subprocess
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import unidecode
import json, csv

from glob import glob
from pprint import pprint

oo = 11111111

def slugify(text):
    text = unidecode.unidecode(text).lower()
    text = re.sub(r"[\W_]+", "-", text)
    if text[-1] == "-":
        return text[0:-1]
    return text

#####################################################
# FIXME: runner script???? to make all experiments???
#           ------------------> AND PLOT THEM????????
#####################################################

# FIXME: Config class?
PATH_RESULTS = "results/" #"results_small_low/"
PATH_FIGURES = f"figures/{slugify(PATH_RESULTS)}/"
IS_LEGEND_ON_FIGURE = False
FIGURE_SIZE = (5, 3) # (12.5, 3)

if PATH_RESULTS is None:
    PATH_RESULTS = "results/"
if PATH_FIGURES is None:
    PATH_FIGURES = "figures/default/"

###############################
# [IDEAS]
# FIXME: 3d
# FIXME: wyswietlaj dla wszystkich z tabeli plot? ale na legendzie tylko najl.?
###############################

################################################################################

def call(cmd):
    print(
        f"\x1b[7;37;40m{cmd}\x1b[0m\n" + subprocess.check_output(
            cmd, stderr=subprocess.STDOUT, shell=True).decode("utf-8"),
        end="",
    )

call(f"mkdir -p {PATH_FIGURES}")

# (1) better style
plt.style.use(["science", "ieee"])
plt.rcParams.update({"text.usetex": True})

fig, ax = plt.subplots()
fig.set_size_inches(*FIGURE_SIZE)
ax.autoscale(tight=True)

# plt2 = None
class figure:
    def __init__(self, name=None, prefix=None):
        self.name = name
        self.prefix = prefix

    def __enter__(self):
        global plt2
        print("--- FIGURE ---")
        print(f"`{self.name}`")
        plt.cla()
        #ax.get_legend().remove()
        ax = plt.gca()
        ax.legend_ = None # delete legend

        plt.title(self.name)
        # plt2 = plt.twiny()

    def __exit__(self, x, y, z):
        print("--- SAVE ---")
        figure_prefix = "figure-"
        if self.prefix is not None:
            figure_prefix += f"{str(self.prefix)}-"
        fig.savefig(f"{PATH_FIGURES}/{figure_prefix}{slugify(self.name)}.pdf")

def read_file(path=None, x_name=None, y_name=None):
    X, Y = [], []
    with open(path, "r") as jsonfile:
        json_txt = jsonfile.read()
        log = json.loads(json_txt)
        for i in range(len(log["score"])):
            if x_name is not None:
                X.append(max(0, float(log[x_name][i])))
            else:
                X.append(max(0, i))
            Y.append(max(0, float(log[y_name][i])))
    log["name"] = log["name"].replace("_", "")
    return X, Y, log

def apply_SOTA(x_name=None, y_name=None, sort=False):
    # for [JFA]
    path = f"{PATH_RESULTS}/jfa.json"
    x, y, log = read_file(path, x_name=x_name, y_name=y_name)
    if sort:
        y = sorted(y)
    plt.plot(x, y, label=log["name"], color="black", linestyle='solid')
    ax.fill_between(x, y, 0, facecolor='black', alpha=0.2)

    # for [BRUTEFORCE]
    if y_name == "score":
        plt.plot(x, [100]*len(x), label="bruteforce", color="red", linestyle='solid')
        ax.fill_between(x, [100]*len(x), 0, facecolor='red', alpha=0.2)
    if y_name == "loss":
        plt.plot(x, [0]*len(x), label="bruteforce", color="red", linestyle='solid')
        ax.fill_between(x, [0]*len(x), 0, facecolor='red', alpha=0.2)
    if y_name == "time":
        plt.plot(x, [1]*len(x), label="bruteforce", color="red", linestyle='solid')
        ax.fill_between(x, [1]*len(x), 0, facecolor='red', alpha=0.2)
    ax.grid(linestyle="--", linewidth=0.5, color='.25', zorder=-10, alpha=0.1)

def globlog(sota=False):
    vec = []
    for path in glob(f"{PATH_RESULTS}/log/*.json"):
        x = float(path.replace(".json", "").replace(f"{PATH_RESULTS}/log/", ""))
        if x < 0:
            continue
        vec.append([x, path])

    if sota:
        path = f"{PATH_RESULTS}/jfa.json"
        _, _, log_sota = read_file(path, x_name=None, y_name="score")
        sota_score = int(log_sota["name"].split()[0][1:-1])
        print(f"===========================> JFA (score) = {sota_score}")
        vec.append([sota_score, path])

    return sorted(vec)[::-1]

### FIGURE (1): underfitting ###

# FIXME:
# - u gory: segmenty z np. 64x64, 128x128
# - na dole: jakie `seeds` jakie `rho`

def apply_domain__classic(x=None, with_shape=True):
    path = f"{PATH_RESULTS}/domain.json"
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
    path = f"{PATH_RESULTS}/domain.json"
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

    # ax2 = ax.twiny()
    # plt2.spines["bottom"].set_position(("axes", -.1))
    # plt2.set_xlim(ax.get_xlim())
    # plt2.set_xticks(x)
    # plt2.set_xticklabels(x, rotation='horizontal', fontsize=2)
    # plt2.set_xlabel(r"density ($\rho$)")

    #ax2 = ax.secondary_xaxis("top")
    #ax2 = ax.twiny()

    domain_str = []
    for d in domain:
        q = round(d['num']/(d['shape'][0]*d['shape'][1]), 4)
        domain_str.append(f"$\\Omega$={d['num']}\n$\\rho$={q}")
    plt.xticks(x, domain_str, rotation='vertical', fontsize=2)
    #ax2.set_xticks(x)
    #ax2.set_xticklabels(domain_str, rotation='horizontal', fontsize=2)

    """
    domain_str = []
    for d in domain:
        q = round(d['num']/(d['shape'][0]*d['shape'][1]), 4)
        domain_str.append(f"$\\rho$={q}")
    ax2 = ax.secondary_xaxis("top", ticks=domain_str)
    # https://stackoverflow.com/questions/10514315/how-to-add-a-second-x-axis-in-matplotlib
    pos1 = ax.get_position() # get the original position 
    pos2 = [pos1.x0 + 0.3, pos1.y0 + 0.3,  pos1.width / 2.0, pos1.height / 2.0] 
    ax2.set_position(pos2)
    ax2.set_xticks(x)
    ax2.set_xticklabels(domain_str, rotation='horizontal', fontsize=2)
    ax2.set_xlabel(r"density ($\rho$)", labelpad=20)
    """

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

legend_saved = False
def apply_legend(force=False):
    global legend_saved
    if IS_LEGEND_ON_FIGURE:
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    elif force == True or \
        (legend_saved == False and not os.path.isfile(f'{PATH_FIGURES}/legend.png')):
        print("======================== ONCE =======================")
        figlegend = pylab.figure(figsize=(3,2))
        figlegend.legend(*ax.get_legend_handles_labels(), loc="center")
        figlegend.savefig(f'{PATH_FIGURES}/legend.png')
        del figlegend
        legend_saved = True
    #else:
    #    print("OKAY NORMAL =========================================")
    #    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

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
    
    apply_legend()

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
    
    apply_legend()

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

    apply_legend()

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

    apply_legend(force=True)

################################################################################
# TABLE
################################################################################

# FIXME: tabela nie po A x B tylko po GESTOSCIACH!!!!!!!!!!!!!!!!!!!!!!!
#        ajajaja bedzie trzeba je skulstrowac

from tabulate import tabulate

# FIXME: domain
path = f"{PATH_RESULTS}/domain.json"
with open(path, "r") as jsonfile:
    json_txt = jsonfile.read()
    domain = json.loads(json_txt)

# FIXME: print JFA

print(domain)

GROUP_COL = 2 # FIXME: mozna zredykowac ilosc kolumn
MAX_ROWS = 38
density_map = {}
ROWS = []
sota_in = False
for i, (_, path) in enumerate(globlog(sota=True)):
    if i > MAX_ROWS and sota_in == True:
        break
    x, y, log = read_file(path, x_name=None, y_name="score")
    # [DEBUG]: for Kamil
    # for j, e in enumerate(y):
    #    print("---->", domain[j], e)
    score, name = log["name"].split(' ', 1) # FIXME: only once
    score = int(score[1:-1])
    if i >= MAX_ROWS and "JFA" not in name and sota_in == False:
        continue
    # print(f"==============> {score} {name}")
    local_avg = {}
    last_shape, slow_i, cur_i = "?x?", 0, 0
    for j, e in enumerate(y):
        # FIXME: tak samo ale dla AxB
        shape = f"{domain[j]['shape'][0]}x{domain[j]['shape'][1]}"
        if last_shape != shape:
            last_shape, cur_i = shape, 0
        print("---->", shape, cur_i, e)
        if cur_i not in local_avg:
            local_avg[cur_i] = []
        local_avg[cur_i].append(e)
        ############
        # SMALL CHEAT
        if cur_i not in density_map:
            density_map[cur_i] = []
        density_map[cur_i].append(domain[j]['num']/(domain[j]['shape'][0]*domain[j]['shape'][1]))
        ############
        if slow_i+1 < GROUP_COL:
            print("SLOW")
            slow_i += 1
        else:
            print("FAST")
            cur_i += 1
            slow_i = 0
    # pprint(local_avg)
    local_row = []
    for i in range(0, +oo):
        if i not in local_avg:
            break
        avg = round(sum(local_avg[i])/len(local_avg[i]), 1)
        if avg >= 100:
            avg = "\\textbf{" + str(avg) + "}"
        # print("--------->", i, avg)
        local_row.append(avg)
    # FIXME: if > 100 BOLDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD???
    # ----------------------> COLOR BEST IN COLUMN
    if "JFA" in name:
        sota_in = True
        ROWS.append(["\cellcolor{blue!25}\\textbf{" + name + "}"] + local_row + [score])
    else:
        ROWS.append(["\longvar{" + name + "}"] + local_row + [score])
    print("\n\n")

header_row = []
for i in range(0, +oo):
    if i not in density_map:
        break
    print(density_map[i])
    avg = round(sum(density_map[i])/len(density_map[i]), 4)
    header_row.append(f"$\\rho$={avg}")

header_row = ["Algorithm"] + header_row + ["Score"]
pprint(header_row)
# pprint(ROWS)

print("=============================================")

table = ROWS
headers = header_row
tab = tabulate(table, headers=headers,
               tablefmt="latex_raw")
tab_s = str(tab)
tab_s = tab_s.replace("begin{tabular}{l", "begin{tabular}{L")

path_table = f"{PATH_FIGURES}/table.tex"
text_file = open(path_table, "w")
text_file.write(tab_s)
text_file.close()

print(tab_s)

# FIXME: 3d wykres?
