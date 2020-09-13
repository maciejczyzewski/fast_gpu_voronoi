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


def fill_between(X, Y, color="blue", alpha=0.05, factor=1):
    sigma = factor * np.array(Y).std(axis=0)  # ls = '--'
    ax.fill_between(X, Y + sigma, Y - sigma, facecolor=color, alpha=alpha)

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

def globlog():
    vec = []
    for path in glob("results/log/*.json"):
        x = float(path.replace(".json", "").replace("results/log/", ""))
        vec.append([x, path])
    return sorted(vec)[::-1]

### FIGURE (1): underfitting ###

# FIXME: funny name?

def apply_domain(x=None):
    # FIXME: dodac umiejetnosc pomijania niektorych? jak jest gesto?
    path = "results/domain.json"
    with open(path, "r") as jsonfile:
        json_txt = jsonfile.read()
        domain = json.loads(json_txt)
    pprint(domain)
    domain_str = []
    for d in domain:
        q = round(d['num']/(d['shape'][0]*d['shape'][1]), 4)
        domain_str.append(f"{d['shape'][0]}x{d['shape'][1]}\nseeds={d['num']}\n$\\rho$={q}")
        # domain_str.append(f"{d['shape'][0]}x{d['shape'][1]}\n{d['num']}")
    plt.xticks(x, domain_str, rotation='horizontal', fontsize=2)

COLORS = ['g', 'b', 'm', 'orange']

with figure("time", prefix=1):
    # FIXME: gray -----> alpha change on i?
    for i, (_, path) in enumerate(globlog()):
        x, y, log = read_file(path, x_name=None, y_name="time")
        if i <= 3:
            plt.plot(x, y, label=log["name"], color=COLORS[i], linestyle='--')
        else:
            plt.plot(x, y, label=log["name"], color="gray", alpha=0.1, linestyle='solid')

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

    apply_SOTA(x_name=None, y_name="score", sort=True)    
    apply_domain(x)

    plt.ylabel("score")
    plt.xlabel("case (unordered)")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

"""
with figure("test accuracy (without augment)", prefix=1):
    x1, y1 = read_file("decay=1e-4/log_EfficientNet_batchboost_1")
    plt.plot(x1, y1, label="boostbatch (alpha=1.0)", color="darkred")

    x1, y1 = read_file("decay=1e-4/log_EfficientNet_batchboost_3")
    plt.plot(x1, y1, label="boostbatch (alpha=0.4)", color="red")

    x2, y2 = read_file("decay=1e-4/log_EfficientNet_mixup_1")
    plt.plot(x2, y2, label="mixup (alpha=1.0)", color="darkblue")

    x2, y2 = read_file("decay=1e-4/log_EfficientNet_mixup_3")
    plt.plot(x2, y2, label="mixup (alpha=0.4)", color="blue")

    x3, y3 = read_file("decay=1e-4/log_EfficientNet_baseline_13")
    plt.plot(x3, y3, label="baseline", color="black")

    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    # plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

with figure("loss train (without augment)", prefix=1):
    x1a, y1a = read_file("decay=1e-4/log_EfficientNet_batchboost_1", col=1)
    plt.plot(x1a, y1a, label="boostbatch (alpha=1.0)", color="darkred")

    x1b, y1b = read_file("decay=1e-4/log_EfficientNet_batchboost_3", col=1)
    plt.plot(x1b, y1b, label="boostbatch (alpha=0.4)", color="red")

    fill_between(x1a,
                 np.mean([y1a, y1b], axis=0),
                 color="red",
                 factor=1,
                 alpha=0.1)

    x2a, y2a = read_file("decay=1e-4/log_EfficientNet_mixup_1", col=1)
    plt.plot(x2a, y2a, label="mixup (alpha=1.0)", color="darkblue")

    x2b, y2b = read_file("decay=1e-4/log_EfficientNet_mixup_3", col=1)
    plt.plot(x2b, y2b, label="mixup (alpha=0.4)", color="blue")

    fill_between(x2a,
                 np.mean([y2a, y2b], axis=0),
                 color="blue",
                 factor=1,
                 alpha=0.1)

    x3, y3 = read_file("decay=1e-4/log_EfficientNet_baseline_13", col=1)
    plt.plot(x3, y3, label="baseline", color="black")

    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

### FIGURE (2): overfitting (compirason to mixup) ###

with figure("test accuracy (with augment)", prefix=2):
    x1a, y1a = read_file("decay=1e-5/log_EfficientNet_batchboost_2")
    plt.plot(x1a, y1a, label="boostbatch (alpha=1.0)", color="darkred")

    x1b, y1b = read_file("decay=1e-5/log_EfficientNet_batchboost_4")
    plt.plot(x1b, y1b, label="boostbatch (alpha=0.4)", color="red")

    fill_between(x1a,
                 np.mean([y1a, y1b], axis=0),
                 color="red",
                 factor=0.5,
                 alpha=0.1)

    x2a, y2a = read_file("decay=1e-5/log_EfficientNet_mixup_2")
    plt.plot(x2a, y2a, label="mixup (alpha=1.0)", color="darkblue")

    x2b, y2b = read_file("decay=1e-5/log_EfficientNet_mixup_4")
    plt.plot(x2b, y2b, label="mixup (alpha=0.4)", color="blue")

    fill_between(x2a,
                 np.mean([y2a, y2b], axis=0),
                 color="blue",
                 factor=0.5,
                 alpha=0.1)

    # x3, y3 = read_file("decay=1e-5/log_EfficientNet_baseline_24")
    # plt.plot(x3, y3, label="baseline", color="black")

    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

with figure("train accuracy (with augment)", prefix=2):
    x1, y1 = read_file("decay=1e-5/log_EfficientNet_batchboost_2", col=3)
    plt.plot(x1, y1, label="boostbatch (alpha=1.0)", color="darkred")

    x1, y1 = read_file("decay=1e-5/log_EfficientNet_batchboost_4", col=3)
    plt.plot(x1, y1, label="boostbatch (alpha=0.4)", color="red")

    x2, y2 = read_file("decay=1e-5/log_EfficientNet_mixup_2", col=3)
    plt.plot(x2, y2, label="mixup (alpha=1.0)", color="darkblue")

    x2, y2 = read_file("decay=1e-5/log_EfficientNet_mixup_4", col=3)
    plt.plot(x2, y2, label="mixup (alpha=0.4)", color="blue")

    # x3, y3 = read_file("decay=1e-5/log_EfficientNet_baseline_24", col=3)
    # plt.plot(x3, y3, label="baseline", color="black")

    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    # plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
"""
