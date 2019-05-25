"""JFA.
Usage:
  main.py test <algorithm> [--x=<num>] [--y=<num>] [--seeds=<num>]
  main.py 2d | 3d

Options:
  -h --help      Show this screen.
  --version      Show version.
  --x=<num>      X axis in pixels [default: 720].
  --y=<num>      Y axis in pixels [default: 720].
  --seeds=<num>  Number of seeds [default: 2000].
"""
from docopt import docopt

import numpy as np
np.random.seed(136698)

from jfa   import JFA, JFA_plus, JFA_plus_inplace, JFA_star
from utils import do_sample
from debug import save

# export PYOPENCL_CTX='0:1'
# https://www.utdallas.edu/~xxg061000/GPU-CVT.pdf

# python3 main.py test JFA --x=100 --y=200 --seeds=2000
# python3 main.py test JFA --x=720 --y=720 --seeds=2000
# python3 main.py test JFA --x=512 --y=512 --seeds=10000
# python3 main.py test JFA_star --x=512 --y=512 --seeds=44

##########################################################################


def run_test(algorithm, x, y, seeds):
    print("[RUN_TEST] algorithm={} x={} y={} seeds={}"
          .format(algorithm, x, y, seeds))

    sample = do_sample(x=int(x), y=int(y), seeds=int(seeds))
    algos = [JFA, JFA_plus, JFA_star]

    for var in algos:
        if var.__name__ == algorithm:
            alg = var(*sample)

    result = alg.run()
    print("TIME", result)

    save(alg.M, alg.x_size, alg.y_size, force=True)

##########################################################################


def run_3d():
    print("[RUN_3D]")

    from tqdm import tqdm
    from collections import defaultdict
    import matplotlib.pyplot as plt

    import os.path
    import pickle

    repeat = 2
    # 1000 ---------> potrzebne punkty
    seeds = 64  # FIXME: 3 wymiarowy plot (seeds, size, time)
    # algos = [JFA, JFA_preplus, JFA_plus]
    algos = [JFA, JFA_plus, JFA_star]
    #seeds = [8, 64, 1024, 65536]
    seeds = [8, 64, 1024, 65536]

    if not (os.path.isfile("X.pickle") and
            os.path.isfile("Y.pickle") and
            os.path.isfile("Z.pickle")):
        X = defaultdict(list)
        Y = defaultdict(list)
        Z = defaultdict(list)
        for seed in seeds:
            for size in tqdm(range(64, 1024, 16)):
                sample = do_sample(x=size, y=size, seeds=seed)
                for algo in algos:
                    arr = []
                    for _ in range(repeat):
                        alg = algo(*sample)
                        arr.append(alg.run())
                    res = sum(arr) / len(arr)
                    X[algo.__name__].append(size)
                    Y[algo.__name__].append(res)
                    Z[algo.__name__].append(seed)

        with open('X.pickle', 'wb') as handle:
            pickle.dump(X, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('Y.pickle', 'wb') as handle:
            pickle.dump(Y, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('Z.pickle', 'wb') as handle:
            pickle.dump(Z, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open('X.pickle', 'rb') as handle:
            X = pickle.load(handle)
        with open('Y.pickle', 'rb') as handle:
            Y = pickle.load(handle)
        with open('Z.pickle', 'rb') as handle:
            Z = pickle.load(handle)

    print(X)
    print(Y)
    print(Z)

    from mpl_toolkits import mplot3d
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.view_init(60, 35)

    from matplotlib import cm

    colors_1 = [(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)]
    colors_2 = [(1, 0, 0, 0.05), (0, 1, 0, 0.05), (0, 0, 1, 0.05)]

    import matplotlib.tri as mtri

    i = 0
    for algo in algos:
        name = algo.__name__
        print(name)
        #plt.plot(X[name], Y[name], label=name)
        #ax.scatter3D(X[name], Y[name], Z[name], cmap='binary')

        ax.scatter3D(X[name], Y[name], np.log10(Z[name]), color=colors_1[i],
                     label=name)

        triangles = mtri.Triangulation(X[name], Y[name]).triangles
        triang = mtri.Triangulation(X[name], Y[name], triangles)
        ax.plot_trisurf(
            triang,
            np.log10(
                Z[name]),
            linewidth=0.1,
            lw=0.2,
            shade=False,
            color=colors_2[i])
        i += 1

    ax.set_xlabel('size [pixels]')
    ax.set_ylabel('time [seconds]')
    ax.set_zlabel('seeds [number]')
    # ax.set_zscale('symlog')
    plt.autoscale(enable=True, axis='z')
    ax.set_zticks([1, 2, 3, 5])
    ax.set_zticklabels(seeds)
    # ax.zaxis.set_rotate_label(False)
    #ax.set_zticks([], seeds)

    ax.view_init(50, -35)

    #labels = [8, 64, 1024, 65536]
    #zticks = [1,2,3,4]
    #plt.zticks(zticks, labels)

    plt.legend()
    #plt.show()
    fig.savefig("figure_3d.png", dpi=300, bbox_inches='tight', pad_inches=0.4)


##########################################################################


def run_2d():
    print("[RUN_2D]")

    from tqdm import tqdm
    from collections import defaultdict
    import matplotlib.pyplot as plt

    import os.path
    import pickle

    repeat = 2
    # 1000 ---------> potrzebne punkty
    seeds = 64  # FIXME: 3 wymiarowy plot (seeds, size, time)
    # algos = [JFA, JFA_preplus, JFA_plus]
    algos = [JFA, JFA_plus, JFA_star]
    #seeds = [8, 64, 1024, 65536]
    seeds = [8, 64, 1024, 65536]

    if not (os.path.isfile("X.pickle") and
            os.path.isfile("Y.pickle") and
            os.path.isfile("Z.pickle")):
        X = defaultdict(list)
        Y = defaultdict(list)
        Z = defaultdict(list)
        for seed in seeds:
            for size in tqdm(range(64, 1024, 16)):
                sample = do_sample(x=size, y=size, seeds=seed)
                for algo in algos:
                    arr = []
                    for _ in range(repeat):
                        alg = algo(*sample)
                        arr.append(alg.run())
                    res = sum(arr) / len(arr)
                    X[algo.__name__].append(size)
                    Y[algo.__name__].append(res)
                    Z[algo.__name__].append(seed)

        with open('X.pickle', 'wb') as handle:
            pickle.dump(X, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('Y.pickle', 'wb') as handle:
            pickle.dump(Y, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('Z.pickle', 'wb') as handle:
            pickle.dump(Z, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open('X.pickle', 'rb') as handle:
            X = pickle.load(handle)
        with open('Y.pickle', 'rb') as handle:
            Y = pickle.load(handle)
        with open('Z.pickle', 'rb') as handle:
            Z = pickle.load(handle)

    for seed in seeds:
        print("--- seed={} ----".format(seed))
        Xa = defaultdict(list)
        Ya = defaultdict(list)
        Za = defaultdict(list)
        for algo in algos:
            for i in range(0, len(Z[algo.__name__])):
                if Z[algo.__name__][i] == seed:
                    x = X[algo.__name__][i]
                    y = Y[algo.__name__][i]
                    z = Z[algo.__name__][i]
                    Xa[algo.__name__].append(x)
                    Ya[algo.__name__].append(y)
                    Za[algo.__name__].append(z)
                    # print(x,y,z)
        print(Xa)

        #ax.set_xlabel('size [pixels]')
        #ax.set_ylabel('time [seconds]')
        #ax.set_zlabel('seeds [number]')
        colors_1 = [(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)]
        fig = plt.figure(0)
        ax = fig.add_subplot(111)
        ax.set_title("seeds={}".format(seed))
        ax.set_ylim([0, 0.35])
        k = 0
        for algo in algos:
            ax.plot(Xa[algo.__name__], Ya[algo.__name__], label=algo.__name__,
                    color=colors_1[k])
            # plt.text(Xa[algo.__name__][-1], Ya[algo.__name__][-1], \
            #        'seeds {}'.format(seed))
            k += 1
        plt.legend()
        # set title
        fig.savefig("figure_seed_{}.png".format(
            seed), dpi=300, bbox_inches='tight')
        fig.clf()

        print("---------------")

##########################################################################
# MAIN
##########################################################################


if __name__ == '__main__':
    arguments = docopt(__doc__, version='Maciej A. Czyzewski / 1.0')
    if arguments["test"]:
        run_test(arguments["<algorithm>"],
                 arguments["--x"], arguments["--y"],
                 arguments["--seeds"])
    if arguments["2d"]:
        run_2d()
    if arguments["3d"]:
        run_3d()
