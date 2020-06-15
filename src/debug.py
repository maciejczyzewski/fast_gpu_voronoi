import os
import imageio
import numpy as np
from tqdm import tqdm

DEBUG = False

if DEBUG:
    os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1"

debug_iterator = 0
debug_name = 'debug'

os.system("rm __*.png")

def matrix_rgb(img, colors=256):
    """Zamienia seed-y w macierzy na kolory."""
    # dodajemy 2 kanaly zeby finalnie miec [r, g, b]
    img = np.concatenate((img,
                          np.zeros((img.shape[0], img.shape[1], 2))), axis=2)
    # szukamy i usuwamy
    dcolors = {}
    for i in range(0, colors):
        h = i * 2147483647 + 131071
        dcolors[i] = [h % 255, (h % 8191) % 255, (h % 524287) % 255]
    # wybieramy kolor
    def __f(cell):
        return dcolors[int(cell[0]) % colors]
    for x in tqdm(range(0, img.shape[0])):
        for y in range(0, img.shape[1]):
            img[x, y] = __f(img[x, y])
    # XXX (wolniejsze):
    # img = np.apply_along_axis(__f, 2, img)
    return img


def save_to_file(M, prefix=""):
    """Zapisuje macierz M do pliku."""
    global debug_name, debug_iterator
    print("======== \033[92mSAVE {}\033[m ========".format(debug_iterator))
    M_color = matrix_rgb(M)
    debug_iterator += 1
    if prefix != "":
        name = '__{}_{}_{}.png'.format(debug_iterator, prefix, debug_name)
    else:
        name = '__{}_{}.png'.format(debug_iterator, debug_name)
    imageio.imwrite(name, M_color)
    # imageio.imwrite("invalid"+name, (M < 1).astype(np.float))


def save(M, x_size, y_size, force=False, prefix=""):
    """Szybki debug dowolnej postaci macierzy M."""
    if DEBUG or force:
        save_to_file(np.reshape(M, (x_size, y_size, 1)), prefix=prefix)

