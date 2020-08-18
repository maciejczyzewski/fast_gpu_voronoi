import numpy as np
from numba import njit, prange
from PIL import Image

from . import MEMORY, Config

# FIXME: add plots?
# FIXME: save for later / results?

@njit(parallel=True, fastmath=True)
def mat2d_to_rgb(img, colors=256):
    dcolors = []
    for i in range(0, colors):
        h = i * 2147483647 + 131071
        dcolors.append([h % 255, (h % 8191) % 255, (h % 524287) % 255])
    img = np.expand_dims(img, axis=2)
    img = np.concatenate((img, np.zeros((img.shape[0], img.shape[1], 2))), axis=2)
    for x in prange(0, img.shape[0]):
        for y in prange(0, img.shape[1]):
            img[x, y] = dcolors[int(img[x, y][0]) % colors]
    return img

def save(M, force=False, prefix=""):
    if Config.DEBUG or force:
        print("======== \033[92mSAVE {}\033[m ========".format(MEMORY.debug_iterator))

        M_color = mat2d_to_rgb(M)
        MEMORY.debug_iterator += 1

        if prefix != "":
            name = "__{}_{}_{}.png".format(MEMORY.debug_iterator, prefix, MEMORY.debug_name)
        else:
            name = "__{}_{}.png".format(MEMORY.debug_iterator, MEMORY.debug_name)

        im = Image.fromarray(np.uint8(M_color), "RGB")
        im = im.resize((512, 512), Image.NEAREST)  # FIXME
        im.save(name, "PNG")
