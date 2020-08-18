import numpy as np
import numba as nb
from numba import njit, prange

# more fancy cases

@njit(parallel=True, fastmath=True)
def gen_uniform(shape, num=0):
    dhash = nb.typed.Dict.empty(key_type=nb.types.int64, value_type=nb.types.b1,)
    pts = np.empty((int(num) + 1, 2), np.int32)

    C, E, i = max(shape), int(num) + 1, 0
    while E > 0:
        for _ in range(E):
            x = np.random.randint(0, shape[0])
            y = np.random.randint(0, shape[1])
            checksum = x * C + y
            s = 1
            if checksum not in dhash:
                pts[i] = x, y
                dhash[checksum] = True
                i += 1
                s = -1
            E += s
    return pts[0 : int(num)], np.arange(1, int(num) + 1, 1).astype(np.uint32)

