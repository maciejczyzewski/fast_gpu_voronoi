import os
import numpy as np
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '0'

"""
from fast_gpu_voronoi       import Instance
from fast_gpu_voronoi.jfa   import JFA_star
from fast_gpu_voronoi.debug import save

I = Instance(alg=JFA_star, x=50, y=50, \
        pts=[[ 7,14], [33,34], [27,10],
             [35,10], [23,42], [34,39]])
I.run()

save(I.M, I.x, I.y, force=True)
"""

"""
from jfa import JFA_star
from debug import save

I = Instance(alg=JFA_star, x=50, y=50, \
        pts=[[ 7,14], [33,34], [27,10],
             [35,10], [23,42], [34,39]])
I.run()

print(I.M.shape)
"""

class Instance:
    def __init__(self, alg, x, y, pts, ids=[]):
        self.alg = alg
        self.x   = x
        self.y   = y
        self.pts = np.array(pts, dtype=np.uint16)
        self.ids = np.array(ids, dtype=np.uint32)
        if ids == []:
            self.ids = np.arange(1,len(self.pts)+1,1)\
                    .astype(np.uint32)
        self.M = None

    def run(self):
        tmp = self.alg(self.x, self.y, self.pts, self.ids)
        ti = tmp.run()
        self.M_1d = tmp.M
        self.M    = np.reshape(self.M_1d, (self.x, self.y, 1))
        return ti
