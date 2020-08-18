import os
import math
import numpy as np
from . import MEMORY
from .bench import bench
from .instance import Instance
import pyopencl as cl
from .debug import save
from timeit import default_timer as timer

mf = cl.mem_flags
oo = 16776832

# FIXME: (pyopencl) buffer vs image
# ----> https://milania.de/blog/Buffer_vs._image_performance_for_applying_filters_to_an_image_pyramid_in_OpenCL
# ----> https://gist.github.com/likr/3735779
# ----> https://www.slideshare.net/PaulChao/gpu-79586551

"""
        # === INPUT ===
        #mat2d_in = cl.Buffer(MEMORY.ctx, mf.COPY_HOST_PTR | mf.READ_WRITE,
        #                     hostbuf=mat2d_flatten)
        x.mat2d = x.mat2d.astype(float)
        dev_image_format = cl.ImageFormat(cl.channel_order.RGB,
                                          cl.channel_type.FLOAT)
        mat2d_in = cl.Image(MEMORY.ctx, mf.COPY_HOST_PTR | mf.READ_WRITE,
                            dev_image_format,
                            x.mat2d.shape,
                            hostbuf=x.mat2d)

        #print(mat2d_in)
        #sys.exit()
        print("---------------------------------> INIT DONE, 1")

        # === OUTPUT ===
        #mat2d_out = cl.Buffer(MEMORY.ctx, mf.READ_WRITE, mat2d_flatten.nbytes)
        mat2d_out = cl.Image(MEMORY.ctx, mf.READ_WRITE,
                            dev_image_format,
                            x.mat2d.shape)

        print("---------------------------------> INIT DONE")
"""

def load_prg(name):
    name = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "kernels", name)
    with open(name) as f:
        prg = cl.Program(MEMORY.ctx, f.read()).build()
    return prg

def step_power2(shape):
    steps = []
    for factor in range(1, +oo, 1):
        f = math.ceil(max(shape) / (2**(factor)))
        steps.append(f)
        if f <= 1:
            break
    return steps

class Algorithm_Brutforce():
    name = "BRUTFORCE"

    def __init__(self):
        self.kernel = load_prg("brutforce.cl")

    def do(self, x: Instance):
        T1_all = timer()
        mat2d_flatten = x.mat2d.flatten().astype(np.uint32)

        # === INPUT ===
        points_in = cl.Buffer(MEMORY.ctx, mf.COPY_HOST_PTR |
                              mf.READ_ONLY, hostbuf=x.points)
        seeds_in = cl.Buffer(MEMORY.ctx, mf.COPY_HOST_PTR |
                             mf.READ_ONLY, hostbuf=x.seeds)

        # === OUTPUT ===
        mat2d_out = cl.Buffer(MEMORY.ctx, mf.READ_WRITE, mat2d_flatten.nbytes)

        # === RUN ===
        T1 = timer()
        event = self.kernel.fn(MEMORY.queue, (x.shape[0],x.shape[1], 1), None,
                       points_in,
                       seeds_in,
                       mat2d_out,
                       np.int32(x.shape[0]),
                       np.int32(x.shape[1]),
                       np.int32(x.points.shape[0]),
                       )
        event.wait()
        T2 = timer()

        mat2d_new = np.empty_like(mat2d_flatten)
        event = cl.enqueue_copy(MEMORY.queue, mat2d_new, mat2d_out)
        event.wait()
        x.mat2d = mat2d_new.reshape(x.shape[0], x.shape[1], 3)

        # save(x.mat2d[:, :, 0])
        T2_all = timer()
        # [FIXME]
        # points_in.release()
        # seeds_in.release()
        # mat2d_out.release()
        # del mat2d_new
        return x, T2-T1, T2_all-T1_all


class Algorithm_JFA():
    name = "JFA"

    def __init__(self):
        self.kernel = load_prg("jfa.cl")

    def do(self, x: Instance):
        T1_all = timer()
        mat2d_flatten = x.mat2d.flatten().astype(np.uint32)

        # === INPUT ===
        mat2d_in = cl.Buffer(MEMORY.ctx, mf.COPY_HOST_PTR | mf.READ_WRITE,
                             hostbuf=mat2d_flatten)

        # === OUTPUT ===
        mat2d_out = cl.Buffer(MEMORY.ctx, mf.READ_WRITE, mat2d_flatten.nbytes)

        # === RUN ===
        steps = step_power2(x.shape)

        T_SUM = 0
        for step in steps:
            T1 = timer()
            event = self.kernel.fn(MEMORY.queue, (x.shape[0],x.shape[1], 1), None,
                           mat2d_in,
                           mat2d_out,
                           np.int32(x.shape[0]),
                           np.int32(x.shape[1]),
                           np.int32(step)
                           )
            event.wait()
            T2 = timer()
            T_SUM += T2-T1
            mat2d_in, mat2d_out = mat2d_out, mat2d_in
        mat2d_in, mat2d_out = mat2d_out, mat2d_in

        mat2d_new = np.empty_like(mat2d_flatten)
        event = cl.enqueue_copy(MEMORY.queue, mat2d_new, mat2d_out)
        event.wait()
        x.mat2d = mat2d_new.reshape(x.shape[0], x.shape[1], 3)

        # save(x.mat2d[:, :, 0])
        T2_all = timer()
        return x, T_SUM, T2_all-T1_all

################################################################################
# XXX: JFA_FUSION: to speedup kernel runtime?
################################################################################

class Algorithm_JFA_Fusion():
    name = "JFA_FUSION"

    def __init__(self):
        self.kernel = load_prg("jfa_fusion.cl")

    def do(self, x: Instance):
        T1_all = timer()
        mat2d_flatten = x.mat2d.flatten().astype(np.uint32)

        # === INPUT ===
        mat2d_in = cl.Buffer(MEMORY.ctx, mf.COPY_HOST_PTR | mf.READ_WRITE,
                             hostbuf=mat2d_flatten)

        # === RUN ===
        # FIXME: move to function
        ################################################
        steps = step_power2(x.shape)
        steps += steps  # FIXME: repair filter
        steps = np.array(steps, dtype=np.uint32)
        ################################################

        steps_in = cl.Buffer(MEMORY.ctx, mf.COPY_HOST_PTR | mf.READ_WRITE,
                             hostbuf=steps)

        T1 = timer()
        event = self.kernel.fn(MEMORY.queue, (x.shape[0],x.shape[1], 1),
                       None,
                       mat2d_in,
                       steps_in,
                       np.int32(x.shape[0]),
                       np.int32(x.shape[1]),
                       np.int32(len(steps))
                       )
        event.wait()
        T2 = timer()

        mat2d_new = np.empty_like(mat2d_flatten)
        event = cl.enqueue_copy(MEMORY.queue, mat2d_new, mat2d_in)
        event.wait()
        x.mat2d = mat2d_new.reshape(x.shape[0], x.shape[1], 3)

        # save(x.mat2d[:, :, 0], prefix=self.name)

        T2_all = timer()
        return x, T2-T1, T2_all-T1_all
