__version__ = '0.1.0'

import pyopencl as cl

# FIXME: pakiet jest jako TOOLKIT tylko (wizulizacja, error)
# ----> develop run przeniesiony do sandbox/ sandbox_auto/

# PING-PONG technique
# https://www.mathematik.uni-dortmund.de/~goeddeke/gpgpu/tutorial.html#feedback2

# FIXME: SINGLETON
class Memory:
    debug_iterator = 0
    debug_name = "debug"

    def __init__(self):
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)

MEMORY = Memory()

class Config:
    DEBUG = True
