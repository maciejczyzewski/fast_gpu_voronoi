from timeit import default_timer as timer
from contextlib import contextmanager

# FIXME: toolkit?
#        to sheudle experiments?

@contextmanager
def bench(msg="none", n=1):
    T1 = timer()
    for _ in range(n):
        yield
    T2 = timer()
    print(f"\033[92mtime ({msg.ljust(30)}): {round((T2-T1)/n, 4)}\033[0m")
