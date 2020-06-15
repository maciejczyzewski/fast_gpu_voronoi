import time
import math

from .debug import save, DEBUG
from .utils import do_boxes, do_split

import pyopencl as cl
import numpy as np

# FIXME: szybsza metryka + rozne do testu
# FIXME: optymalizacje GPU (triki z __local)
# FIXME: przepisac na Cython?

# tworzymy kontekst i preparujemy karte
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

# flagi do obslugi pamieci
mf = cl.mem_flags

oo = 6666666666 # definicja nieskonczonosci

# FIXME: [jfa_star] zrobic 2d tablice (rozdzielczosc -> rozny seed)/error
#                   znalesc w ten sposob optymalna baze dla Log* (plynny error)

def load_prg(name):
    import os
    name = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cl", name);
    with open(name) as f:
        prg = cl.Program(ctx, f.read()).build()
    return prg

class algorithm:
    def __init__(self, x_size, y_size, pts, ids, **kwargs):
        """Inicjalizuje algorytm instancja oraz szykuje GPU."""
        self.x_size = x_size
        self.y_size = y_size
        self.pts = pts
        self.ids = ids

        self.prep_memory()
        self.prep_draw()
        self.prep_gpu()
        self.prg_noise = load_prg("noise.cl")
        
        self.dict = kwargs
        self.prepare_prg()
        
    def prepare_prg(self):
        pass

    def prep_memory(self):
        """Przygotowanie pamieci."""
        # print("[PREP_MEMORY]")
        self.M, self.P = do_boxes(self.x_size, self.y_size)

    def prep_draw(self):
        """Narysowanie przykladu."""
        # print("[PREP_DRAW]")
        for i in range(0, len(self.pts)):
            self.M[self.pts[i][0], self.pts[i][1]] = self.ids[i]
            self.P[self.pts[i][0], self.pts[i][1]] = self.pts[i]

    def prep_gpu(self):
        """Przygotowanie dla GPU."""
        # print("[PREP_GPU]")
        self.M, self.P1, self.P2 = \
            do_split(self.M, self.P)

    def run(self):
        """Algorytm Voronoi-a."""
        T1 = time.time_ns()
        ################################################
        save(self.M, self.x_size, self.y_size, prefix="input")
        ################################################
        self.core()
        T2 = time.time_ns()
        return (T2 - T1) / 10**9

    def core(self):
        """Implementacja."""
        pass

    def transferToGPU(self):
        self.M_g = cl.Buffer(ctx, mf.COPY_HOST_PTR | mf.READ_WRITE, hostbuf=self.M)  # x*y*1
        self.P1_g = cl.Buffer(
            ctx,
            mf.READ_WRITE | mf.COPY_HOST_PTR,
            hostbuf=self.P1)  # x*y*1 a
        self.P2_g = cl.Buffer(
            ctx,
            mf.READ_WRITE | mf.COPY_HOST_PTR,
            hostbuf=self.P2)  # x*y*1 a

        self.M_o = cl.Buffer(ctx, mf.READ_WRITE, self.M.nbytes)
        self.P1_o = cl.Buffer(ctx, mf.READ_WRITE, self.P1.nbytes)
        self.P2_o = cl.Buffer(ctx, mf.READ_WRITE, self.P2.nbytes)

    def transferFromGPU(self):
        cl.enqueue_copy(queue, self.M, self.M_g)
        cl.enqueue_copy(queue, self.P1, self.P1_g)
        cl.enqueue_copy(queue, self.P2, self.P2_g)

    def swap(self):
        self.M_g, self.M_o = self.M_o, self.M_g
        self.P1_g, self.P1_o = self.P1_o, self.P1_g
        self.P2_g, self.P2_o = self.P2_o, self.P2_g

    def apply_noise(self):
        """Arrays M, P should be on GPU"""
        self.NOISE = np.random.randint(
            len(self.pts), size=self.x_size * self.y_size, dtype=np.uint32)
        self.PTS_g = cl.Buffer(ctx, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf=self.pts)
        self.IDS_g = cl.Buffer(ctx, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf=self.ids)
        self.NOISE_g = cl.Buffer(ctx, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf=self.NOISE)
        self.prg_noise.noise(queue, self.M.shape, None,
                    self.M_g, self.P1_g, self.P2_g, self.PTS_g, self.IDS_g, self.NOISE_g, self.M_o, self.P1_o, self.P2_o,
                    np.int32(self.x_size), np.int32(self.y_size))
        self.swap()
    def save(self, pref):
        if DEBUG:
            self.transferFromGPU()
            save(self.M, self.x_size, self.y_size, prefix=pref)

class Brute(algorithm):
    def prepare_prg(self):
        self.prg = load_prg("brute.cl")
    def core(self):
        self.PTS_g = cl.Buffer(ctx, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf=self.pts)
        self.IDS_g = cl.Buffer(ctx, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf=self.ids)
        self.M_g = cl.Buffer(ctx, mf.WRITE_ONLY, self.M.nbytes)
        self.P1_g = cl.Buffer(ctx, mf.WRITE_ONLY, self.P1.nbytes)
        self.P2_g = cl.Buffer(ctx, mf.WRITE_ONLY, self.P2.nbytes)
        self.prg.Brute(queue, self.M.shape, None,
          self.PTS_g, self.IDS_g, self.M_g, self.P1_g, self.P2_g, 
          np.int32(self.pts.shape[0]), np.int32(self.x_size), np.int32(self.y_size))
        self.save("jfa")
        self.transferFromGPU()

class JFA(algorithm):
    def prepare_prg(self):
        self.prg = load_prg("jfa.cl")

    def core(self):
        """JFA: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.101.8568&rep=rep1&type=pdf"""
        self.transferToGPU()
        for factor in range(1, +oo, 1):
            f = math.ceil(max(self.x_size, self.y_size) / (2**(factor)))
            print("[step] = {}".format(f))
            self.prg.JFA(queue, self.M.shape, None,
              self.M_g, self.P1_g, self.P2_g, self.M_o, self.P1_o, self.P2_o,
              np.int32(self.x_size), np.int32(self.y_size), np.int32(f))
            self.swap()
            ################################################
            self.save("jfa")
            ################################################
            if f <= 1:
                break
        self.transferFromGPU()

class JFA_mod(algorithm):
    def prepare_prg(self):
        self.prg = load_prg("jfa.cl")
    def core(self):
        """JFA: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.101.8568&rep=rep1&type=pdf"""
        self.transferToGPU()
        self.apply_noise()
        import math
        f = 2**int(math.log(max(self.x_size, self.y_size), 2))
        print(math.log(2, max(self.x_size, self.y_size)))
        while True:
            print("[step] = {}".format(f))
            self.prg.JFA(queue, self.M.shape, None,
              self.M_g, self.P1_g, self.P2_g, self.M_o, self.P1_o, self.P2_o,
              np.int32(self.x_size), np.int32(self.y_size), np.int32(f))
            self.swap()
            ################################################
            self.save("jfa")
            ################################################
            if f <= 1:
                break
            f/=2
        self.transferFromGPU()

class JFA_test(algorithm):
    """
    options:
        * method: square/circle/circle_random
        * step_size: Integer
        * step_as_power: True/False
        * noise: True/False
    """
    def prepare_prg(self):
        self.prg_square = load_prg("jfa.cl")
        self.prg_circle = load_prg("jfa_star.cl")
        self.prg_random_circle = load_prg("jfa_circle.cl")
        self.prg_regular_circle = load_prg("jfa_regular.cl")
    def core(self):
        self.transferToGPU()
        if self.dict.get("noise", False):
            self.apply_noise()

        base = self.dict.get("step_size", 2)
        d = max(self.x_size, self.y_size)
        steps = []
        if self.dict.get("step_as_power", False):
            f = 1
            while f < d:
                steps.append(f)
                f*=base
            steps=reversed(steps)
        else:
            f = 1
            while True:
                steps.append(d/f)
                if steps[-1] <= 1:
                    break
                f*=base

        # steps = [16, 8, 4, 2, 1, 2, 1]

        method = self.dict.get("method", "square")
        if method == "circle":
            alg=self.prg_circle
        elif method == "random_circle":
            alg=self.prg_random_circle
            angle = np.random.uniform(0, 2*np.pi, size=256)
            angle.astype(np.float32)
            cv = np.cos(angle, dtype=np.float32)
            sv = np.sin(angle, dtype=np.float32)
            self.cos_val = cl.Buffer(ctx, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf=cv)
            self.sin_val = cl.Buffer(ctx, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf=sv)
        elif method == "regular_circle":
            alg=self.prg_regular_circle
            angle = np.array([np.pi*i/6.0 for i in range(12)])
            angle.astype(np.float32)
            cv = np.cos(angle, dtype=np.float32)
            sv = np.sin(angle, dtype=np.float32)
            self.cos_val = cl.Buffer(ctx, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf=cv)
            self.sin_val = cl.Buffer(ctx, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf=sv)
        else:
            alg=self.prg_square

        for step in steps:
            if method == "random_circle":
                offset = np.random.randint(0, self.x_size*self.y_size)
                alg.JFA(queue, self.M.shape, None,
                  self.M_g, self.P1_g, self.P2_g, self.M_o, self.P1_o, self.P2_o,
                  self.cos_val, self.sin_val, np.int32(offset),
                  np.int32(self.x_size), np.int32(self.y_size), np.int32(step))
            elif method == "regular_circle":
                alg.JFA(queue, self.M.shape, None,
                  self.M_g, self.P1_g, self.P2_g, self.M_o, self.P1_o, self.P2_o,
                  self.cos_val, self.sin_val,
                  np.int32(self.x_size), np.int32(self.y_size), np.int32(step))
            else:
                alg.JFA(queue, self.M.shape, None,
                  self.M_g, self.P1_g, self.P2_g, self.M_o, self.P1_o, self.P2_o,
                  np.int32(self.x_size), np.int32(self.y_size), np.int32(step))
            self.swap()
            self.save("jfa_test")
        self.transferFromGPU()

class JFA_star(algorithm):
    def prepare_prg(self):
        self.prg = load_prg("jfa_star.cl")
        
    def core(self):
        """JFA*/JFA_star: szybka heurystyczna wersja JFA+1.

        Zalety:

            - zlozonosc O(log*n) gdzie n to ilosc seedow (czyli szybsze niz
              orginalne O(log p) gdzie p to dlugosc boku kwadratu w pixelach)

            - algorytm na wejscie akceptuje dowolny rozmiarow obrazkow (nie
              tylko kwadraty o boku dlugosci p bedacych potega dwojki)

        Zmiany:

            - algorytm aplikuje maske z szumem (algorytm teraz dziala jako
              reduktur szumu/korekcji pixeli), dajac przyblizony wynik juz po
              pierwszej iteracji (klasyczne JFA nalezy wykonac do konca aby miec
              voronoi-a)

            - selekcja pixeli jest okregiem zlozonych z 12 punktow
              (klasyczny JFA kwadrat zlonony z 9 punktow)

            - krok jest podstawa trojki, a nie dwojki
        """

        def LogStar(n):
            if n <= 1:
                return 0
            if 1 < n and n <= 8:
                return 1
            if 8 < n and n <= 64:
                return 2
            if 64 < n and n <= 1024:
                return 3
            if 1024 < n and n <= 65536:
                return 4
            if 65536 < n:
                return 5

        # https://en.wikipedia.org/wiki/Iterated_logarithm
        n = LogStar(len(self.pts))
        print("-------------------------> pts={} => {} LogStar"
              .format(len(self.pts), n))

        self.transferToGPU()
        self.apply_noise()

        i = 0
        for factor in range(1, +oo, 1):
            i += 1
            f = math.ceil(max(self.x_size, self.y_size) / (3**(factor)))
            print("[step] = {}".format(f))
            self.prg.JFA(queue, self.M.shape, None,
              self.M_g, self.P1_g, self.P2_g, self.M_o, self.P1_o, self.P2_o,
              np.int32(self.x_size), np.int32(self.y_size), np.int32(f))
            self.swap()
            ################################################
            self.save("jfa_star")
            ################################################
            if f <= 1 or i == n + 1:
                break

        f = 1 # tak jak JFA+1
        print("[step] = {}".format(f))
        self.prg.JFA(queue, self.M.shape, None,
              self.M_g, self.P1_g, self.P2_g, self.M_o, self.P1_o, self.P2_o,
              np.int32(self.x_size), np.int32(self.y_size), np.int32(f))
        self.swap()
        ################################################
        self.save("jfa_star")
        self.transferFromGPU()
