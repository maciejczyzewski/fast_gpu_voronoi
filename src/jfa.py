import time
import math

from .debug import save
from .gpu import GPU_noise, GPUJFA9, GPUJFA9_noise, GPUJFA_star
from .utils import do_boxes, do_split

oo = 6666666666 # definicja nieskonczonosci

# FIXME: [jfa_star] zrobic 2d tablice (rozdzielczosc -> rozny seed)/error
#                   znalesc w ten sposob optymalna baze dla Log* (plynny error)

class algorithm:
    def __init__(self, x_size, y_size, pts, ids):
        """Inicjalizuje algorytm instancja oraz szykuje GPU."""
        self.x_size = x_size
        self.y_size = y_size
        self.pts = pts
        self.ids = ids

        self.prep_memory()
        self.prep_draw()
        self.prep_gpu()

    def prep_memory(self):
        """Przygotowanie pamieci."""
        print("[PREP_MEMORY]")
        self.M, self.P = do_boxes(self.x_size, self.y_size)

    def prep_draw(self):
        """Narysowanie przykladu."""
        print("[PREP_DRAW]")
        for i in range(0, len(self.pts)):
            self.M[self.pts[i][0], self.pts[i][1]] = self.ids[i]
            self.P[self.pts[i][0], self.pts[i][1]] = self.pts[i]

    def prep_gpu(self):
        """Przygotowanie dla GPU."""
        print("[PREP_GPU]")
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


class JFA(algorithm):
    def core(self):
        """JFA: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.101.8568&rep=rep1&type=pdf"""
        for factor in range(1, +oo, 1):
            f = math.ceil(max(self.x_size, self.y_size) / (2**(factor)))
            print("[step] = {}".format(f))
            self.M, self.P1, self.P2 = \
                GPUJFA9(self.M, self.P1, self.P2,
                        self.x_size, self.y_size, step=f)
            ################################################
            save(self.M, self.x_size, self.y_size, prefix="jfa")
            ################################################
            if f <= 1:
                break


class JFA_plus(algorithm):
    def prep_gpu(self):
        print("[PREP_GPU]")
        self.M, self.P1, self.P2 = \
            do_split(self.M, self.P)
        import numpy as np # nalezy zaaplikowac szum
        self.NOISE = np.random.randint(
            len(self.pts), size=self.x_size * self.y_size, dtype=np.uint32)
        self.M, self.P1, self.P2 = GPU_noise(self.M, self.P1, self.P2,
                                             self.pts, self.ids, self.NOISE,
                                             self.x_size, self.y_size)

    def core(self):
        """Moja modyfikacja JFA: szum i redukcja kroku."""
        for factor in range(1, +oo, 2):
            f = math.ceil(max(self.x_size, self.y_size) / (2**(factor)))
            print("[step] = {}".format(f))
            self.M, self.P1, self.P2 = \
                GPUJFA9(self.M, self.P1, self.P2,
                        self.x_size, self.y_size, step=f)
            ################################################
            save(self.M, self.x_size, self.y_size, prefix="jfa_plus")
            ################################################
            if f <= 1:
                break


class JFA_plus_inplace(algorithm):
    def prep_draw(self):
        print("[PREP_DRAW]")
        for i in range(0, len(self.pts)):
            self.M[self.pts[i][0], self.pts[i][1]] = self.ids[i]
            self.P[self.pts[i][0], self.pts[i][1]] = self.pts[i]
        import numpy as np # nalezy zaaplikowac szum
        self.NOISE = np.random.randint(
            len(self.pts), size=self.x_size * self.y_size, dtype=np.uint32)

    def core(self):
        """Moja modyfikacja JFA: szum i redukcja kroku.
        Wersja gdzie szum jest wykorzystywany tylko przy braku informacji."""
        for factor in range(1, +oo, 2):
            f = math.ceil(max(self.x_size, self.y_size) / (2**(factor)))
            print("[step] = {}".format(f))
            self.M, self.P1, self.P2 = \
                GPUJFA9_noise(self.M, self.P1, self.P2,
                              self.pts, self.ids, self.NOISE,
                              self.x_size, self.y_size, step=f)
            ################################################
            save(self.M, self.x_size, self.y_size, prefix="jfa_plus_inplace")
            ################################################
            if f <= 1:
                break


class JFA_star(algorithm):
    def prep_gpu(self):
        print("[PREP_GPU]")
        self.M, self.P1, self.P2 = \
            do_split(self.M, self.P)
        import numpy as np # nalezy zaaplikowac szum
        self.NOISE = np.random.randint(
            len(self.pts), size=self.x_size * self.y_size, dtype=np.uint32)
        self.M, self.P1, self.P2 = GPU_noise(self.M, self.P1, self.P2,
                                             self.pts, self.ids, self.NOISE,
                                             self.x_size, self.y_size)

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

        i = 0
        for factor in range(1, +oo, 1):
            i += 1
            f = math.ceil(max(self.x_size, self.y_size) / (3**(factor)))
            print("[step] = {}".format(f))
            self.M, self.P1, self.P2 = \
                GPUJFA_star(self.M, self.P1, self.P2,
                            self.x_size, self.y_size, step=f)
            ################################################
            save(self.M, self.x_size, self.y_size, prefix="jfa_star")
            ################################################
            if f <= 1 or i == n + 1:
                break

        f = 1 # tak jak JFA+1
        print("[step] = {}".format(f))
        self.M, self.P1, self.P2 = \
            GPUJFA_star(self.M, self.P1, self.P2,
                        self.x_size, self.y_size, step=f)
        ################################################
        save(self.M, self.x_size, self.y_size, prefix="jfa_star")
