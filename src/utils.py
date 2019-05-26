import numpy as np


def do_points(x_size, y_size, num=0):
    """Generuje losowe punkty na plaszczyznie X x Y."""
    # zrob losowe punkty na plaszczysnie 2d (troche wiecej)
    # pts = np.random.randint(low=0, high=x_size,
    #                        size=(int(num * 2), 2), dtype=np.uint16)
    pts = np.array([
        [np.random.randint(0,x_size),
         np.random.randint(0,y_size)] for _ in range(int(num*2))],
        dtype=np.uint16)
    pts = np.unique(pts, axis=0)  # usuwamy duplikaty
    np.random.shuffle(pts)  # losowa permutacja
    #         PUNKTY  |            IDENTYFIKATORY
    return pts[0:num], np.arange(1, num + 1, 1).astype(np.uint32)


def do_boxes(x_size, y_size):
    """Rezerwuje pamiec dla macierzy M oraz P."""
    # 2d + identyfikator seeda
    M = np.zeros([x_size, y_size, 1], dtype=np.uint32)
    M.fill(0)
    # 2d + trzymamy pozycje seeda [x, y]
    P = np.zeros([x_size, y_size, 2], dtype=np.uint32)  # jaka pozycja
    P.fill(0)
    return M, P


def do_sample(x, y, seeds):
    """Generuje losowy przypadek testowy."""
    pts, ids = do_points(x, y, seeds)
    return x, y, pts, ids  # struktura


def do_split(M, P):
    """Przystosuwuje macierz M oraz P do postaci dla GPU."""
    M = M.flatten().astype(np.uint32)            # odpowiedni typ
    P1 = P[:, :, 0].flatten().astype(np.uint32)  # musi byc 1d wektorem
    P2 = P[:, :, 1].flatten().astype(np.uint32)
    return M, P1, P2  # pierwsza, druga wspolrzedna
