import numpy as np
import pyopencl as cl

# FIXME: szybsza metryka + rozne do testu
# FIXME: optymalizacje GPU (triki z __local)
# FIXME: przepisac na Cython?

# tworzymy kontekst i preparujemy karte
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

# flagi do obslugi pamieci
mf = cl.mem_flags

##########################################################################

prg_noise = cl.Program(ctx, """
__kernel void noise(
    __global const uint *M_g,
    __global const uint *P1_g,
    __global const uint *P2_g,
    __global const ushort2 *PTS_g, // @NOISE
    __global const uint *IDS_g,    // @NOISE
    __global const uint *NOISE_g,
    __global uint  *M_o,
    __global uint *P1_o,
    __global uint *P2_o,
    int X_size,
    int Y_size)
{
    int gid = get_global_id(0);

    int m = M_g[gid], p1 = P1_g[gid], p2 = P2_g[gid];

    if (p1 == 0 && p2 == 0) {
        int ridx = NOISE_g[gid];
        m  = IDS_g[ridx];
        p1 = PTS_g[ridx].x;
        p2 = PTS_g[ridx].y;
    }

    M_o[gid]  = m;
    P1_o[gid] = p1;
    P2_o[gid] = p2;
}
""").build()


def GPU_noise(M, P1, P2, PTS, IDS, NOISE, X_size, Y_size):
    """GPU: implementacja szybkiego szumu dla voronoi-a."""

    M_g = cl.Buffer(ctx, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf=M)  # x*y*1
    P1_g = cl.Buffer(
        ctx,
        mf.COPY_HOST_PTR | mf.READ_ONLY,
        hostbuf=P1)  # x*y*1 a
    P2_g = cl.Buffer(
        ctx,
        mf.COPY_HOST_PTR | mf.READ_ONLY,
        hostbuf=P2)  # x*y*1 a
    PTS_g = cl.Buffer(ctx, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf=PTS)
    IDS_g = cl.Buffer(ctx, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf=IDS)
    NOISE_g = cl.Buffer(ctx, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf=NOISE)

    M_o = cl.Buffer(ctx, mf.WRITE_ONLY, M.nbytes)
    P1_o = cl.Buffer(ctx, mf.WRITE_ONLY, P1.nbytes)
    P2_o = cl.Buffer(ctx, mf.WRITE_ONLY, P2.nbytes)

    prg_noise.noise(queue, M.shape, None,
                    M_g, P1_g, P2_g, PTS_g, IDS_g, NOISE_g, M_o, P1_o, P2_o,
                    np.int32(X_size), np.int32(Y_size))

    # przepisujemy z GPU na CPU
    M_n = np.empty_like(M)
    P1_n = np.empty_like(P1)
    P2_n = np.empty_like(P2)
    cl.enqueue_copy(queue, M_n, M_o)
    cl.enqueue_copy(queue, P1_n, P1_o)
    cl.enqueue_copy(queue, P2_n, P2_o)

    return M_n, P1_n, P2_n

##########################################################################


prg_9 = cl.Program(ctx, """
float metric(int x1, int y1, int x2, int y2) {
    float x = x1 - x2;
    float y = y1 - y2;
    return x*x + y*y;
}

__kernel void JFA(
    __global const uint  *M_g,
    __global const uint *P1_g,
    __global const uint *P2_g,
    __global uint  *M_o,
    __global uint *P1_o,
    __global uint *P2_o,
    int X_size,
    int Y_size,
    int step)
{
    int gid = get_global_id(0);
    int y = gid%Y_size;
    int x = (gid-y)/X_size;
#define POS(X, Y) ((X)*X_size + (Y))

    int best_0  = M_g[POS(x,y)];
    int best_1a = P1_g[POS(x, y)];
    int best_1b = P2_g[POS(x, y)];
    float bestS = metric(best_1a, best_1b, x, y);
    if (best_0 == 0)
        bestS = 4294967296;

    int pos[]    = {-step, 0, step};

    for(int i = 0; i < 3; i++)
        for(int j = 0; j < 3; j++) {
            int idx = POS(x+pos[i], y+pos[j]);
            if (P1_g[idx] == 0 && P2_g[idx] == 0) continue;
            float s2 = metric(P1_g[idx], P2_g[idx], x, y);
            if (bestS >= s2) {
                best_0  = M_g[idx];
                best_1a = P1_g[idx];
                best_1b = P2_g[idx];
                bestS   = s2;
            }
        }

    M_o[POS(x,y)]  = best_0;
    P1_o[POS(x,y)] = best_1a;
    P2_o[POS(x,y)] = best_1b;
}
""").build()


def GPUJFA9(M, P1, P2, X_size, Y_size, step=1):
    """GPU: implementacja klasycznego JFA (z 9 punktami selekcji)."""

    M_g = cl.Buffer(ctx, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf=M)  # x*y*1
    P1_g = cl.Buffer(
        ctx,
        mf.COPY_HOST_PTR | mf.READ_ONLY,
        hostbuf=P1)  # x*y*1 a
    P2_g = cl.Buffer(
        ctx,
        mf.COPY_HOST_PTR | mf.READ_ONLY,
        hostbuf=P2)  # x*y*1 a

    M_o = cl.Buffer(ctx, mf.WRITE_ONLY, M.nbytes)
    P1_o = cl.Buffer(ctx, mf.WRITE_ONLY, P1.nbytes)
    P2_o = cl.Buffer(ctx, mf.WRITE_ONLY, P2.nbytes)

    # wykonuje pojedynczy krok
    prg_9.JFA(queue, M.shape, None,
              M_g, P1_g, P2_g, M_o, P1_o, P2_o,
              np.int32(X_size), np.int32(Y_size), np.int32(step))

    # przepisujemy z GPU na CPU
    M_n = np.empty_like(M)
    P1_n = np.empty_like(P1)
    P2_n = np.empty_like(P2)
    cl.enqueue_copy(queue, M_n, M_o)
    cl.enqueue_copy(queue, P1_n, P1_o)
    cl.enqueue_copy(queue, P2_n, P2_o)

    return M_n, P1_n, P2_n

##########################################################################


prg_9_noise = cl.Program(ctx, """
float metric(int x1, int y1, int x2, int y2) {
    float x = x1 - x2;
    float y = y1 - y2;
    return x*x + y*y;
}

__kernel void JFA(
    __global const uint  *M_g,
    __global const uint *P1_g,
    __global const uint *P2_g,
    __global const ushort2 *PTS_g, // @NOISE
    __global const uint *IDS_g,    // @NOISE
    __global const uint *NOISE_g,
    __global uint  *M_o,
    __global uint *P1_o,
    __global uint *P2_o,
    int X_size,
    int Y_size,
    int step)
{
    int gid = get_global_id(0);
    int y = gid%Y_size;
    int x = (gid-y)/X_size;
#define POS(X, Y) ((X)*X_size + (Y))

    int best_0  = M_g[POS(x,y)];
    int best_1a = P1_g[POS(x, y)];
    int best_1b = P2_g[POS(x, y)];
    float bestS = metric(best_1a, best_1b, x, y);
    if (best_0 == 0)
        bestS = 4294967296;

    int pos[]    = {-step, 0, step};

    for(int i = 0; i < 3; i++)
        for(int j = 0; j < 3; j++) {
            int idx = POS(x+pos[i], y+pos[j]);
            int m = M_g[idx], p1 = P1_g[idx], p2 = P2_g[idx];
            if (p1 == 0 && p2 == 0) {
                int ridx = NOISE_g[idx];
                m  = IDS_g[ridx];
                p1 = PTS_g[ridx].x;
                p2 = PTS_g[ridx].y;
            }
            float s2 = metric(p1, p2, x, y);
            if (bestS >= s2) {
                best_0  = m;
                best_1a = p1;
                best_1b = p2;
                bestS   = s2;
            }
        }

    M_o[POS(x,y)]  = best_0;
    P1_o[POS(x,y)] = best_1a;
    P2_o[POS(x,y)] = best_1b;
}
""").build()


def GPUJFA9_noise(M, P1, P2, PTS, IDS, NOISE, X_size, Y_size, step=1):
    """GPU: implementacja klasycznego JFA (z 9 punktami selekcji).
    [bonus] szum jest aplikowany tylko gdy jest potrzebny."""

    M_g = cl.Buffer(ctx, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf=M)  # x*y*1
    P1_g = cl.Buffer(
        ctx,
        mf.COPY_HOST_PTR | mf.READ_ONLY,
        hostbuf=P1)  # x*y*1 a
    P2_g = cl.Buffer(
        ctx,
        mf.COPY_HOST_PTR | mf.READ_ONLY,
        hostbuf=P2)  # x*y*1 a
    PTS_g = cl.Buffer(ctx, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf=PTS)
    IDS_g = cl.Buffer(ctx, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf=IDS)
    NOISE_g = cl.Buffer(ctx, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf=NOISE)

    M_o = cl.Buffer(ctx, mf.WRITE_ONLY, M.nbytes)
    P1_o = cl.Buffer(ctx, mf.WRITE_ONLY, P1.nbytes)
    P2_o = cl.Buffer(ctx, mf.WRITE_ONLY, P2.nbytes)

    # wykonuje pojedynczy krok
    prg_9_noise.JFA(queue, M.shape, None,
                    M_g, P1_g, P2_g, PTS_g, IDS_g, NOISE_g, M_o, P1_o, P2_o,
                    np.int32(X_size), np.int32(Y_size),
                    np.int32(step))

    # przepisujemy z GPU na CPU
    M_n = np.empty_like(M)
    P1_n = np.empty_like(P1)
    P2_n = np.empty_like(P2)
    cl.enqueue_copy(queue, M_n, M_o)
    cl.enqueue_copy(queue, P1_n, P1_o)
    cl.enqueue_copy(queue, P2_n, P2_o)

    return M_n, P1_n, P2_n

##########################################################################


prg_star = cl.Program(ctx, """
float metric(int x1, int y1, int x2, int y2) {
    float x = x1 - x2;
    float y = y1 - y2;
    return x*x + y*y;
}

__kernel void JFA(
    __global const uint  *M_g,
    __global const uint *P1_g,
    __global const uint *P2_g,
    __global uint  *M_o,
    __global uint *P1_o,
    __global uint *P2_o,
    int X_size,
    int Y_size,
    int step)
{
    int gid = get_global_id(0);
    int y = gid%Y_size;
    int x = (gid-y)/X_size;
#define POS(X, Y) ((X)*X_size + (Y))

    int best_0  = M_g[POS(x,y)];
    int best_1a = P1_g[POS(x, y)];
    int best_1b = P2_g[POS(x, y)];
    float bestS = metric(best_1a, best_1b, x, y);
    if (best_0 == 0)
        bestS = 4294967296;

    for(int i = 0; i < 12; i++) {
        // float( ((360/6) * i)*(3.14 / 180) )
        int A = (step * cos(float( ((6.28/12) * i) )));
        int B = (step * sin(float( ((6.28/12) * i) )));
        int idx = POS(x+A, y+B);
        if (P1_g[idx] == 0 && P2_g[idx] == 0) continue;
        float s2 = metric(P1_g[idx], P2_g[idx], x, y);
        if (bestS >= s2) {
            best_0  = M_g[idx];
            best_1a = P1_g[idx];
            best_1b = P2_g[idx];
            bestS   = s2;
        }
    }

    M_o[POS(x,y)]  = best_0;
    P1_o[POS(x,y)] = best_1a;
    P2_o[POS(x,y)] = best_1b;
}
""").build()


def GPUJFA_star(M, P1, P2, X_size, Y_size, step=1):
    """GPU: implementacja szybkiego algorytmu heurystycznego "JFA_star".
    Punkty selekcji sa ustawione w okrag 12 punktow.
    Szum nalezy zaaplikowac jako wejsciowa maska."""

    M_g = cl.Buffer(ctx, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf=M)  # x*y*1
    P1_g = cl.Buffer(
        ctx,
        mf.COPY_HOST_PTR | mf.READ_ONLY,
        hostbuf=P1)  # x*y*1 a
    P2_g = cl.Buffer(
        ctx,
        mf.COPY_HOST_PTR | mf.READ_ONLY,
        hostbuf=P2)  # x*y*1 a

    M_o = cl.Buffer(ctx, mf.WRITE_ONLY, M.nbytes)
    P1_o = cl.Buffer(ctx, mf.WRITE_ONLY, P1.nbytes)
    P2_o = cl.Buffer(ctx, mf.WRITE_ONLY, P2.nbytes)

    # wykonuje pojedynczy krok
    prg_star.JFA(queue, M.shape, None,
                 M_g, P1_g, P2_g, M_o, P1_o, P2_o,
                 np.int32(X_size), np.int32(Y_size), np.int32(step))

    # przepisujemy z GPU na CPU
    M_n = np.empty_like(M)
    P1_n = np.empty_like(P1)
    P2_n = np.empty_like(P2)
    cl.enqueue_copy(queue, M_n, M_o)
    cl.enqueue_copy(queue, P1_n, P1_o)
    cl.enqueue_copy(queue, P2_n, P2_o)

    return M_n, P1_n, P2_n
