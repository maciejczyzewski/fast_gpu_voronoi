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
    int x = (gid-y)/Y_size;
#define POS(X, Y) ((X)*Y_size + (Y))

    int best_0  = M_g[POS(x,y)];
    int best_1a = P1_g[POS(x, y)];
    int best_1b = P2_g[POS(x, y)];
    float bestS = metric(best_1a, best_1b, x, y);
    if (best_0 == 0)
        bestS = 4294967296;

    int pos[]    = {-step, 0, step};

    for(int i = 0; i < 3; i++)
        for(int j = 0; j < 3; j++) {
            int nx = x+pos[i];
            int ny = y+pos[j];
            if(nx < 0 || X_size <= nx || ny < 0 || Y_size <= ny) continue;
            int idx = POS(nx, ny);
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