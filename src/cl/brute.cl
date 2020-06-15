float metric(int x1, int y1, int x2, int y2) {
    float x = x1 - x2;
    float y = y1 - y2;
    return x*x + y*y;
}

__kernel void Brute(
    __global const ushort2 *PTS_g,
    __global const uint *IDS_g,
    __global uint  *M_o,
    __global uint *P1_o,
    __global uint *P2_o,
    int numbef_of_seeds,
    int X_size,
    int Y_size)
{
    int gid = get_global_id(0);
    int y = gid%Y_size;
    int x = (gid-y)/Y_size;
#define POS(X, Y) ((X)*Y_size + (Y))
    int best_0 = IDS_g[0];
    int best_1a = PTS_g[0].x;
    int best_1b = PTS_g[0].y;
    float bestS = metric(best_1a, best_1b, x, y);
    if (best_0 == 0)
        bestS = 4294967296;

    for(int i=1; i < numbef_of_seeds; i++) {
        float s2 = metric(PTS_g[i].x, PTS_g[i].y, x, y);
        if (bestS >= s2) {
            best_0  = IDS_g[i];
            best_1a = PTS_g[i].x;
            best_1b = PTS_g[i].y;
            bestS   = s2;
        }
    }
    
    M_o[POS(x,y)]  = best_0;
    P1_o[POS(x,y)] = best_1a;
    P2_o[POS(x,y)] = best_1b;
}