#pragma OPENCL EXTENSION cl_ext_device_fission : enable

float metric(int x1, int y1, int x2, int y2) {
    float x = x1 - x2;
    float y = y1 - y2;
    return x*x + y*y;
}

int xor_shift(int x) {
    x^= x << 13;
    x^= x >> 17;
    x^= x << 15;
    return x;
}

__kernel void JFA(
    __global const uint  *M_g,
    __global const uint *P1_g,
    __global const uint *P2_g,
    __global uint  *M_o,
    __global uint *P1_o,
    __global uint *P2_o,
    __global const float *cos_val,
    __global const float *sin_val,
    int offset,
    int X_size,
    int Y_size,
    int step)
{
    int gid = get_global_id(0);
    int y = gid%Y_size;
    int x = (gid-y)/Y_size;
#define POS(X, Y) ((X)*Y_size + (Y))

    int trig_id = xor_shift(gid+offset);

    int best_0  = M_g[POS(x,y)];
    int best_1a = P1_g[POS(x, y)];
    int best_1b = P2_g[POS(x, y)];
    float bestS = metric(best_1a, best_1b, x, y);
    if (best_0 == 0)
        bestS = 4294967296;

    for(int i = 0; i < 9; i++) {
        int A = (step * cos_val[(uchar)trig_id]);
        int B = (step * sin_val[(uchar)trig_id]);
        trig_id=xor_shift(trig_id);
        int nx = x+A;
        int ny = y+B;
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