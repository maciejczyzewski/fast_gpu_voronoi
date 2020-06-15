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