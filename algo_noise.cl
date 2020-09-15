static float noise3D(float x, float y, float z) {
	float ptr = 0.0f;
	return fract(sin(x*112.9898f + y*179.233f + z*237.212f) * 43758.5453f, &ptr);
}

__kernel void fn(
	__global const uint2 *points_in,
    __global const uint  *seeds_in,
    __global uint        *mat2d_out,
    int                  x_size,
    int                  y_size,
    int                  number_of_seeds
) {
    int y = get_global_id(0);
    int x = get_global_id(1);
#define POS(X, Y, Z) ((X)*y_size*3 + (Y)*3 + Z)

	if (mat2d_out[POS(x,y,0)] == 0) {
		int pos = noise3D(y, 0.0f, x) * (number_of_seeds);
		mat2d_out[POS(x,y,0)] = seeds_in[pos];
		mat2d_out[POS(x,y,1)] = points_in[pos].x;
		mat2d_out[POS(x,y,2)] = points_in[pos].y;
	}
}
