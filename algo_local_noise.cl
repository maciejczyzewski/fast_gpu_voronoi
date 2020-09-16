static float noise3D(float x, float y, float z) {
	float ptr = 0.0f;
	return fract(sin(x*112.9898f + y*179.233f + z*237.212f) * 43758.5453f, &ptr);
}

__kernel void fn(
	__global const uint2 *points_in,
    __global const uint  *seeds_in,
    __global uint       *id_out,
    __global uint       *x_out,
    __global uint       *y_out,
    int                  x_size,
    int                  y_size,
    int                  number_of_seeds
) {
    int y = get_global_id(0);
    int x = get_global_id(1);
#define POS(X, Y) ((X)*y_size + (Y))

	int pos = noise3D(y, 0.0f, x) * (number_of_seeds);
	int lx = (x+points_in[pos].x)/2;
	int ly = (y+points_in[pos].y)/2;

	if (id_out[POS(lx,ly)] == 0) {
		id_out[POS(lx,ly)] = seeds_in[pos];
		x_out[POS(lx,ly)] = points_in[pos].x;
		y_out[POS(lx,ly)] = points_in[pos].y;
	}

	if (id_out[POS(x,y)] == 0) {
		id_out[POS(x,y)] = seeds_in[pos];
		x_out[POS(x,y)] = points_in[pos].x;
		y_out[POS(x,y)] = points_in[pos].y;
	}
}
