#pragma OPENCL EXTENSION cl_ext_device_fission : enable

int metric(int x1, int y1, int x2, int y2) {
    int x = x1 - x2;
    int y = y1 - y2;
    return x*x + y*y;
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

	int best_seed = 0;
    int best_pos_x = 0;
    int best_pos_y = 0;
    int best_score = 16776832;
	
    for(int i=0; i < number_of_seeds; i++) {
        int current_score = metric(points_in[i].x, points_in[i].y, x, y);
        if (best_score >= current_score) {
            best_seed  = seeds_in[i];
            best_pos_x = points_in[i].x;
            best_pos_y = points_in[i].y;
            best_score = current_score;
        }
    }
    
    id_out[POS(x,y)] = best_seed;
    x_out[POS(x,y)] = best_pos_x;
    y_out[POS(x,y)] = best_pos_y;
}
