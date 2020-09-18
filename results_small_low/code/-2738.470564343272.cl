/*
{   'A': 1.969186881089581,
    'B': 0.007116247386750653,
    'C': 0.08081591348648177,
    'D': 1.1945952299616525,
    'X': 0.6411196154353387,
    'anchor_distance_ratio': 0.75,
    'anchor_double': True,
    'anchor_num': 9,
    'anchor_number_ratio': 0.5,
    'anchor_type': <function mod_anchor_type__circle at 0x11ee17ef0>,
    'bruteforce': False,
    'noise': 'none',
    'step_function': <function mod_step_function__special at 0x11ee17cb0>}
*/

#pragma OPENCL EXTENSION cl_ext_device_fission : enable

int metric(int x1, int y1, int x2, int y2) {
    int x = x1 - x2;
    int y = y1 - y2;
    return x*x + y*y;
}

__kernel void fn(
    __global const uint *id_in,
    __global const uint *x_in,
    __global const uint *y_in,
    __global uint       *id_out,
    __global uint       *x_out,
    __global uint       *y_out,
    int                 x_size,
    int                 y_size,
	int                 step
) {
    int y = get_global_id(0);
    int x = get_global_id(1);
#define POS(X, Y) ((X)*y_size + (Y))

	// FIXME: setting to 0 gives similar results?
	int best_seed = id_in[POS(x,y)];
    int best_pos_x = x_in[POS(x,y)];
    int best_pos_y = y_in[POS(x,y)];
    int best_score = metric(best_pos_x, best_pos_y, x, y);
    if (best_seed == 0)
        best_score = 16776832;

	
        for(int j = -1; j < 1; j++)
        for(int i = 0; i < 4 * (
                ( 1 + j) * (1 - 0.5) 
                +
                (-1 * j) * (0.5)
            ); i++) {
            int A = (step * cos((float) ((6.28/4) * i) ));
            int B = (step * sin((float) ((6.28/4) * i) ));
            if (j == 0) {
               A *= 0.75;
               B *= 0.75;
            }
            int nx = x+A;
            int ny = y+B;
        
            if(nx < 0 || x_size <= nx || ny < 0 || y_size <= ny) continue;
            int anchor_seed = id_in[POS(nx, ny)];
            if (anchor_seed == 0) continue;
			int anchor_pos_x = x_in[POS(nx, ny)];
            int anchor_pos_y = y_in[POS(nx, ny)];
			int anchor_score = metric(anchor_pos_x, anchor_pos_y, x, y);
			if (best_score > anchor_score) {
                best_seed  = anchor_seed;
                best_pos_x = anchor_pos_x;
                best_pos_y = anchor_pos_y;
                best_score = anchor_score;
            }
        }
    
    id_out[POS(x,y)] = best_seed;
    x_out[POS(x,y)] = best_pos_x;
    y_out[POS(x,y)] = best_pos_y;
}
