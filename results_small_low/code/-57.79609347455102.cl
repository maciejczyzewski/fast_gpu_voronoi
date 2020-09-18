/*
{   'A': 1.001909157285772,
    'B': 0.9787885494986106,
    'C': 0.18525066388080405,
    'D': 1.7747877397361382,
    'X': 0.7906063590309227,
    'anchor_distance_ratio': 0.6666666666666666,
    'anchor_double': False,
    'anchor_num': 9,
    'anchor_number_ratio': 0.25,
    'anchor_type': <function mod_anchor_type__circle at 0x11ee17ef0>,
    'bruteforce': False,
    'noise': 'lnoise',
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

	
        for(int i = 0; i < 9; i++) {
            int A = (step * cos((float) ((6.28/9) * i) ));
            int B = (step * sin((float) ((6.28/9) * i) ));
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
