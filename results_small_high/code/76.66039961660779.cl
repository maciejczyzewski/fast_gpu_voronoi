/*
{   'A': 1.3041768897147341,
    'B': 0.5109643735113631,
    'C': 0.15357123759706506,
    'D': 1.2587270373746717,
    'X': 0.8273668915460026,
    'anchor_distance_ratio': 0.25,
    'anchor_double': False,
    'anchor_num': 12,
    'anchor_number_ratio': 0.75,
    'anchor_type': <function mod_anchor_type__circle at 0x121a55ef0>,
    'bruteforce': False,
    'noise': 'none',
    'step_function': <function mod_step_function__special at 0x121a55cb0>}
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

	
        for(int i = 0; i < 12; i++) {
            int A = (step * cos((float) ((6.28/12) * i) ));
            int B = (step * sin((float) ((6.28/12) * i) ));
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
