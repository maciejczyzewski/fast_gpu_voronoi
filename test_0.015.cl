/*
{   'anchor_double': False,
    'anchor_num': 7,
    'anchor_type': <function mod_anchor_type__circle at 0x121da7710>,
    'brutforce': False,
    'noise': False,
    'step_function': <function step_function_star at 0x121da74d0>}
*/

#pragma OPENCL EXTENSION cl_ext_device_fission : enable

int metric(int x1, int y1, int x2, int y2) {
    int x = x1 - x2;
    int y = y1 - y2;
    return x*x + y*y;
}

__kernel void fn(
    __global const uint *mat2d_in,
    __global uint       *mat2d_out,
    int                 x_size,
    int                 y_size,
	int                 step
) {
    int y = get_global_id(1);
    int x = get_global_id(0);
#define POS(X, Y, Z) ((X)*y_size*3 + (Y)*3 + Z)

	// FIXME: setting to 0 gives similar results?
	int best_seed = mat2d_in[POS(x,y,0)];
    int best_pos_x = mat2d_in[POS(x,y,1)];
    int best_pos_y = mat2d_in[POS(x,y,2)];
    int best_score = metric(best_pos_x, best_pos_y, x, y);
    if (best_seed == 0)
        best_score = 16776832;

	
        for(int i = 0; i < 7; i++) {
            int A = (step * cos((float) ((6.28/7) * i) ));
            int B = (step * sin((float) ((6.28/7) * i) ));
            int nx = x+A;
            int ny = y+B;
        
            if(nx < 0 || y_size <= nx || ny < 0 || y_size <= ny) continue;
            int anchor_seed = mat2d_in[POS(nx, ny, 0)];
            if (anchor_seed == 0) continue;
			int anchor_pos_x = mat2d_in[POS(nx, ny, 1)];
            int anchor_pos_y = mat2d_in[POS(nx, ny, 2)];
			int anchor_score = metric(anchor_pos_x, anchor_pos_y, x, y);
			if (best_score > anchor_score) {
                best_seed  = anchor_seed;
                best_pos_x = anchor_pos_x;
                best_pos_y = anchor_pos_y;
                best_score = anchor_score;
            }
        }
    
    mat2d_out[POS(x,y,0)] = best_seed;
    mat2d_out[POS(x,y,1)] = best_pos_x;
    mat2d_out[POS(x,y,2)] = best_pos_y;
}
