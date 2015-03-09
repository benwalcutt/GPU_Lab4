/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/
 
#define FILTER_SIZE 5
#define TILE_SIZE 12
#define BLOCK_SIZE (TILE_SIZE + FILTER_SIZE - 1)
#define RADIUS (FILTER_SIZE / 2)

__constant__ float M_c[FILTER_SIZE][FILTER_SIZE];

__global__ void convolution(Matrix N, Matrix P)
{
	/********************************************************************
	Determine input and output indexes of each thread
	Load a tile of the input image to shared memory
	Apply the filter on the input image tile
	Write the compute values to the output image at the correct indexes
	********************************************************************/
	
	__shared__ float N_s[BLOCK_SIZE][BLOCK_SIZE];

    //INSERT KERNEL CODE HERE

	int global_row;
	int global_col;
	
	int shared_row;
	int shared_col;
	
	global_row = blockIdx.y * TILE_SIZE + threadIdx.y;
	global_col = blockIdx.x * TILE_SIZE + threadIdx.x;
	
	shared_row = threadIdx.y + RADIUS;
	shared_col = threadIdx.x + RADIUS;
	
	// use the 0,0 thread to populate the shared memory
	// inefficient and probably incorrect but at this point...
	
	if (threadIdx.x == 0 && threadIdx.y == 0) {
		for (int i = 0; i < 16; i++) {
			for (int j = 0; j < 16; j++) {
				if (((global_row + i - 2 >= 0) && (global_col + j - 2 >= 0)) && ((global_row + i - 2 < N.height) && (global_col + j - 2 < N.width))) {
					N_s[i][j] = N.elements[(global_row + i - 2) * N.width + global_col + j - 2];
				}
				else {
					N_s[i][j] = 0.0;
				}
			}
		}
	}

	__syncthreads();
	
	float Pvalue = 0.0;
	int mi = 0;
	int mj = 0;
	
	// calculate the stuff
	if (global_row < N.height && global_col < N.width) {
		for (int i = -2; i < 3; i++) {
			for (int j = -2; j < 3; j++) {
				mi = i + 2;
				mj = j + 2;
				Pvalue += N_s[shared_row + i][shared_col + j] * M_c[mi][mj];
			}
		}
		P.elements[global_row * P.width + global_col] = Pvalue;
	}
}
