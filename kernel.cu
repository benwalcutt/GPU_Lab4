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
	
	int row;
	int col;
	
	global_row = blockIdx.y * TILE_SIZE + threadIdx.y;
	global_col = blockIdx.x * TILE_SIZE + threadIdx.x;
	
	shared_row = threadIdx.y + RADIUS;
	shared_col = threadIdx.x + RADIUS;
	
	// First have each thread copy itself into shared memory assuming it passes the test

	if (global_row >= 0 && global_row < N.height && global_col >= 0 && global_col < N.width) {
		N_s[shared_row][shared_col] = N.elements[global_row * N.width + global_col];
	}
	else {
		N_s[shared_row][shared_col] = 0.0;
	}
	
	// Next we need to populate the halo
	// A
	// check for the top left corner so y = 0, x = 0
	
	if (threadIdx.y < RADIUS && threadIdx.x < RADIUS) {
		// grab right and down
		row = (blockIdx.y + 1) * TILE_SIZE + threadIdx.y;
		col = (blockIdx.x + 1) * TILE_SIZE + threadIdx.x;
		
		// bounds checking
		if (row < N.height && col < N.width && row >= 0 && col >= 0) {
			N_s[shared_row + TILE_SIZE][shared_col + TILE_SIZE] = N.elements[row * N.width + col];
		}
		else
			N_s[shared_row + TILE_SIZE][shared_col + TILE_SIZE] = 0.0;
	}
	
	// B
	// check for bottom left corner so y = TILE_SIZE, x = 0
	
	if (threadIdx.y >= TILE_SIZE - RADIUS && threadIdx.x < RADIUS) {
		// grab up and right
		row = (blockIdx.y - 1) * TILE_SIZE + threadIdx.y;
		col = (blockIdx.x + 1) * TILE_SIZE + threadIdx.x; 
		
		if (row < N.height && col < N.width && row >= 0 && col >= 0) {
			N_s[shared_row - TILE_SIZE][shared_col + TILE_SIZE] = N.elements[row * N.width + col];
		}
		else
			N_s[shared_row - TILE_SIZE][shared_col + TILE_SIZE] = 0.0;
	}
	
	// C
	// check for upper right corner so y = 0; x = TILE_SIZE
	
	if (threadIdx.y < RADIUS && threadIdx.x >= TILE_SIZE - RADIUS) {
		// grab down and left
		row = (blockIdx.y + 1) * TILE_SIZE + threadIdx.y;
		col = (blockIdx.x - 1) * TILE_SIZE + threadIdx.x;
		
		if (row < N.height && col < N.width && row >= 0 && col >= 0) {
			N_s[shared_row + TILE_SIZE][shared_col - TILE_SIZE] = N.elements[row * N.width + col];
		}
		else
			N_s[shared_row + TILE_SIZE][shared_col - TILE_SIZE] = 0.0;
	}
	
	// D
	// check for bottom right corner so y = tile size, x = tile size
	
	if (threadIdx.y >= TILE_SIZE - RADIUS && threadIdx.x >= TILE_SIZE - RADIUS) {
		// grab up and left
		row = (blockIdx.y - 1) * TILE_SIZE + threadIdx.y;
		col = (blockIdx.x - 1) * TILE_SIZE + threadIdx.x;
		
		if (row < N.height && col < N.width && row >= 0 && col >= 0) {
			N_s[shared_row - TILE_SIZE][shared_col - TILE_SIZE] = N.elements[row * N.width + col];
		}
		else
			N_s[shared_row - TILE_SIZE][shared_col - TILE_SIZE] = 0.0;
	}
	
	// E
	// check for top row
	
	if (threadIdx.y < RADIUS) {
		// grab to the bottom
		row = (blockIdx.y + 1) * TILE_SIZE + threadIdx.y;
		col = (blockIdx.x + 0) * TILE_SIZE + threadIdx.x;
		
		if (row < N.height && col < N.width && row >= 0 && col >= 0) {
			N_s[shared_row + TILE_SIZE][shared_col - 0] = N.elements[row * N.width + col];
		}
		else
			N_s[shared_row + TILE_SIZE][shared_col - 0] = 0.0;
	}
	
	// F
	// check for left column
	
	if (threadIdx.x < RADIUS) {
		// grab to the right
		row = (blockIdx.y + 0) * TILE_SIZE + threadIdx.y;
		col = (blockIdx.x + 1) * TILE_SIZE + threadIdx.x;
		
		if (row < N.height && col < N.width && row >= 0 && col >= 0) {
			N_s[shared_row + 0][shared_col + TILE_SIZE] = N.elements[row * N.width + col];
		}
		else
			N_s[shared_row + 0][shared_col + TILE_SIZE] = 0.0;
	}
	
	// G
	// check for bottom row
	
	if (threadIdx.y >= TILE_SIZE - RADIUS) {
		// grab above
		row = (blockIdx.y - 1) * TILE_SIZE + threadIdx.y;
		col = (blockIdx.x + 0) * TILE_SIZE + threadIdx.x;
		
		if (row < N.height && col < N.width && row >= 0 && col >= 0) {
			N_s[shared_row - TILE_SIZE][shared_col + 0] = N.elements[row * N.width + col];
		}
		else
			N_s[shared_row - TILE_SIZE][shared_col + 0] = 0.0;
	}
	
	// H
	// check for right column
	
	if (threadIdx.x >= TILE_SIZE - RADIUS) {
		// grab to the left
		row = (blockIdx.y + 0) * TILE_SIZE + threadIdx.y;
		col = (blockIdx.x - 1) * TILE_SIZE + threadIdx.x;
		
		if (row < N.height && col < N.width && row >= 0 && col >= 0) {
			N_s[shared_row - 0][shared_col - TILE_SIZE] = N.elements[row * N.width + col];
		}
		else
			N_s[shared_row - 0][shared_col - TILE_SIZE] = 0.0;
	}

	__syncthreads();
	
	// calculate
	float Pvalue = 0.0;
	int mi = 0;
	int mj = 0;
	
	// calculate the stuff
	if (global_row < P.height && global_col < P.width) {
		for (int i = -2; i < 3; i++) {
			for (int j = -2; j < 3; j++) {
				// can't use negative indices
				mi = i + 2;
				mj = j + 2;
				Pvalue += N_s[shared_row + i][shared_col + j] * M_c[mi][mj];
			}
		}
		P.elements[global_row * P.width + global_col] = Pvalue;
	}
}
