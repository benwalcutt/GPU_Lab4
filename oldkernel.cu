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

__global__ void convolution(Matrix N, Matrix P, int Height, int Width)
{
	/********************************************************************
	Determine input and output indexes of each thread
	Load a tile of the input image to shared memory
	Apply the filter on the input image tile
	Write the compute values to the output image at the correct indexes
	********************************************************************/

    //INSERT KERNEL CODE HERE

	__shared__ float N_s[16][16];
	
	// determine row and column for each thread
	int global_row;
	int global_column;
	
	int shared_row;
	int shared_column;
	
	global_row    	= blockIdx.y * TILE_SIZE + threadIdx.y;
	global_column 	= blockIdx.x * TILE_SIZE + threadIdx.x;
	
	shared_row		= threadIdx.y + RADIUS;
	shared_column 	= threadIdx.x + RADIUS;
	
	// initial boundary checking
	if (global_row < Height && global_column < Width) {
		
	// each thread needs to copy its N element into the shared memory
	N_s[shared_row][shared_column] = N[global_row * Width + global_column];
	
	// the halo elements need to be copied into the shared memory using the edge threads
	// if the halo element is outside the original N matrix, input 0.0
	
	// if top rows in block
	if (threadIdx.y < RADIUS) {
		// if top rows in N
		if (global_row < RADIUS) {
			// zero out directly above by shifting row but not column
			N_s[shared_row - RADIUS][shared_column] = 0.0;
			
			// first two zero out the diagonal to the up and left because this is the top of N
			if (shared_column - RADIUS == 0 || shared_column - RADIUS == 1) {
				// shift up and left
				N_s[shared_row - RADIUS][shared_column - RADIUS] = 0.0;
			} // end diagonal if
		} // end global radius check before else
			
		// not top row in N
		else {
			// copy directly above by shifting row but not column
			N_s[shared_row - RADIUS][shared_column] = N[global_row - RADIUS * Width + global_column];
		
			// zero out for diagonal because this is the left most blocks of N
			if (global_column - RADIUS < 0)
				// shift up and left
				N_s[shared_row - RADIUS][shared_column - RADIUS] = 0.0;
			// otherwise, just copy in what is there in the diagonal of N by using the left two threads
			else if (shared_column == RADIUS || shared_column == (RADIUS + 1)) {
				N_s[shared_row - RADIUS][shared_column - RADIUS] = N[(global_row - RADIUS) * Width + global_column - RADIUS];
			} // end diagonal if
		} // final end of global radius check
	} // end block top row check
	
	// if the two leftmost columns in block
	if (threadIdx.x < RADIUS) {
		// if leftmost rows in N
		if (global_column < RADIUS) {
			// zero out directly left by shifting column but not row
			N_s[shared_row][shared_column - RADIUS] = 0.0;
			
			// last two zero out the diagonal by using the bottom threads
			if (shared_row == (BLOCK_SIZE - 1)) {
				// shift down and left
				N_s[(BLOCK_SIZE - 1) + RADIUS][shared_column - RADIUS] = 0.0;
			} // end diagonal if
			if (shared_row == (BLOCK_SIZE - 2)) {
				// shift down and left
				N_s[(BLOCK_SIZE - 2) + RADIUS][shared_column - RADIUS] = 0.0;
			} // end diagonal if
			
		} // end leftmost global row check before else
		else {
			// copy directly to the left by shifting the column but not row
			N_s[shared_row][shared_column - RADIUS] = N[global_row * Width + global_column - RADIUS];
			
			// zero out for the diagonal because these are the bottom blocks of N
			if (global_row == (BLOCK_SIZE - 1)) {
				// shift down and left
				N_s[(BLOCK_SIZE - 1) + RADIUS][shared_column - RADIUS] = 0.0;
			}
			if (global_row == (BLOCK_SIZE - 2)) {
				// shift down and left
				N_s[(BLOCK_SIZE - 2) + RADIUS][shared_column - RADIUS] = 0.0;
			}
			// otherwise copy what is there
			else if (shared_row == (BLOCK_SIZE - 1)) {
				// shift down and left
				N_s[(BLOCK_SIZE - 1) + RADIUS][shared_column - RADIUS] = N[(global_row + RADIUS) * Width + global_column - RADIUS];
			} // end diagonal if
			else if (shared_row == (BLOCK_SIZE - 2)) {
				// shift down and left
				N_s[(BLOCK_SIZE - 2) + RADIUS][shared_column - RADIUS] = N[(global_row + RADIUS) * Width + global_column - RADIUS];
			} // end diagonal if
		} // final end of global leftmost check
	} // end block left column check
	
	// if the two rightmost columns in block
	if (threadIdx.x == (BLOCK_SIZE - 1) || threadIdx.x == (BLOCK_SIZE - 2)) {
		// if the two rightmost global columns
		if (global_column == (Width - 1) || global_column == (Width - 2)) {
			// zero out directly right by shifting column but not row
			N_s[shared_row][shared_column + RADIUS] = 0.0;
			
			// the last two zero out the diagonal by using the rightmost threads
			if (shared_column == (BLOCK_SIZE - 1) || shared_column == (BLOCK_SIZE - 2)) {
				// shift up and right
				N_s[shared_row - RADIUS][shared_column + RADIUS] = 0.0;
			} // end diagonal if
			
		} // end global rightmost check
		else {
			// copy directly to the right by shifting the column but not row
			N_s[shared_row][(BLOCK_SIZE - 1) + RADIUS] = N[global_row * Width + global_column + RADIUS];
			
			// zero out for diagonal because these are the top most block of N
			if (global_row - RADIUS < 0) {
				// shift up and right
				N_s[shared_row - RADIUS][shared_column + RADIUS] = 0.0;
			} // end diagonal zeroing
			//otherwise copy what is there
			else if (shared_row < RADIUS) {
				// shift up and right
				N_s[shared_row - RADIUS][shared_column + RADIUS] = N[(global_row - RADIUS) * Width + global_column + RADIUS];
			}
			
		} // final end global rightmost check
	} // end block right column check
	
	// if the two bottom rows in block	
	if (threadIdx.y == (BLOCK_SIZE - 1) || threadIdx.y == (BLOCK_SIZE - 2)) {
		// if the two bottom global rows
		if (global_row == (BLOCK_SIZE - 1) || global_row == (BLOCK_SIZE - 2)) {
			// zero out directly below by shifting row but not column
			N_s[shared_row + RADIUS][shared_column] = 0.0;
			
			// the last two zero out the diagonal by using the bottommost threads
			if (shared_row == (BLOCK_SIZE -1) || shared_row == (BLOCK_SIZE - 2)) {
				// shift down and right
				N_s[shared_row + RADIUS][shared_column + RADIUS] = 0.0;
			} // end diagonal if
		} // end global bottom check
		else {
			// copy directly underneath by shifting row but not column
			N_s[shared_row + RADIUS][shared_column] = N[(global_row + RADIUS) * Width + global_column];
			
			// zero out for diagonal because this is the rightmost block of N
			if (global_column == (BLOCK_SIZE - 1) || global_column == (BLOCK_SIZE - 2)) {
				// shift down and right
				N_s[shared_row + RADIUS][shared_column + RADIUS] = 0.0;
			} // end diagonal zeroing
			// otherwise copy what is there
			else if (shared_column == (BLOCK_SIZE - 1) || shared_column == (BLOCK_SIZE - 2)) {
				// shift down and right
				N_s[shared_row + RADIUS][shared_column + RADIUS] = N[(global_row + RADIUS) * Width + global_column + RADIUS];
			}
		} // final end global bottom check
	} // end block bottom row check

	
	// the filter needs to be applied using a for loop from -2 to 2 and applied to row (i) and column (j) from N_s
	// accumulating as you go
	
























		} // end global height and width if
} // end kernel
