/*
 * CUDA CHALLENGE INDIA
 * 
 * Team Name: Flash
 * Team members: Dhruv Kohli, Vedant Kohli
 *
 * Libraries used: thrust
 *
 * Control Flow:
 * main -> processMap -> saturateMap -> saturateMapParallel ->
 * binarizeMap -> calcThreshold -> binarizeMapParallel -> 
 * performCCLnPrintNCC -> initLabels -> scanning -> analysis ->
 * makeOnePointPerCC -> printNCC
 *
 * References:
 * Connected Component Labelling:
 * 1. K. Hawick, A. Leist, D. Playne, "Parallel graph component labelling with GPUs
 * and CUDA", Parallel Computing 36 (12) (2010) 655–678.
 * 2. O. Kalentev, A. Rai, S. Kemnitz, and R. Schneider, "Connected component labeling 
 * on a 2D grid using CUDA," J. Parallel Distributed Computing, pp. 615-620, 2011.
 * 3. NVIDIA, Cuda programming guide 6.5.
 * 
 * This code makes use of Label equivalence algorithm described in [2] for CCL.
 */


#include <stdlib.h>	//exit, malloc
#include "input.h"	//getinput
#include <thrust/device_vector.h>	//thrust::reduce one device pointer
#include <stdio.h>	//print

typedef unsigned int uint;

#define BLOCKSIZE 256
#define cudaMemcpyHTD(dest, src, nBytes) cudaMemcpy(dest, src, nBytes, cudaMemcpyHostToDevice)
#define cudaMemcpyDTH(dest, src, nBytes) cudaMemcpy(dest, src, nBytes, cudaMemcpyDeviceToHost)

//process map
void processMap(uint*, uint, uint);

//finding hills and dales and replacing by mean and median respectively
void saturateMap(uint*, uint, uint);
__global__ void saturateMapParallel(uint*, uint*, bool*, uint, uint);

//binarize map
void binarizeMap(uint*, uint, uint);
int calcThreshold(uint*, uint, uint);
__global__ void binarizeMapParallel(uint*, uint, uint, uint);

//connected component labelling and finding number of connected components
void performCCLnPrintNCC(uint*, uint, uint);
__global__ void initLabels(int*, uint*, uint, uint);
__global__ void scanning(int*, bool*, uint, uint);
__global__ void analysis(int*, uint, uint);
__global__ void markOnePointPerCC(uint*, int*, uint, uint);
void printNCC(uint*, uint, uint);

uint *get_my_input() {
	int MYSIZE = 16384;
	uint *input = new uint[3+MYSIZE*MYSIZE];
	input[0]=1;
	input[1]=MYSIZE;
	input[2]=MYSIZE;
	for(int i = 0; i < MYSIZE; ++i) {
		for(int j = 0; j < MYSIZE; ++j) {
			input[3+j+i*MYSIZE] =1;
		}
	}
	return input;
}

int
main(int argc, char **argv) {
	uint *input = get_my_input();
	uint numMaps;
	uint numRows, numCols;

	if(input != NULL) {
		numMaps = input[0];
	} else {
		printf("Error: input is NULL!\n");
		std::exit(EXIT_FAILURE);
	}

	uint i = 0;
	uint offset = 1;
	while(i < numMaps) {
		numRows = input[offset];
		numCols = input[offset + 1];
		offset += 2;
		printf("MAP #%d: ", i+1);
		processMap(input + offset, numRows, numCols);
		offset += numRows * numCols;
		i++;
	}
	return 0;
}
/*
 * saturateMap description:
 * -Reuses buffer by swapping pointers. This prevents DTDcpy of the map.
 * -Calls SaturateMapParallel (one thread per cell)until all values converge or iterations exceed 
 *  max iterations.
 * -Once saturated, the map is binarized and CCL is performed over binarized map.
 */
void
saturateMap(uint *h_map, uint rows, uint cols) {
	//variable dec
	int netCells = rows*cols, iter = 0;
	uint *d_mapOut, *d_mapIn;
	bool *d_notConverged, h_notConverged = true;//flags to check if values in map converged
	dim3 threadsPerBlock(BLOCKSIZE);
	dim3 numBlocks((netCells-1)/threadsPerBlock.x + 1); 

	//memory allocation on device and copy map data from host to device
	cudaMalloc((void**)&d_mapIn, rows*cols*sizeof(uint));
	cudaMalloc((void**)&d_mapOut, rows*cols*sizeof(uint));
	cudaMalloc((void**)&d_notConverged, sizeof(bool));
	cudaMemcpyHTD(d_mapIn, h_map, rows*cols*sizeof(uint));

	//processing
	while(1) {
		cudaMemcpyHTD(d_notConverged, &h_notConverged, sizeof(bool));
		if(NUM_ITERATIONS != 0) {
			if(iter%2==0)
				saturateMapParallel<<<numBlocks, threadsPerBlock>>>(d_mapIn, d_mapOut, 
					d_notConverged, rows, cols);
			else
				saturateMapParallel<<<numBlocks, threadsPerBlock>>>(d_mapOut, d_mapIn, 
					d_notConverged, rows, cols);
		}
		cudaMemcpyDTH(&h_notConverged, d_notConverged, sizeof(bool));
		h_notConverged = !h_notConverged;
		++iter;
		//converged or max number of iterations reached
		if(iter >= NUM_ITERATIONS || !h_notConverged) {
			if(iter%2==1) {
				binarizeMap(d_mapOut, rows, cols);
				performCCLnPrintNCC(d_mapOut, rows, cols);
			} else {
				binarizeMap(d_mapIn, rows, cols);
				performCCLnPrintNCC(d_mapIn, rows, cols);
			}
			break;
		}
	}

	//free up the memory
	cudaFree(d_notConverged);
	cudaFree(d_mapIn);
	cudaFree(d_mapOut);
}

/*
 * saturateMapParallel Description:
 * -Checks if value at idx of input map is a hill or Dale.
 * -If hill then put avg. of surrounding cells at idx of output map and raise flag not cvg.
 * -If dale then put median of surrounding cells at idx of outmap map and raise flag not cvg.
 * -Else put the same value i.e. value at idx in input map at idx of output map
 *
 * Other variants of this kernel which failed i.e. increased execution time:
 * -Shared Memory version increased execution time due to the synchronization barrier.
 *  It was observed that the net synchronization time was greater than the time to access 7
 *  extra global cells per thread.
 * -Lookup table version which stores flag for each cell, whether the cell can be a hill or
 *  a dale in its lifetime. That means that if a cell and one of the surrounding cell has
 *  same value then that cell can never be a hill or a dale in its life time, this boolean
 *  flag is stored in the lookup table but that too took extra time due to more overhead in 
 *  global memory access than to return from the kernel if the cell has value equal to one of
 *  the corresponding cell.
 *
 */
__global__ void
saturateMapParallel(uint *inputMap, uint *outputMap, bool *notConverged, uint rows, uint cols) {
	uint idx = threadIdx.x + blockIdx.x*blockDim.x;
	int i = idx/cols;
	int j = idx%cols;
	if(i >= rows || j >= cols)
		return;
	uint cell = j+i*cols;
	uint focusElem = inputMap[cell];
	if(i == 0 || i == rows - 1 || j == 0 || j == cols - 1) {
		outputMap[cell] = focusElem;
		return;
	}
	uint localMap[8];
	int cntU = 0, cntD = 0, cnt = 0, sum = 0;

	//checks if the cell is hill/dale/none
	for(int l = i - 1; l <= i + 1; ++l) {
		for(int k = j - 1; k <= j + 1; ++k) {
			if(!(l == i && k == j)) {
				localMap[cnt] = inputMap[k+l*cols];
				sum += localMap[cnt];
				if(localMap[cnt] > focusElem)
					++cntD;
				else if(localMap[cnt] < focusElem)
					++cntU;
				else {
					//cell is neither hill nor dale and can never be in its lifetime
					outputMap[cell]=focusElem;
					return;
				}
				++cnt;
			}
		}
	}
	if(cntU == 8) {//hill
		outputMap[cell]=sum/8;
		*notConverged = false;
	} else if(cntD == 8) {//dale
		//sorting to get first 5 terms in sorted array localMap
		for(int l = 0; l < 4; ++l) {
			for(int k = 0; k < 7 - l; ++k) {
				if(localMap[k] > localMap[k+1]) {
					uint temp = localMap[k];
					localMap[k] = localMap[k+1];
					localMap[k+1]=temp;
				}
			}
		}
		outputMap[cell] = (localMap[3]+localMap[4])/2;
		*notConverged = false;
	} else {//none
		outputMap[cell] = focusElem;
	}
		
}

/*
 * binarizeMap description:
 * -Obtains threshold and passes it to the binarizeMapParallel (with one cell per thread)
 *  which puts the binarized map in d_input.
 */
void
binarizeMap(uint *d_input, uint rows, uint cols) {
	int netCells = rows*cols, threshold;
	dim3 threadsPerBlock(BLOCKSIZE);
	dim3 numBlocks((netCells-1)/threadsPerBlock.x + 1); 
	threshold = calcThreshold(d_input, rows, cols);
	binarizeMapParallel<<<numBlocks, threadsPerBlock>>>(d_input, rows, cols, threshold);
}

/*
 * calcThreshold description:
 * -Calculates the sum of the values in d_input map using thrust::reduce and then
 * divides the sum with rows*cols and finally returns this value as threshold.
 */
int
calcThreshold(uint *d_input, uint rows, uint cols) {
	int sum, threshold;
	thrust::device_ptr<uint> dev_ptr(d_input);
	sum = thrust::reduce(dev_ptr, dev_ptr+rows*cols, (int)0, thrust::plus<int>());
	threshold = sum/(rows*cols);
	return threshold;
}

/*
 * binarizeMapParallel description:
 * -Checks whether the value of the input map represented by idx is less than threshold.If it
 *  is then assign 0 to that cell in the input map, else assign 1.
 */
__global__ void
binarizeMapParallel(uint *inputMap, uint rows, uint cols, uint threshold) {
	uint idx = threadIdx.x + blockIdx.x*blockDim.x;
	uint i = idx/cols;
	uint j = idx%cols;
	if(i >= rows || j >= cols)
		return;
	uint cell = j+i*cols;
	if(inputMap[cell] < threshold)
		inputMap[cell]=0;
	else
		inputMap[cell]=1;
}

/*
 * performCCLnPrintNCC description:
 * -Uses the algorithm mentioned in reference[2] to label the connected components of
 *  the input binary map.
 * -Calls markOnePointPerCC which marks only one cell of each connected component.
 * -Then calls printNCC to calculate the number of cells which are marked and that
 *  number will be the number of components.
 */
void
performCCLnPrintNCC(uint *d_input, uint rows, uint cols) {
	//variable declaration
	int netCells = rows*cols;
	int *d_label; 
	bool h_notConverged = true, *d_notConverged;
	uint rowsPad = rows+2, colsPad = cols+2;
	dim3 threadsPerBlock(BLOCKSIZE);
	dim3 numBlocks((netCells-1)/threadsPerBlock.x + 1);

	//memory allocation
	cudaMalloc((void**)&d_label, rowsPad*colsPad*sizeof(int));
	cudaMalloc((void**)&d_notConverged, sizeof(bool));

	//initializing labels
	initLabels<<<(rowsPad*colsPad-1)/threadsPerBlock.x+1, threadsPerBlock>>>(d_label, d_input, rowsPad, colsPad); 
	//computation
	while(h_notConverged) {
		cudaMemcpyHTD(d_notConverged, &h_notConverged, sizeof(bool));
		scanning<<<numBlocks, threadsPerBlock>>>(d_label, d_notConverged, rows, cols);
		cudaMemcpyDTH(&h_notConverged, d_notConverged, sizeof(bool));
		h_notConverged = !h_notConverged;
		if(h_notConverged) {
			analysis<<<numBlocks, threadsPerBlock>>>(d_label, rows, cols);
		}
	}

	markOnePointPerCC<<<numBlocks, threadsPerBlock>>>(d_input, d_label, rows, cols);
	printNCC(d_input, rows, cols);

	//free up the memory
	cudaFree(d_notConverged);
	cudaFree(d_label);
}

/*
 * initLabels description:
 * -If the cell corresponding to idx in the input map is on boundary then label of 
 *  that cell will be 0, else if it's value is zero then label of that cell is 0, 
 *  else if it's value is 1 then label of that cell will be the sequential index of that cell.
 */
__global__ void
initLabels(int *label, uint *map, uint rows, uint cols) {
	uint idx = threadIdx.x + blockIdx.x*blockDim.x;
	int i = idx/cols;
	int j = idx%cols;
	if(i >= rows || j >= cols)
		return;
	int cell_ = j+(i)*(cols);
	if(i == 0 || i == rows-1 || j == 0 || j == cols-1) {
		label[cell_]=0;
		return;
	}
	int cell = j-1+(i-1)*(cols-2);
	label[cell_]=(cell_)*map[cell];
}

/*
 * scanning description:
 * -If the value of the label(LAB) represented by idx is less than minimum(MIN) value of the
 *  surrounding labels then assign label[LAB], the min of the label[LAB] and MIN and raise flag
 *  not converged else do nothing.
 */
__global__ void
scanning(int *label, bool *notConverged, uint rows, uint cols) {
	uint idx = threadIdx.x + blockIdx.x*blockDim.x;
	int i = idx/cols;
	int j = idx%cols;
	if(i >= rows || j >= cols)
		return;
	int cell_ = j+1+(i+1)*(cols+2);
	int l = label[cell_];
	if(l == 0)
		return;
	int lw = label[cell_-1];
	int minl = (rows+2)*(cols+2) + 1;
	if(lw)minl=lw;
	int le = label[cell_+1];
	if(le&&le<minl)minl=le;
	int lwn = label[cell_-cols-3];
	if(lwn&&lwn<minl)minl=lwn;
	int lws = label[cell_+cols+1];
	if(lws&&lws<minl)minl=lws;
	int len = label[cell_-cols-1];
	if(len&&len<minl)minl=len;
	int les = label[cell_+cols+3];
	if(les&&les<minl)minl=les;
	int ln = label[cell_-cols-2];
	if(ln&&ln<minl)minl=ln;
	int ls = label[cell_+cols+2];
	if(ls&&ls<minl)minl=ls;
	if(minl < l) {
		int ll = label[l];
		if(minl<ll)
			label[l]=minl;
		else
			label[l]=ll;
		*notConverged=false;
	}
}

/*
 * analysis description:
 * -Starting with l equal to label corresponding to the idx and reference equal to
 *  label[l], we iterate by replacing l with label[reference] and then reference with 
 *  label[l] until reference becomes equal to l (let this value be VAL).
 * -Relabels the cell represented by idx with VAL.
 */
__global__ void
analysis(int *label, uint rows, uint cols) {
	uint idx = threadIdx.x + blockIdx.x*blockDim.x;
	int i = idx/cols;
	int j = idx%cols;
	if(i >= rows || j >= cols)
		return;
	int cell_ = j+1+(i+1)*(cols+2);
	int l = label[cell_];
	if(l == 0)
		return;
	int ref = label[l];
	while(ref!=l) {
		l=label[ref];
		ref=label[l];
	}
	label[cell_]=l;
}

/*
 * markOnePointPerCC description:
 * -If value of the label represented by idx is same as its index then assign the corresponding
 *  cell in map, the value 1 else 0.
 */
__global__ void
markOnePointPerCC(uint *map, int *label, uint rows, uint cols) {
	uint idx = threadIdx.x + blockIdx.x*blockDim.x;
	int i = idx/cols;
	int j = idx%cols;
	if(i >= rows || j >= cols)
		return;
	int cell = j+i*cols;
	map[cell]=0;
	int cell_ = j+1+(i+1)*(cols+2);
	int l = label[cell_];
	if(l == 0)
		return;
	if(l==cell_)
		map[cell]=1;
}

/*
 * printNCC description:
 * -Calculates the number of Ones in d_input (which is a binary map) by summing
 *  over all values of the cells using thrust::reduce.
 * -Prints this sum which is nothing but the number of components.
 */
void
printNCC(uint *d_input, uint rows, uint cols) {
	int numCC;
	thrust::device_ptr<uint> dev_ptr1(d_input);
	numCC = thrust::reduce(dev_ptr1, dev_ptr1+rows*cols, (int) 0, thrust::plus<int>());
	printf("%d\n", numCC);
}

/*
 * processMap description:
 * -Transfers control to saturateMap which initiates actual processing.
 */
void
processMap(uint *map, uint rows, uint cols) {
	saturateMap(map, rows, cols);
}