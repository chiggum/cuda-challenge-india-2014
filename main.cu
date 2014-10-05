#include <iostream>
#include <cstdlib>
#include <vector>
//#include "input.h"
#include <algorithm>
#include <math.h>
#include <cmath>
#include "input_large.h"
#include <queue>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <stdio.h>

typedef unsigned int uint;

__global__ void assignNewValCmpThresh(uint *map, uint r, uint c, uint *d_in, uint T) {
	uint idx = threadIdx.x + blockIdx.x*blockDim.x;
	uint i = idx/c;
	uint j = idx%c;
	if(i >= r || j >= c)
		return;
	if(map[j+i*c] < T)
		d_in[j+i*c]=0;
	else
		d_in[j+i*c]=1;
}

__global__ void assignLabels(int *d_label, uint *d_map, uint r, uint c) {
	uint idx = threadIdx.x + blockIdx.x*blockDim.x;
	uint i = idx/c;
	uint j = idx%c;
	if(i >= r || j >= c)
		return;
	uint j_ = j+1;
	uint i_ = i+1;
	uint c_ = c+2;
	d_label[j_+i_*c_]=(j_+i_*c_)*d_map[j+i*c];
}

__global__ void assignMinSurrLabel(int *d_label, bool *notConverged, uint r, uint c) {
	uint idx = threadIdx.x + blockIdx.x*blockDim.x;
	int i = idx/c;
	int j = idx%c;
	if(i >= r || j >= c)
		return;
	uint j_ = j+1;
	uint i_ = i+1;
	uint c_ = c+2;
	uint r_ = r+2;
	int l = d_label[j_+i_*c_];
	if(l == 0)
		return;
	int lw = d_label[j_-1+i_*c_];
	int minl = r_*c_ + 1;
	if(lw)minl=lw;
	int le = d_label[j_+1+i_*c_];
	if(le&&le<minl)minl=le;
	int lwn = d_label[j_-1+(i_-1)*c_];
	if(lwn&&lwn<minl)minl=lwn;
	int lws = d_label[j_-1+(i_+1)*c_];
	if(lws&&lws<minl)minl=lws;
	int len = d_label[j_+1+(i_-1)*c_];
	if(len&&len<minl)minl=len;
	int les = d_label[j_+1+(i_+1)*c_];
	if(les&&les<minl)minl=les;
	int ln = d_label[j_+(i_-1)*c_];
	if(ln&&ln<minl)minl=ln;
	int ls = d_label[j_+(i_+1)*c_];
	if(ls&&ls<minl)minl=ls;
	if(minl < l) {
		int ll = d_label[l];
		if(minl<ll)
			d_label[l]=minl;
		else
			d_label[l]=ll;
		*notConverged=false;
	}
}

__global__ void analysisPhase(int *d_label, uint r, uint c) {
	uint idx = threadIdx.x + blockIdx.x*blockDim.x;
	int i = idx/c;
	int j = idx%c;
	if(i >= r || j >= c)
		return;
	uint j_ = j+1;
	uint i_ = i+1;
	uint c_ = c+2;
	int l = d_label[j_+i_*c_];
	if(l == 0)
		return;
	int ref = d_label[l];
	while(ref!=l) {
		l=d_label[ref];
		ref=d_label[l];
	}
	d_label[j_+i_*c_]=l;
}

__global__ void makeValuesOne(uint *d_map, int *d_label, uint r, uint c) {
	uint idx = threadIdx.x + blockIdx.x*blockDim.x;
	uint i = idx/c;
	uint j = idx%c;
	if(i >= r || j >= c)
		return;
	d_map[j+i*c]=0;
	uint j_ = j+1;
	uint i_ = i+1;
	uint c_ = c+2;
	int l = d_label[j_+i_*c_];
	if(l == 0)
		return;
	if(d_label[j_+i_*c_]==j_+i_*c_)
		d_map[j+i*c]=1;
}

__global__ void assignNewValuesParallel(uint *map, uint r, uint c, uint *d_map, bool *notConverged) {
	uint idx = threadIdx.x + blockIdx.x*blockDim.x;
	uint i = idx/c;
	uint j = idx%c;
	if(i >= r || j >= c)
		return;
	uint focusElem = map[j+i*c];
	if(i == 0 || i == r - 1 || j == 0 || j == c - 1) {
		d_map[j + i*c] = focusElem;
		return;
	}
	uint localMap[8];
	int cntU = 0;
	int cntD = 0;
	int cnt = 0;
	uint sum = 0;
	for(int l = i - 1; l <= i + 1; ++l) {
		for(int k = j - 1; k <= j + 1; ++k) {
			if(!(l == i && k == j)) {
				localMap[cnt] = map[k+l*c];
				sum += localMap[cnt];
				if(localMap[cnt] > focusElem)
					++cntD;
				else if(localMap[cnt] < focusElem)
					++cntU;
				++cnt;
			}
		}
	}
	if(cntU == 8) {
		d_map[j + i * c]=sum/8;
		*notConverged = false;
	} else if(cntD == 8) {
		for(int l = 0; l < 4; ++l) {
			for(int k = 0; k < 7 - l; ++k) {
				if(localMap[k] > localMap[k+1]) {
					uint temp = localMap[k];
					localMap[k] = localMap[k+1];
					localMap[k+1]=temp;
				}
			}
		}
		d_map[j + i * c] = (localMap[3]+localMap[4])/2;
		*notConverged = false;
	} else {
		d_map[j+i*c] = focusElem;
	}
		
}

void processNPrintSol(uint *map, uint rows, uint cols) {
	int iter = 0;
	long long int total = rows*cols;
	dim3 threadsPerBlock(1024);
	dim3 numBlocks((total-1)/threadsPerBlock.x + 1); 
	uint *d_map_out, *d_map_in;
	cudaMalloc((void**)&d_map_in, rows*cols*sizeof(uint));
	cudaMalloc((void**)&d_map_out, rows*cols*sizeof(uint));
	cudaMemcpy(d_map_in, map, rows*cols*sizeof(uint), cudaMemcpyHostToDevice);
	bool *d_isCvg;
	cudaMalloc((void**)&d_isCvg, sizeof(bool));
	bool notConverged = true;
	while(iter < NUM_ITERATIONS && notConverged) {
		//notConverged = true;
		cudaMemcpy(d_isCvg, &notConverged, sizeof(bool), cudaMemcpyHostToDevice);
		assignNewValuesParallel<<<numBlocks, threadsPerBlock>>>(d_map_in, rows, cols, d_map_out, d_isCvg);
		cudaDeviceSynchronize();
		cudaMemcpy(&notConverged, d_isCvg, sizeof(bool), cudaMemcpyDeviceToHost);
		notConverged = !notConverged;
		++iter;
		//cudaMemcpy(map, d_map_out, rows*cols*sizeof(uint), cudaMemcpyDeviceToHost);
		if(iter >= NUM_ITERATIONS || !notConverged) {
			thrust::device_ptr<uint> dev_ptr(d_map_out);
			uint sum = thrust::reduce(dev_ptr, dev_ptr+rows*cols, (uint) 0, thrust::plus<uint>());
			sum = sum / (rows*cols);
			assignNewValCmpThresh<<<numBlocks, threadsPerBlock>>>(d_map_out, rows, cols, d_map_in, sum);
			cudaDeviceSynchronize();
			int *d_label;
			uint r_ = rows+2;
			uint c_ = cols+2;
			cudaMalloc((void**)&d_label, r_*c_*sizeof(int));
			cudaMemset(d_label, 0, r_*c_*sizeof(int));
			assignLabels<<<numBlocks, threadsPerBlock>>>(d_label, d_map_in, rows, cols); 
			cudaDeviceSynchronize();
			bool notConverged1 = true;
			bool *d_isCvg1;
			cudaMalloc((void**)&d_isCvg1, sizeof(bool));
			while(notConverged1) {
				//notConverged1 = true;
				cudaMemcpy(d_isCvg1, &notConverged1, sizeof(bool), cudaMemcpyHostToDevice);
				assignMinSurrLabel<<<numBlocks, threadsPerBlock>>>(d_label, d_isCvg1, rows, cols);
				cudaDeviceSynchronize();
				cudaMemcpy(&notConverged1, d_isCvg1, sizeof(bool), cudaMemcpyDeviceToHost);
				notConverged1 = !notConverged1;
				if(notConverged1) {
					analysisPhase<<<numBlocks, threadsPerBlock>>>(d_label, rows, cols);
					cudaDeviceSynchronize();
				}
			}
			cudaFree(d_isCvg1);
			makeValuesOne<<<numBlocks, threadsPerBlock>>>(d_map_in, d_label, rows, cols);
			cudaDeviceSynchronize();
			thrust::device_ptr<uint> dev_ptr1(d_map_in);
			uint sum1 = thrust::reduce(dev_ptr1, dev_ptr1+rows*cols, (uint) 0, thrust::plus<uint>());
			std::cout << sum1 << std::endl;
			cudaFree(d_label);
		} else {
			thrust::device_ptr<uint> dev_ptr2(d_map_in);
			thrust::device_ptr<uint> dev_ptr3(d_map_out);
			thrust::copy(dev_ptr3, dev_ptr3+rows*cols, dev_ptr2);
			//cudaMemcpy(d_map_in, d_map_out, rows*cols*sizeof(uint), cudaMemcpyDeviceToDevice);
		}
	}
	cudaFree(d_isCvg);
	cudaFree(d_map_in);
	cudaFree(d_map_out);
}

int main() {
	uint *input = get_input();
	uint numMaps;
	uint *numRows, *numCols;

	if(input != NULL) {
		numMaps = input[0];	//input[0] contains the number of maps
		numRows = new uint[numMaps];
		numCols = new uint[numMaps];
	}

	uint i = 0;
	uint offset = 1;
	while(i < numMaps) {
		numRows[i] = input[offset];		//offset position always contain rows
		numCols[i] = input[offset + 1];
		offset += 2;
		
		//Allocating space (2d matrix) for storing this map
		std::cout << "MAP #" << (i + 1) << ": ";
		//PROCESS AND PRINT SOLUTION
		processNPrintSol(input + offset, numRows[i], numCols[i]);
		offset += numRows[i] * numCols[i];
		i++;
	}
/*
	i = 0;
	while(i < numMaps) {
		std::cout << "MAP #" << (i + 1) << ": ";
		//PROCESS AND PRINT SOLUTION
		processNPrintSol(mapInFocus[i], numRows[i], numCols[i]);
		delete[] mapInFocus[i];
		i++;
	}
*/
	return 0;
}