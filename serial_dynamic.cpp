#include <iostream>
#include <cstdlib>
#include <vector>
//#include "input.h"
#include <algorithm>
#include <cmath>
#include "input_large.h"
#include <queue>
//#include "inp.h"
typedef unsigned int uint;

int check = 0;

__device__ std::vector< std::pair<int, bool> > getHillsNDales(double **map, uint r, uint c) {
	std::vector< std::pair<int, bool> > v;
	for(uint i = 1; i < r - 1; ++i) {
		for(uint j = 1; j < c - 1; ++j) {
			int cntU = 0;
			int cntD = 0;
			for(int l = i - 1; l <= i + 1; ++l) {
				for(int k = j - 1; k <= j + 1; ++k) {
					if(!(l == i && k == j)) {
						if(map[l][k] > map[i][j])
							++cntD;
						else if(map[l][k] < map[i][j])
							++cntU;
					}
				}
			}
			if(cntU == 8)
				v.push_back(std::make_pair(j + i * c, true));
			else if(cntD == 8)
				v.push_back(std::make_pair(j + i * c, false));
		}
	}
	return v;
}

__device__ void assignNewValuesInMap(double **map, uint rows, uint cols, std::vector< std::pair<int, bool> > hillsNDales) {
	std::vector<double> meanNMedian;
	for(int it = 0; it < hillsNDales.size(); ++it) {
		//std::cout << hillsNDales[it].first%cols << " " << hillsNDales[it].first/cols << " " << hillsNDales[it].second << std::endl;
		int i = hillsNDales[it].first/cols;
		int j = hillsNDales[it].first%cols;
		if(hillsNDales[it].second) {
			double sum = 0;
			for(int l = i - 1; l <= i + 1; ++l) {
				for(int k = j - 1; k <= j + 1; ++k) {
					if(!(l == i && k == j)) {
						sum += map[l][k];
					}
				}
			}
			meanNMedian.push_back(sum/8.0);
		} else {
			std::vector<double> myVec;
			for(int l = i - 1; l <= i + 1; ++l) {
				for(int k = j - 1; k <= j + 1; ++k) {
					if(!(l == i && k == j)) {
						myVec.push_back(map[l][k]);
					}
				}
			}
			std::sort(myVec.begin(), myVec.end());
			meanNMedian.push_back((myVec[3] + myVec[4]) * 0.5);
		}
	}

	for(int it = 0; it < hillsNDales.size(); ++it) {
		//std::cout << hillsNDales[i].first%cols << " " << hillsNDales[i].first/cols << " " << hillsNDales[i].second << std::endl;
		int i = hillsNDales[it].first/cols;
		int j = hillsNDales[it].first%cols;
		map[i][j] = floor(meanNMedian[it]);
	}
}

__device__ void saturateMap(double **map, uint rows, uint cols) {
	std::vector< std::pair<int, bool> > hillsNDales;
	int iter = 0;
	while(iter < NUM_ITERATIONS) {
		hillsNDales = getHillsNDales(map, rows, cols);
		if(hillsNDales.empty())
			break;
		assignNewValuesInMap(map, rows, cols, hillsNDales);
		hillsNDales.clear();
		iter++;
	}
}

__device__ double calcThreshold(double **map, uint r, uint c) {
	double sum = 0;
	for(uint i = 0; i < r; ++i) {
		for(uint j = 0; j < c; ++j) {
			sum += map[i][j];
		}
	}
	return floor(sum/(r*c));
}

__device__ void binarize(double **map, uint rows, uint cols, double T) {
	for(uint i = 0; i < rows; ++i) {
		for(uint j = 0; j < cols; ++j) {
			if(map[i][j] < T) {
				map[i][j] = 0;
			} else
				map[i][j] = 1;
		}
	}
}

__device__ void convertToBinaryMap(double **map, uint rows, uint cols) {
	double T = calcThreshold(map, rows, cols);
	//std::cout << "Threshold " << T << std::endl;
	binarize(map, rows, cols, T);
}

__device__ void calcNPrintNumConnComp(double **map, uint rows, uint cols) {
	uint numComp = 0;
	for(uint i = 0; i < rows; ++i) {
		for(uint j = 0; j < cols; ++j) {
			if(map[i][j] == 1) {
				map[i][j]=0;
				++numComp;
				std::queue<uint> Q;
				Q.push(j+i*cols);
				while(!Q.empty()) {
					uint idx = Q.front();
					Q.pop();
					int i_ = idx/cols;
					int j_ = idx%cols;
					for(int l = i_-1; l <= i_+1; ++l) {
						for(int k = j_-1; k <= j_+1; ++k) {
							if(l >= 0 && k >= 0 && l < rows && k < cols && !(l==i_ && k==j_)) {
								if(map[l][k]==1) {
									map[l][k] = 0;
									Q.push(k+l*cols);
								}
							}
						}
					}
				}
			}
		}
	}
	std::cout << numComp << std::endl;
}

void processNPrintSol(uint **map, uint *rows, uint *cols) {
	//saturate map
	uint idx = threadIdx.x + blockIdx.x*blockDim.x;
	saturateMap(map, rows, cols);
	convertToBinaryMap(map, rows, cols);
	calcNPrintNumConnComp(map, rows, cols);
}

int main() {
	uint *input = get_input();
	uint numMaps;
	uint *numRows, *numCols;
	uint *offse;
	uint *mapInFocus;
	uint *ans;


	if(input != NULL) {
		numMaps = input[0];	//input[0] contains the number of maps
		numRows = new uint[numMaps];
		numCols = new uint[numMaps];
		offse = new uint[numMaps];
	}

	uint i = 0;
	uint offset = 1;
	while(i < numMaps) {
		numRows[i] = input[offset];		//offset position always contain rows
		numCols[i] = input[offset + 1];
		offset += 2;
		offse[i] = offset;
		//PROCESS AND PRINT SOLUTION
		offset += numRows[i] * numCols[i];
		i++;
	}
	cudaMalloc((void**)&mapInFocus, offset*sizeof(uint));
	cudaMemcpy(mapInFocus, input, offset*sizeof(uint), cudaMemcpyHostToDevice);
	processNPrintSol<<<1, numMaps>>>(mapInFocus, numRows, numCols);

	return 0;
}