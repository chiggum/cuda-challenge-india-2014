#include <iostream>

using namespace std;

__global__ void process2(int *d_in, int *d_out, int idx) {
	int i = threadIdx.x;
	int tot = i + idx*10;
	d_out[tot] = d_in[tot]*d_in[tot];
}

__global__ void process(int *d_in, int *d_out) {
	int i = threadIdx.x;
	process2<<<1, 10>>>(d_in, d_out, i);
}

int main() {
	int h[100];
	for(int i = 0; i < 100; ++i)
		h[i] = i;
	int *d_in, *d_out;
	cudaMalloc((void**)&d_in, 100*sizeof(int));
	cudaMalloc((void**)&d_out, 100*sizeof(int));
	cudaMemcpy(d_in, h, 100*sizeof(int), cudaMemcpyHostToDevice);
	process<<<1, 10>>>(d_in, d_out);
	cudaMemcpy(h, d_out, 100*sizeof(int), cudaMemcpyDeviceToHost);
	for(int i = 0; i < 100; ++i)
		std::cout<<h[i]<<std::endl;
	return 0;
}