#include "WeightedSumKernel.h"

__global__ void weightedSum2Kernel(double *output, const double *inputs, const double *weights)
{
    int i = threadIdx.x;
	extern __shared__ float buffer[];
	buffer[i] = inputs[i] * weights[i];

	__syncthreads();

	if(threadIdx.x == 0){
		*output = buffer[0] + buffer[1];
	}
}