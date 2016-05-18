#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "XORNetwork.h"

__global__ void weightedSum2Kernel(double *output, const double *inputs, const double *weights);