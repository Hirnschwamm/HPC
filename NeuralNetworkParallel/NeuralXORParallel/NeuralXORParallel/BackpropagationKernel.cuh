#include "cuda_runtime.h"
#include "math_functions.h"
#include "math_constants.h"
#include "math.h"
#include "device_launch_parameters.h"
#include <device_functions.h>

__global__ void backpropagationPass(double* inputData, double* ouputData, double* weights, int weightPitch);