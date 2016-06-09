#include "cuda_runtime.h"
#include "math_functions.h"
#include "math_constants.h"
#include "math.h"
#include "device_launch_parameters.h"
#include <device_functions.h>

__constant__ double inputData[8];
void setInputData(double* i);

__constant__ double outputData[4];
void setOutputData(double* o);

__constant__ int layerwidth;
void setLayerWidth(int l);

__constant__ int nodeWidth;
void setNodeWidth(int n);

__constant__ int weightBufferOffset;
void setWeightBufferOffset(int w);

__constant__ int biasBufferOffset;
void setBiasBufferOffset(int b);

__constant__ int trainingDataSize;
void setTrainingDataSize(int t);

__constant__ int threadNum;
void setThreadNum(int t);

__constant__ int numNodesPerLayer[3];
void setNumNodesPerLayer(int* n);

__constant__ int weightsWidth;
void setWeightsWidth(int w);

__constant__ int biasWidth;
void setBiasWidth(int b);

__constant__ double learningRate;
void setLearningRate(double l);

__global__ void backpropagationPass(double* weights, double* biasWeights, double* error);