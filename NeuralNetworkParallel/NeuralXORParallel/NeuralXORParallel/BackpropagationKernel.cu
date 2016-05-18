#include "BackpropagationKernel.cuh"

__device__ double squash(double weightedSum){
	const double e = 2.71828;
	return 1.0 / (1.0 + pow(e, -weightedSum));
}

__global__ void backpropagationPass(double* inputData, double* ouputData, double* weights, int weightPitch){
	const int layerNum = 3;
	const int maxNodeNum = 2;

	double nodeData[layerNum][maxNodeNum];
	nodeData[0][0] = inputData[threadIdx.x * 2];
	nodeData[0][1] = inputData[threadIdx.x * 2 + 1];
	for(int i = 1; i < layerNum; i++){
		for(int j = 0; j < maxNodeNum; j++){
			nodeData[i][j] = 0.0;
		}
	}

	//TEMP?
	int numNodesPerLayer[3] = {2, 2, 1};

	//forward pass
	for(int i = 1; i < layerNum; i++){
		for(int j = 0; j < numNodesPerLayer[i]; j++){
			for(int k = 0; k < numNodesPerLayer[i - 1]; k++){
				double* weightsRow = (double*)((char*)weights + (weightPitch * (i - 1))); //calculate the row for weights[i-1][j + k * maxNodeNum]
				nodeData[i][j] += nodeData[i - 1][k] * weightsRow[j + k * numNodesPerLayer[i]]; 
			}
			nodeData[i][j] = squash(nodeData[i][j]);
		} 
	}

	inputData[threadIdx.x] = nodeData[2][0];
}
