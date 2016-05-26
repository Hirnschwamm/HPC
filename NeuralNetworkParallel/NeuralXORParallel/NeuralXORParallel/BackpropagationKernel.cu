#include "BackpropagationKernel.cuh"

__device__ double squash(double weightedSum){
	const double e = 2.71828;
	return 1.0 / (1.0 + pow(e, -weightedSum));
}

__global__ void backpropagationPass(double* inputData, double* outputData, double* weights, int weightsWidth, double* biasWeights, int biasWidth, double* error, double learningRate){
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
				int index = weightsWidth * (i - 1) + (j * maxNodeNum + k); //Calculate weights[i-1][j * maxNodeNum + k]
				nodeData[i][j] += nodeData[i - 1][k] * weights[index];
			}
			int index = biasWidth * (i - 1) + j;
			nodeData[i][j] += biasWeights[index];
			nodeData[i][j] = squash(nodeData[i][j]);
		} 
	}

	error[threadIdx.x] = 0.5 * ((outputData[threadIdx.x] - nodeData[2][0]) * (outputData[threadIdx.x] - nodeData[2][0]));

	double weightBuffer[2][4] = {{0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}};
	double biasBuffer[2][2] =  {{0.0, 0.0}, {0.0, 0.0}};
	double target = outputData[threadIdx.x];
	double delta = 0.0;
	double correction = 0.0;

	//calculate weightcorrections for the output layer and store them in the buffer
	delta = -(target - nodeData[2][0]) * nodeData[2][0] * (1.0 - nodeData[2][0]);
	for(int j = 0; j < numNodesPerLayer[1]; j++){
		correction = delta * nodeData[1][j];
		weightBuffer[1][j] = learningRate * correction;
	}
	biasBuffer[1][0] = learningRate * delta;

	//calculate weightcorrections for the hidden layer and store them in the buffer
	double outputDeltaSummed = 0.0;
	double hiddenDelta = 0.0;
	for(int i = 0; i < numNodesPerLayer[1]; i++){
		int index = weightsWidth + i;
		outputDeltaSummed = delta * weights[index]; //weights[1][i]
		hiddenDelta = outputDeltaSummed * nodeData[1][i] * (1.0 - nodeData[1][i]);
		for(int j = 0; j < numNodesPerLayer[0]; j++){
			correction = hiddenDelta * nodeData[0][j];
			weightBuffer[0][i * 2 + j] = learningRate * correction;
		}
		biasBuffer[0][i] = learningRate * hiddenDelta;
	}

	//Gather weight and bias correction in shared memory
	__shared__ double sharedBufferWeights[4][8];
	int k = 0;
	for(int i = 0; i < 2; i++){
		for(int j = 0; j < 4; j++){
			sharedBufferWeights[threadIdx.x][k] = weightBuffer[i][j];
			k++;
		}
	}

	__shared__ double sharedBufferBias[4][4];
	k = 0;
	for(int i = 0; i < 2; i++){
		for(int j = 0; j < 2; j++){
			sharedBufferBias[threadIdx.x][k] = biasBuffer[i][j];
			k++;
		}
	}

	__syncthreads();

	if(threadIdx.x == 0){
		k = 0;
		for(int i = 0; i < 2; i++){
			for(int j = 0; j < 4; j++){
				int index = weightsWidth * i + j; //weights[i][j]
				double correction = sharedBufferWeights[0][k] + sharedBufferWeights[1][k] + sharedBufferWeights[2][k] + sharedBufferWeights[3][k];
				weights[index] -= correction;
				k++;
			}
		}

		k = 0;
		for(int i = 0; i < 2; i++){
			for(int j = 0; j < 2; j++){
				int index = biasWidth * i + j; //biasweights[i][j]
				double correction = sharedBufferBias[0][k] + sharedBufferBias[1][k] + sharedBufferBias[2][k] + sharedBufferBias[3][k];
				biasWeights[index] -= correction;
				k++;
			}
		}
	}
	
}
