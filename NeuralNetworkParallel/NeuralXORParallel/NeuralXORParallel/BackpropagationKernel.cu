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
				int index = weightsWidth * (i - 1) + (j + k * numNodesPerLayer[i]); //Calculate weights[i-1][j + k * maxNodeNum]
				nodeData[i][j] += nodeData[i - 1][k] * weights[index];
			}
			int index = biasWidth * (i - 1) + j;
			nodeData[i][j] += biasWeights[index];
			nodeData[i][j] = squash(nodeData[i][j]);
		} 
	}

	error[threadIdx.x] = 0.5 * ((nodeData[2][0] - outputData[threadIdx.x]) * (nodeData[2][0] - outputData[threadIdx.x]));

	//DEBUG
	/*double* weightsRow = (double*)((char*)weights + (weightPitch * (1 - 1))); //calculate the row for weights[i-1][j + k * maxNodeNum]
	double* biasWeightsRow = (double*)((char*)biasWeights + (biasWeightPitch * (1 - 1)));
	nodeData[1][0] += nodeData[0][0] * weightsRow[0];
	nodeData[1][0] += nodeData[0][1] * weightsRow[1];
	nodeData[1][0] += biasWeightsRow[0];
	nodeData[1][0] = squash(nodeData[1][0]);

	nodeData[1][1] += nodeData[0][0] * weightsRow[2];
	nodeData[1][1] += nodeData[0][1] * weightsRow[3];
	nodeData[1][1] += biasWeightsRow[1];
	nodeData[1][1] = squash(nodeData[1][1]);


	weightsRow = (double*)((char*)weights + (weightPitch * (2 - 1))); //calculate the row for weights[i-1][j + k * maxNodeNum]
	biasWeightsRow = (double*)((char*)biasWeights + (biasWeightPitch * (2 - 1)));
	nodeData[2][0] += nodeData[1][0] * weightsRow[0];
	nodeData[2][0] += nodeData[1][1] * weightsRow[1];
	nodeData[2][0] += biasWeightsRow[0];
	nodeData[2][0] = squash(nodeData[2][0]);

	weightsRow = (double*)((char*)weights + (weightPitch * (1 - 1)));
	
	outputData[0] = nodeData[1][0];
	outputData[1] = nodeData[1][1];
	outputData[2] = nodeData[2][0];
	outputData[3] = 123.456;*/
	//DEBUG
	
	double weightBuffer[2][4];
	double biasBuffer[2][2];
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
		int index = weightsWidth + (i * 2);
		outputDeltaSummed = delta * weights[index]; //weights[0][i * 2]
		hiddenDelta = outputDeltaSummed * nodeData[1][i] * (1.0 - nodeData[1][i]);
		for(int j = 0; j < numNodesPerLayer[0]; j++){
			correction = hiddenDelta * nodeData[0][j];
			weightBuffer[0][i * 2 + j] = learningRate * correction;
		}
		biasBuffer[0][i] = learningRate * hiddenDelta;
	}

	__syncthreads();

	for(int i = 0; i < 2; i++){
		for(int j = 0; j < 4; j++){
			int index = weightsWidth * i + j; //weights[i][j]
			weights[index] -= weightBuffer[i][j];
		}
	}

	for(int i = 0; i < 2; i++){
		for(int j = 0; j < 2; j++){
			int index = biasWidth * i + j; //biasweights[i][j]
			biasWeights[index] -= biasBuffer[i][j];
		}
	}	
}
