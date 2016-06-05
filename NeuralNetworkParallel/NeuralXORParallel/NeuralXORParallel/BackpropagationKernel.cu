#include "BackpropagationKernel.cuh"

__device__ double squash(double weightedSum){
	const double e = 2.71828;
	return 1.0 / (1.0 + pow(e, -weightedSum));
}

__global__ void backpropagationPass(int layerwidth,
									int nodeWidth, 
									int nodeBufferOffset,
									int biasBufferOffset,
									int trainingDataSize,
									int* numNodesPerLayer,
									double* inputData, 
									double* outputData, 
									double* weights, 
									int weightsWidth, 
									double* biasWeights, 
									int biasWidth, 
									double* error, 
									double learningRate){
	
	extern __shared__ double sharedMemory[]; 
	double* nodeData = (double*)&sharedMemory;						//3D-Array: nodeData[threadId][layer][node]
	double* weightBuffer = (double*)&nodeData[nodeBufferOffset];	//3D-Array: weightBuffer[threadId][layer][node] 
	double* biasBuffer = (double*)&weightBuffer[biasBufferOffset];  //3D-Array: biasBuffer[threadId][layer][node]

	int threadOffset = nodeWidth * layerwidth;

	nodeData[threadIdx.x * threadOffset + 0 * nodeWidth + 0] = inputData[threadIdx.x * 2];
	nodeData[threadIdx.x * threadOffset + 0 * nodeWidth + 1] = inputData[threadIdx.x * 2 + 1];
	for(int i = 1; i < layerwidth; i++){
		for(int j = 0; j < nodeWidth; j++){
			nodeData[threadIdx.x * threadOffset + i * nodeWidth + j] = 0.0;
		}
	}

	//forward pass
	for(int i = 1; i < layerwidth; i++){
		for(int j = 0; j < numNodesPerLayer[i]; j++){
			for(int k = 0; k < numNodesPerLayer[i - 1]; k++){
				int index = weightsWidth * (i - 1) + (j * nodeWidth + k); //Calculate weights[i-1][j * maxNodeNum + k]
				nodeData[threadIdx.x * threadOffset + i * nodeWidth + j] += nodeData[threadIdx.x * threadOffset + (i - 1) * nodeWidth + k] * weights[index];
			}
			int index = biasWidth * (i - 1) + j;
			nodeData[threadIdx.x * threadOffset + i * nodeWidth + j] += biasWeights[index];
			nodeData[threadIdx.x * threadOffset + i * nodeWidth + j] = squash(nodeData[threadIdx.x * threadOffset + i * nodeWidth + j]);
		} 
	}

	int outputLayerIndex = layerwidth - 1;
	error[threadIdx.x] = 0.5 * ((outputData[threadIdx.x] - nodeData[threadIdx.x * threadOffset + outputLayerIndex * nodeWidth + 0]) * (outputData[threadIdx.x] - nodeData[threadIdx.x * threadOffset + outputLayerIndex * nodeWidth + 0]));

	double target = outputData[threadIdx.x];
	double delta = 0.0;
	double correction = 0.0;

	//calculate weightcorrections for the output layer and store them in the buffer
	delta = -(target - nodeData[threadIdx.x * threadOffset + outputLayerIndex * nodeWidth + 0]) * nodeData[threadIdx.x * threadOffset + outputLayerIndex * nodeWidth + 0] * (1.0 - nodeData[threadIdx.x * threadOffset + outputLayerIndex * nodeWidth + 0]);
	for(int j = 0; j < numNodesPerLayer[1]; j++){
		correction = delta * nodeData[threadIdx.x * threadOffset + 1 * nodeWidth + j];
		weightBuffer[threadIdx.x * threadOffset + nodeWidth * 1 + j] = learningRate * correction;
	}
	biasBuffer[threadIdx.x * threadOffset + nodeWidth * 1 + 0] = learningRate * delta;

	//calculate weightcorrections for the hidden layer and store them in the buffer
	double outputDeltaSummed = 0.0;
	double hiddenDelta = 0.0;
	for(int i = 0; i < numNodesPerLayer[1]; i++){
		int index = weightsWidth + i;
		outputDeltaSummed = delta * weights[index]; //weights[1][i]
		hiddenDelta = outputDeltaSummed * nodeData[threadIdx.x * threadOffset + 1 * nodeWidth + i] * (1.0 - nodeData[threadIdx.x * threadOffset + 1 * nodeWidth + i]);
		for(int j = 0; j < numNodesPerLayer[0]; j++){
			correction = hiddenDelta * nodeData[threadIdx.x * threadOffset + 0 * nodeWidth + j];
			weightBuffer[threadIdx.x * threadOffset + nodeWidth * 0 + (i * 2 + j)] = learningRate * correction;
		}
		biasBuffer[threadIdx.x * threadOffset + nodeWidth * 0 + i] = learningRate * hiddenDelta;
	}
	
	__syncthreads();

	if(threadIdx.x == 0){
		for(int i = 0; i < layerwidth - 1; i++){
			for(int j = 0; j < weightsWidth; j++){
				int index = weightsWidth * i + j; 

				double correction = 0.0;
				for(int k = 0; k < trainingDataSize; k++){
					correction += weightBuffer[k * threadOffset + nodeWidth * i + j];
				}
				
				weights[index] -= correction;
			}
		}

		for(int i = 0; i < layerwidth - 1; i++){
			for(int j = 0; j < biasWidth; j++){
				int index = biasWidth * i + j; 

				double correction = 0.0;
				for(int k = 0; k < trainingDataSize; k++){
					correction += biasBuffer[k * threadOffset + nodeWidth * i + j];
				}

				biasWeights[index] -= correction;
			}
		}
		
	}

}
