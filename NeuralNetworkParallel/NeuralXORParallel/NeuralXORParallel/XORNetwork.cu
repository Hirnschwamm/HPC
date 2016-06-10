#include "XORNetwork.h"

#include <cstdio>
#include <tuple>
#include <chrono>

#include "rapidjson\document.h"
#include "rapidjson\writer.h"
#include "rapidjson\stringbuffer.h"
#include "rapidjson\filereadstream.h"


XORNetwork::XORNetwork(std::string dataPath){
	int inputDim = 2;
	int outputDim = 1;
	
	for(int i = 0; i < inputDim; i++){
		inputLayer.push_back(new Perceptron(0.0f));
	}

	for(int i = 0; i < inputDim; i++){
		hiddenLayer.push_back(new Perceptron(inputLayer));
	}

	for(int i = 0; i < outputDim; i++){
		outputLayer.push_back(new Perceptron(hiddenLayer));
	}

	initjsonTrainingData(dataPath);
}


XORNetwork::~XORNetwork(void)
{
	for(unsigned int i = 0; i < inputLayer.size(); i++){
		delete inputLayer[i];
	}

	for(unsigned int i = 0; i < hiddenLayer.size(); i++){
		delete hiddenLayer[i];
	}

	for(unsigned int i = 0; i < outputLayer.size(); i++){
		delete outputLayer[i];
	}
}

void XORNetwork::trainByBackpropagation(unsigned int passes, double learningRate){
	std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

	//Gather inputs and corresponding outputs in one big array
	int inputSetSize = std::get<0>(trainingData[0]).size();
	int outputSetSize = std::get<1>(trainingData[0]).size();
	int totalInputSize = inputSetSize * trainingData.size();
	int totalOutputSize = outputSetSize * trainingData.size();

	double* gatheredInputData = new double[totalInputSize];
	double* gatheredOutputData = new double[totalOutputSize];
	std::vector<double>* inputData = 0;
	std::vector<double>* outputData = 0;

	int index = 0;
	for(unsigned int i = 0; i < trainingData.size(); i++){
		inputData = &std::get<0>(trainingData[i]);
		outputData = &std::get<1>(trainingData[i]);

		gatheredInputData[index] = inputData->at(0);
		gatheredInputData[index + 1] = inputData->at(1);
		index += 2;

		gatheredOutputData[i] = outputData->at(0);	
	}

	//Gather existing weights and biases
	int weightlayerNum = 2; //0: hiddenlayer, 1: outputlayer
	int nodesPerLayer[3];
	nodesPerLayer[0] = inputLayer.size();
	nodesPerLayer[1] = hiddenLayer.size();
	nodesPerLayer[2] = outputLayer.size();

	int maxWeightNum = max(inputLayer.size() * hiddenLayer.size(), hiddenLayer.size() * outputLayer.size());
	int totalWeightSize = weightlayerNum * maxWeightNum;
	double *gatheredWeights = new double[totalWeightSize]; 
	
	for(unsigned int i = 0; i < hiddenLayer.size(); i++){
		gatheredWeights[maxWeightNum * 0 + (i * 2)] = hiddenLayer[i]->getWeight(0);     //gatheredWeights[0][i*2]
		gatheredWeights[maxWeightNum * 0 + (i * 2 + 1)] = hiddenLayer[i]->getWeight(1); //gatheredWeights[0][i*2+1]
	}
	for(unsigned int i = 0; i < outputLayer.size(); i++){
		gatheredWeights[maxWeightNum * 1 + (i * 2)] = outputLayer[i]->getWeight(0);     //gatheredWeights[1][i*2]
		gatheredWeights[maxWeightNum * 1 + (i * 2 + 1)] = outputLayer[i]->getWeight(1);	//gatheredWeights[1][i*2+1]
	}

	int maxBiasNum = max(hiddenLayer.size(), outputLayer.size());
	int totalBiasSize = weightlayerNum * maxBiasNum;
	double *gatheredBiases = new double[totalBiasSize];
	for(unsigned int i = 0; i < hiddenLayer.size(); i++){
		gatheredBiases[maxBiasNum * 0 + i] = hiddenLayer[i]->getBiasWeight();
	}
	for(unsigned int i = 0; i < outputLayer.size(); i++){
		gatheredBiases[maxBiasNum * 1 + i] = outputLayer[i]->getBiasWeight();
	}

	//Allocate device memory and copy gathered inputs, output, weights and biases to device
    double *dev_weights = 0;
	double *dev_bias = 0;
	double *dev_error = 0;

	cudaError_t cudaStatus;

	cudaStatus = cudaMalloc((void**)&dev_weights, totalWeightSize * sizeof(double));
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }

	cudaStatus = cudaMalloc((void**)&dev_bias, totalBiasSize * sizeof(double));
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }

	cudaStatus = cudaMalloc((void**)&dev_error, trainingData.size() * sizeof(double));
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }

	cudaStatus = cudaMemcpy(dev_weights, gatheredWeights, totalWeightSize * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }

	cudaStatus = cudaMemcpy(dev_bias, gatheredBiases, totalBiasSize * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }

	int speedUpFac = 44;
	int numThreads = trainingData.size() * speedUpFac;

	int maxNodeNum = max(inputLayer.size(), max(hiddenLayer.size(), outputLayer.size()));
	int sharedNodeDataSize = numThreads * ((weightlayerNum + 1) * maxNodeNum);
	int sharedWeightBufferSize = numThreads * weightlayerNum * maxWeightNum;//((inputLayer.size() * hiddenLayer.size()) + (hiddenLayer.size() * outputLayer.size()));
	int sharedBiasBufferSize = numThreads * weightlayerNum * maxNodeNum;//(hiddenLayer.size() + outputLayer.size());

	int sharedMemorySize = sharedNodeDataSize + sharedWeightBufferSize + sharedBiasBufferSize;
	
	int weightBufferOffset = sharedNodeDataSize;
	int sharedBiasBufferOffset = sharedWeightBufferSize;

	setInputData(gatheredInputData);
	setOutputData(gatheredOutputData);
	setLayerWidth(weightlayerNum + 1);
	setNodeWidth(maxNodeNum);
	setWeightBufferOffset(weightBufferOffset);
	setBiasBufferOffset(sharedBiasBufferOffset);
	setTrainingDataSize(trainingData.size());
	setThreadNum(numThreads);
	setNumNodesPerLayer(nodesPerLayer);
	setWeightsWidth(maxWeightNum);
	setBiasWidth(maxBiasNum);
	setLearningRate(learningRate);
	
	//initiate backpropagation
	for(unsigned int pass = 0; pass < (passes / speedUpFac); pass++){
		backpropagationPass<<<1, numThreads, sharedMemorySize  * sizeof(double)>>>(dev_weights, dev_bias, dev_error);
	}

	//Copy results and errors back to host memory
	double *errors = new double[trainingData.size()];
	cudaStatus = cudaMemcpy(errors, dev_error, trainingData.size() * sizeof(double), cudaMemcpyDeviceToHost);
	cudaStatus = cudaMemcpy(gatheredWeights, dev_weights, 8 * sizeof(double), cudaMemcpyDeviceToHost);
	cudaStatus = cudaMemcpy(gatheredBiases, dev_bias, 4 * sizeof(double), cudaMemcpyDeviceToHost);

	//Print final error per set
	printf("Errors: ");
	for(unsigned int i = 0; i < trainingData.size(); i++){
		printf("%d. Set: %f | ", i, errors[i]);
	}
	printf("\n");

	cudaFree(dev_weights);
	cudaFree(dev_bias);

	//copy results back into the network
	for(unsigned int i = 0; i < hiddenLayer.size(); i++){
		hiddenLayer[i]->setWeight(0, gatheredWeights[maxWeightNum * 0 + (i * 2)]);
		hiddenLayer[i]->setWeight(1, gatheredWeights[maxWeightNum * 0 + (i * 2 + 1)]);
	}

	for(unsigned int i = 0; i < outputLayer.size(); i++){
		outputLayer[i]->setWeight(0, gatheredWeights[maxWeightNum * 1 + (i * 2)]);
		outputLayer[i]->setWeight(1, gatheredWeights[maxWeightNum * 1 + (i * 2 + 1)]);
	}

	for(unsigned int i = 0; i < hiddenLayer.size(); i++){
		hiddenLayer[i]->setBias(gatheredBiases[maxBiasNum * 0 + i]);
	}

	for(unsigned int i = 0; i < outputLayer.size(); i++){
		 outputLayer[i]->setBias(gatheredBiases[maxBiasNum * 1 + i]);
	}

	delete[] gatheredInputData;
	delete[] gatheredOutputData;
	delete[] gatheredWeights;
	delete[] gatheredBiases;
	delete[] errors;

	std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
	long duration = (long)std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();
	printf("Done training AI! That took me %d milliseconds!\n", duration);
}

double XORNetwork::xor(int operand1, int operand2){
	std::vector<double> input;
	input.push_back((double)operand1);
	input.push_back((double)operand2);

	setInput(input);

	return outputLayer[0]->getOutput();
}

void XORNetwork::setInput(std::vector<double>& input){
	for(unsigned int i = 0; i < input.size(); i++){
		inputLayer[i]->setDirectInput(input[i]);
	}
}

void XORNetwork::initjsonTrainingData(std::string path){
	FILE* fp;
	fopen_s(&fp, path.c_str(), "rb"); 
	assert(fp);
	char readBuffer[65536];
	rapidjson::FileReadStream is(fp, readBuffer, sizeof(readBuffer));
	rapidjson::Document jsonDoc;
	jsonDoc.ParseStream(is);
	fclose(fp);
	assert(jsonDoc.IsObject());

	rapidjson::Value& moves = jsonDoc["moves"];
	assert(moves.IsArray());
	std::vector<double> input;
	std::vector<double> output;
	for (rapidjson::SizeType i = 0; i < moves.Size(); i++){
		for(rapidjson::SizeType j = 0; j < moves[i]["input"].Size(); j++){
			input.push_back( moves[i]["input"][j].GetDouble());
		}
		for(rapidjson::SizeType j = 0; j < moves[i]["output"].Size(); j++){
			output.push_back( moves[i]["output"][j].GetDouble());
		}
		trainingData.push_back(std::tuple<std::vector<double>, std::vector<double>>(input, output));
		input.clear();
		output.clear();
	}
}
