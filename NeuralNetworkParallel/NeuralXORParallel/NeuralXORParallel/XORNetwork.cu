#include "XORNetwork.h"

#include <cstdio>
#include <tuple>

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

void XORNetwork::trainByBackpropagation(double errorTolerance, double learningRate){

	double gatheredInputData[8];
	double gatheredOutputData[4];
	std::vector<double>* inputData;
	std::vector<double>* outputData;
	int index = 0;
	for(unsigned int i = 0; i < trainingData.size(); i++){
		inputData = &std::get<0>(trainingData[i]);
		outputData = &std::get<1>(trainingData[i]);

		gatheredInputData[index] = inputData->at(0);
		gatheredInputData[index + 1] = inputData->at(1);
		index += 2;

		gatheredOutputData[i] = outputData->at(0);	
	}

	double gatheredWeights[2][4]; //[layer][weight]
	
	for(unsigned int i = 0; i < hiddenLayer.size(); i++){
		gatheredWeights[0][i * 2] = hiddenLayer[i]->getWeight(0);
		gatheredWeights[0][i * 2 + 1] = hiddenLayer[i]->getWeight(1);
	}
	for(unsigned int i = 0; i < outputLayer.size(); i++){
		gatheredWeights[1][i * 2] = outputLayer[i]->getWeight(0);
		gatheredWeights[1][i * 2 + 1] = outputLayer[i]->getWeight(1);
	}

	double *dev_input = 0;
    double *dev_output = 0;
    double *dev_weights = 0;
	cudaError_t cudaStatus;

	cudaStatus = cudaMalloc((void**)&dev_input, 8 * sizeof(double));
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }

	cudaStatus = cudaMalloc((void**)&dev_output, 4 * sizeof(double));
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }

	size_t pitch;
	cudaStatus = cudaMallocPitch((void**)&dev_weights, &pitch, 4 * sizeof(double), 2);
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }

	cudaStatus = cudaMemcpy(dev_input, gatheredInputData, 8 * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }

	cudaStatus = cudaMemcpy(dev_output, gatheredOutputData, 4 * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }

	cudaStatus = cudaMemcpy2D(dev_weights, pitch, gatheredWeights, 4 * sizeof(double), 4 * sizeof(double), 2, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }

	//for(int pass = 0; pass < 1000; pass++){
	//	printf("Training AI... %d. pass", pass);

		backpropagationPass<<<1, 4>>>(dev_input, dev_output, dev_weights, pitch);
			
		/*	//calculate new weights and correct the old ones
			for(unsigned int k = 0; k < outputLayer.size(); k++){
				//outputLayer[k]->calculateWeights(learningRate, NULL, targetOutput[k]);
			}

			for(unsigned int k = 0; k < hiddenLayer.size(); k++){
				hiddenLayer[k]->calculateWeights(learningRate, &outputLayer, 0.0);
			}

			//set corrected weights
			for(unsigned int k = 0; k < outputLayer.size(); k++){
				outputLayer[k]->correctWeights();
				hiddenLayer[k]->correctWeights();
			}
		}*/
		//pass++;
		printf("\n");
	//}
	cudaStatus = cudaMemcpy(gatheredInputData, dev_input, 8 * sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(dev_input);
	cudaFree(dev_output);
	cudaFree(dev_weights);
	
	printf("Done training AI!\n");
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
