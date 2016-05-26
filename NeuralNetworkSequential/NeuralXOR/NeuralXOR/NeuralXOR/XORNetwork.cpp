#include "stdafx.h"
#include "XORNetwork.h"

#include <cstdio>

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
	double highestError = errorTolerance + 0.00001;
	int pass = 0;
	for(pass; pass < 100000; pass++){	
		printf("Training AI... %d. pass", pass);
		
		double totalError = 0.0;
		highestError = 0.0;
		for(unsigned int j = 0; j < trainingData.size(); j++){
		
			setInput(std::get<0>(trainingData[j]));
			std::vector<double> targetOutput = std::get<1>(trainingData[j]);

			//forward pass
			std::vector<double> totalNetOutput;
			for(unsigned int k = 0; k < outputLayer.size(); k++){
				totalNetOutput.push_back(outputLayer[k]->getOutput());
			}

			//Calculating the total error
			totalError = 0.0;
			for(unsigned int k = 0; k < totalNetOutput.size(); k++){
				totalError += 0.5 * ((targetOutput[k] - totalNetOutput[k]) * (targetOutput[k] - totalNetOutput[k]));
			}

			printf(" | %d. move, Error: %f", j, totalError);

			if(totalError > highestError){
				highestError = totalError;
			}
			
			//calculate new weights and correct the old ones
			for(unsigned int k = 0; k < outputLayer.size(); k++){
				outputLayer[k]->calculateWeights(learningRate, NULL, k, targetOutput[k]);
			}

			for(unsigned int k = 0; k < hiddenLayer.size(); k++){
				hiddenLayer[k]->calculateWeights(learningRate, &outputLayer, k, 0.0);
			}

			//set corrected weights
			for(unsigned int k = 0; k < outputLayer.size(); k++){
				outputLayer[k]->correctWeights();
				//hiddenLayer[k]->correctWeights();
			}

			for(unsigned int k = 0; k < hiddenLayer.size(); k++){
				hiddenLayer[k]->correctWeights();
			}
		}
		pass++;
		printf("\n");
	}
	
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