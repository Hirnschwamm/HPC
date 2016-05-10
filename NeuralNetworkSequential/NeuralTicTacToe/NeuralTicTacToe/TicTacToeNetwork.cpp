#include "stdafx.h"
#include "TicTacToeNetwork.h"

#include <cstdio>

#include "TicTacToe.h"
#include "rapidjson\document.h"
#include "rapidjson\writer.h"
#include "rapidjson\stringbuffer.h"
#include "rapidjson\filereadstream.h"


TicTacToeNetwork::TicTacToeNetwork(void){
	int dimension = dim;
	int dimensionSqrd = dimension * dimension;
	
	for(int i = 0; i < dimensionSqrd; i++){
		inputLayer.push_back(new Perceptron(0.0f));
	}

	for(int i = 0; i < dimensionSqrd; i++){
		hiddenLayer.push_back(new Perceptron(inputLayer));
	}

	for(int i = 0; i < dimensionSqrd; i++){
		outputLayer.push_back(new Perceptron(hiddenLayer));
	}

	//initRandomTrainingData(1000); 

	initjsonTrainingData("TrainingData.json");
}


TicTacToeNetwork::~TicTacToeNetwork(void)
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

void TicTacToeNetwork::trainByBackpropagation(int numPasses, float learningRate){
	
	float totalError = 0.0f;
	for(unsigned int j = 0; j < trainingData.size(); j++){
		setInput(std::get<0>(trainingData[j]));
		std::vector<float> targetOutput = std::get<1>(trainingData[j]);
	
		for(int i = 0; i < numPasses; i++){
			//forward pass
			std::vector<float> totalNetOutput;
			for(unsigned int k = 0; k < outputLayer.size(); k++){
				totalNetOutput.push_back(outputLayer[k]->getOutput());
			}

			//Calculating the total error
			totalError = 0.0f;
			for(unsigned int k = 0; k < totalNetOutput.size(); k++){
				totalError += 0.5f * ((targetOutput[k] - totalNetOutput[k]) * (targetOutput[k] - totalNetOutput[k]));
			}

			//calculate new weights and correct the old ones
			for(unsigned int k = 0; k < outputLayer.size(); k++){
				outputLayer[k]->calculateWeights(learningRate, NULL, targetOutput[k]);
			}

			for(unsigned int k = 0; k < hiddenLayer.size(); k++){
				hiddenLayer[k]->calculateWeights(learningRate, &outputLayer, 0.0f);
			}

			for(unsigned int k = 0; k < outputLayer.size(); k++){
				outputLayer[k]->correctWeights();
				hiddenLayer[k]->correctWeights();
			}
		}
		printf("Training AI... %d. move, Error: %f\n", j, totalError);
	}
	
	printf("Done training AI!\n");
}

int TicTacToeNetwork::getIndexforNextToken(std::vector<Faction> input){
	std::vector<float> inputVec; 
	for(int i = 0; i < 9; i++){
		if(input[i] == PLAYER){
			inputVec.push_back(1.0f);
		}else{
			inputVec.push_back(0.05f);
		}
	}
	setInput(inputVec);

	for(unsigned int i = 0; i < outputLayer.size(); i++){
		float output = outputLayer[i]->getOutput();
		if(output >= 0.49f && output <= 0.51f){
			return i;
		}
	}

	return 0;
}

void TicTacToeNetwork::setInput(std::vector<float> input){
	for(unsigned int i = 0; i < input.size(); i++){
		inputLayer[i]->setDirectInput(input[i]);
	}
}

void TicTacToeNetwork::initRandomTrainingData(int num){
	int randNum = 0;
	for(int i = 0; i < num; i++){
		std::vector<float> inputData;
		std::vector<float> targetData;
		for(int j = 0; j < 9; j++){
			randNum = rand();
			if((float)randNum > (float)RAND_MAX/2.0f){
				inputData.push_back(1.0f);
			}else{
				inputData.push_back(0.0f);
			}
			targetData.push_back(1.0f); //TODO: get targetData from oracle
		}
		trainingData.push_back(std::tuple<std::vector<float>, std::vector<float>>(inputData, targetData));
	}
}

void TicTacToeNetwork::initjsonTrainingData(std::string path){
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
	std::vector<float> input;
	std::vector<float> output;
	for (rapidjson::SizeType i = 0; i < moves.Size(); i++){
		for(rapidjson::SizeType j = 0; j < moves[i]["input"].Size(); j++){
			input.push_back( moves[i]["input"][j].GetFloat());
		}
		for(rapidjson::SizeType j = 0; j < moves[i]["input"].Size(); j++){
			output.push_back( moves[i]["output"][j].GetFloat());
		}
		trainingData.push_back(std::tuple<std::vector<float>, std::vector<float>>(input, output));
		input.clear();
		output.clear();
	}
}
