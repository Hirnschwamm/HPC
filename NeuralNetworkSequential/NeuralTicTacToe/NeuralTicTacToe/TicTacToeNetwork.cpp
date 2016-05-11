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

void TicTacToeNetwork::trainByBackpropagation(int numPasses, double learningRate){
	
	for(int i = 0; i < numPasses; i++){
		double totalError = 0.0;
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

			//calculate new weights and correct the old ones
			for(unsigned int k = 0; k < outputLayer.size(); k++){
				outputLayer[k]->calculateWeights(learningRate, NULL, targetOutput[k]);
			}

			for(unsigned int k = 0; k < hiddenLayer.size(); k++){
				hiddenLayer[k]->calculateWeights(learningRate, &outputLayer, 0.0);
			}

			for(unsigned int k = 0; k < outputLayer.size(); k++){
				outputLayer[k]->correctWeights();
				hiddenLayer[k]->correctWeights();
			}
		}
		printf("Training AI... %d. move, Error: %f\n", i, totalError);
	}
	
	printf("Done training AI!\n");
}

unsigned int TicTacToeNetwork::getIndexforNextToken(std::vector<Faction> input){
	std::vector<double> inputVec; 
	for(int i = 0; i < 9; i++){
		if(input[i] == PLAYER){
			inputVec.push_back(1.0);
		}else{
			inputVec.push_back(0.05);
		}
	}
	setInput(inputVec);

	//DEBUG
	std::vector<double> dbg;
	for(unsigned int i = 0; i < outputLayer.size(); i++){
		dbg.push_back(outputLayer[i]->getOutput());
	}
	//DEBUG

	double max = 0.0;
	unsigned int currentBest = 0;
	for(unsigned int i = 0; i < outputLayer.size(); i++){
		double output = outputLayer[i]->getOutput();
		if(output >= max){
			max = output;
			currentBest = i;
		}
	}

	return currentBest;
}

void TicTacToeNetwork::setInput(std::vector<double> input){
	for(unsigned int i = 0; i < input.size(); i++){
		inputLayer[i]->setDirectInput(input[i]);
	}
}

void TicTacToeNetwork::initRandomTrainingData(int num){
	int randNum = 0;
	for(int i = 0; i < num; i++){
		std::vector<double> inputData;
		std::vector<double> targetData;
		for(int j = 0; j < 9; j++){
			randNum = rand();
			if((double)randNum > (double)RAND_MAX/2.0){
				inputData.push_back(1.0);
			}else{
				inputData.push_back(0.0);
			}
			targetData.push_back(1.0); //TODO: get targetData from oracle
		}
		trainingData.push_back(std::tuple<std::vector<double>, std::vector<double>>(inputData, targetData));
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
	std::vector<double> input;
	std::vector<double> output;
	for (rapidjson::SizeType i = 0; i < moves.Size(); i++){
		for(rapidjson::SizeType j = 0; j < moves[i]["input"].Size(); j++){
			input.push_back( moves[i]["input"][j].GetDouble());
		}
		for(rapidjson::SizeType j = 0; j < moves[i]["input"].Size(); j++){
			output.push_back( moves[i]["output"][j].GetDouble());
		}
		trainingData.push_back(std::tuple<std::vector<double>, std::vector<double>>(input, output));
		input.clear();
		output.clear();
	}
}
