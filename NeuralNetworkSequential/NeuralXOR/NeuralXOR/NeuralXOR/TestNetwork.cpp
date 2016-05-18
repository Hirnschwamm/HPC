#include "stdafx.h"
#include "TestNetwork.h"

#include <cstdio>

#include "rapidjson\document.h"
#include "rapidjson\writer.h"
#include "rapidjson\stringbuffer.h"
#include "rapidjson\filereadstream.h"



TestNetwork::TestNetwork(void)
{ 	
	inputLayer.push_back(new Perceptron(0.05));
	inputLayer.push_back(new Perceptron(0.10));


	hiddenLayer.push_back(new Perceptron(inputLayer));
	hiddenLayer.push_back(new Perceptron(inputLayer));
	
	hiddenLayer[0]->setBias(0.35);
	hiddenLayer[1]->setBias(0.35);
	
	hiddenLayer[0]->setWeight(0, 0.15);
	hiddenLayer[0]->setWeight(1, 0.20);
	hiddenLayer[1]->setWeight(0, 0.25);
	hiddenLayer[1]->setWeight(1, 0.30);


	outputLayer.push_back(new Perceptron(hiddenLayer));
	outputLayer.push_back(new Perceptron(hiddenLayer));

	outputLayer[0]->setBias(0.60);
	outputLayer[1]->setBias(0.60);

	outputLayer[0]->setWeight(0, 0.40);
	outputLayer[0]->setWeight(1, 0.45);
	outputLayer[1]->setWeight(0, 0.50);
	outputLayer[1]->setWeight(1, 0.55);

	/*
	//forward pass
	double o1, o2;
	o1 = outputLayer[0]->getOutput();
	o2 = outputLayer[1]->getOutput();

	//Calculating the total error
	double totalError = 0.0;
	totalError += 0.5 * ((0.01 - o1) * (0.01 - o1));
	totalError += 0.5 * ((0.99 - o2) * (0.99 - o2));

	
	//calculate new weights and correct the old ones
	outputLayer[0]->calculateWeights(0.5, NULL, 0.01);
	outputLayer[1]->calculateWeights(0.5, NULL, 0.99);
	
	
	hiddenLayer[0]->calculateWeights(0.5, &outputLayer, 0.0f);
	hiddenLayer[1]->calculateWeights(0.5, &outputLayer, 0.0f);

	
	outputLayer[0]->correctWeights();
	outputLayer[1]->correctWeights();
	
	hiddenLayer[0]->correctWeights();
	hiddenLayer[1]->correctWeights();

	double totalError1 = 0.0;
	o1 = outputLayer[0]->getOutput();
	o2 = outputLayer[1]->getOutput();
	totalError1 += 0.5 * ((0.01 - o1) * (0.01 - o1));
	totalError1 += 0.5 * ((0.99 - o2) * (0.99 - o2));

	double totalError2 = totalError - totalError1;

	int dbg = 0;
	*/
	
	initjsonTrainingData("TestData.json");

	trainByBackpropagation(1, 0.5);
	int dbg = 0;
}


TestNetwork::~TestNetwork(void)
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

void TestNetwork::trainByBackpropagation(int numPasses, double learningRate){
	for(int i = 0; i < numPasses; i++){	
		printf("Training AI... %d. pass", i);
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

			printf(" | %d. move, Error: %f", j, totalError);
			
			//calculate new weights and correct the old ones
			for(unsigned int k = 0; k < outputLayer.size(); k++){
				outputLayer[k]->calculateWeights(learningRate, NULL, targetOutput[k]);
			}

			for(unsigned int k = 0; k < hiddenLayer.size(); k++){
				hiddenLayer[k]->calculateWeights(learningRate, &outputLayer, 0.0);
			}

			//set corrected weights
			for(unsigned int k = 0; k < outputLayer.size(); k++){
				outputLayer[k]->correctWeights();
				hiddenLayer[k]->correctWeights();
			}
		}
		printf("\n");
	}
	
	printf("Done training AI!\n");
}

void TestNetwork::setInput(std::vector<double>& input){
	for(unsigned int i = 0; i < input.size(); i++){
		inputLayer[i]->setDirectInput(input[i]);
	}
}

void TestNetwork::initjsonTrainingData(std::string path){
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

