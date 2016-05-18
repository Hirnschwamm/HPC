#pragma once

#include "Perceptron.h"

class TestNetwork
{
public:
	TestNetwork(void);
	~TestNetwork(void);

	void trainByBackpropagation(int numPasses, double learningRate);

private:
	std::vector<Perceptron*> inputLayer;
	std::vector<Perceptron*> hiddenLayer;
	std::vector<Perceptron*> outputLayer;
	std::vector<std::tuple<std::vector<double>, std::vector<double>>> trainingData;

	void setInput(std::vector<double>& input);
	void initjsonTrainingData(std::string path);
	
};

