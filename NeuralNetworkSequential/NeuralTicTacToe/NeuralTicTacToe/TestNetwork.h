#pragma once

#include "Perceptron.h"

class TestNetwork
{
public:
	TestNetwork(void);
	~TestNetwork(void);

	void trainByBackpropagation(int numPasses, float learningRate);

private:
	std::vector<Perceptron*> inputLayer;
	std::vector<Perceptron*> hiddenLayer;
	std::vector<Perceptron*> outputLayer;
	
};

