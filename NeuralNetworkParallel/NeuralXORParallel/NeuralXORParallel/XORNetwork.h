#pragma once

#include "BackpropagationKernel.cuh"
#include "Perceptron.h"

class XORNetwork
{
public:
	XORNetwork(std::string dataPath);
	~XORNetwork(void);

	void trainByBackpropagation(unsigned int passes, double learningRate);
	double xor(int operand1, int operand2);

private:
	std::vector<Perceptron*> inputLayer;
	std::vector<Perceptron*> hiddenLayer;
	std::vector<Perceptron*> outputLayer;
	std::vector<std::tuple<std::vector<double>, std::vector<double>>> trainingData;

	void setInput(std::vector<double>& input);
	void initjsonTrainingData(std::string path);
};

