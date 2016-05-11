#pragma once

#include "Perceptron.h"
#include "TicTacToe.h"

class TicTacToeNetwork
{
public:
	TicTacToeNetwork(void);
	~TicTacToeNetwork(void);

	void trainByBackpropagation(int numPasses, double learningRate);
	unsigned int getIndexforNextToken(std::vector<Faction> input);

private:
	std::vector<Perceptron*> inputLayer;
	std::vector<Perceptron*> hiddenLayer;
	std::vector<Perceptron*> outputLayer;
	std::vector<std::tuple<std::vector<double>, std::vector<double>>> trainingData;

	void setInput(std::vector<double> input);
	void initRandomTrainingData(int num);
	void initjsonTrainingData(std::string path);

	
};

