#pragma once

#include "Perceptron.h"
#include "TicTacToe.h"

class TicTacToeNetwork
{
public:
	TicTacToeNetwork(void);
	~TicTacToeNetwork(void);

	void trainByBackpropagation(int numPasses, float learningRate);
	int getIndexforNextToken(std::vector<Faction> input);

private:
	std::vector<Perceptron*> inputLayer;
	std::vector<Perceptron*> hiddenLayer;
	std::vector<Perceptron*> outputLayer;
	std::vector<std::tuple<std::vector<float>, std::vector<float>>> trainingData;

	void setInput(std::vector<float> input);
	void initRandomTrainingData(int num);
	void initjsonTrainingData(std::string path);

	
};

