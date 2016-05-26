#pragma once

#define euler 2.71828

class Perceptron
{
public:
	Perceptron();
	Perceptron(std::vector<Perceptron*> inputs);
	Perceptron(double directInput);
	~Perceptron(void);

	double getOutput();

	int getNumInputs();
	double getWeight(int index);
	double getLastDelta();

	void setWeight(int index, double weight);
	void setBias(double bias);
	void setDirectInput(double input);

	void calculateWeights(double learningRate, std::vector<Perceptron*> const * predecessors, int index, double target);
	void correctWeights();

private:
	std::vector<Perceptron*> inputs;
	double directInput;
	std::vector<double> weights;
	std::vector<double> weightBuffer;
	double biasWeight;
	double biasBuffer;
	double weightedSum;
	double lastDelta;

	void calculateWeightedSum();
	double calculateActivationFunc();
	void initWeightsAtRandom(int num);
	double clampWeight(double weight);
};

