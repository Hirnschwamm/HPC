#pragma once
class Perceptron
{
public:
	Perceptron();
	Perceptron(std::vector<Perceptron*> inputs);
	Perceptron(float directInput);
	~Perceptron(void);

	float getOutput();

	int getNumInputs();
	float getWeight(int index);
	float getLastDelta();

	void setWeight(int index, float weight);
	void setDirectInput(float input);

	void calculateWeights(float learningRate, std::vector<Perceptron*> const * predecessors, float target);
	void correctWeights();

private:
	std::vector<Perceptron*> inputs;
	float directInput;
	std::vector<float> weights;
	std::vector<float> weightBuffer;
	float weightedSum;
	float lastDelta;

	void calculateWeightedSum();
	float calculateActivationFunc();
	void initWeightsAtRandom(int num);
};

