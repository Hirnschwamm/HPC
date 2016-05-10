#include "stdafx.h"
#include "Perceptron.h"


Perceptron::Perceptron(void)
{
	weightedSum = 0.0f;
}

Perceptron::Perceptron(std::vector<Perceptron*> inputs){
	weightedSum = 0.0f;
	this->inputs = inputs;
	initWeightsAtRandom(inputs.size());
	directInput = -1.0f;
	lastDelta = 0.0f;
}

Perceptron::Perceptron(float directInput){
	weightedSum = 0.0f;
	this->directInput = directInput;
	lastDelta = 0.0f;
}

Perceptron::~Perceptron(void)
{
}

float Perceptron::getOutput(){
	calculateWeightedSum();
	if(calculateActivationFunc()){
		return weightedSum;
	}else{
		return 0.0f;
	}
}

int Perceptron::getNumInputs(){
	return inputs.size();
}
	
float Perceptron::getWeight(int index){
	return weights[index];
}

float Perceptron::getLastDelta(){
	return lastDelta;
}

void Perceptron::setWeight(int index, float weight){
	weights[index] = weight;
}

void Perceptron::setDirectInput(float input){
	directInput = input;
}

void Perceptron::calculateWeights(float learningRate, std::vector<Perceptron*> const * predecessors, float target){
	
	float correction = 0.0f;
	if(predecessors){
		float deltaPredecessors = 0.0f;
		float predecessorOutput = 0.0f;
		for(unsigned int i = 0; i < predecessors->size(); i++){
			predecessorOutput = predecessors->at(i)->getOutput();
			deltaPredecessors += predecessors->at(i)->getLastDelta() * weights[i];
		}

		for(unsigned int i = 0; i < inputs.size(); i++){
			correction = deltaPredecessors * weightedSum * (1.0f - weightedSum) * inputs[i]->getOutput();
			weightBuffer[i] = weights[i] - learningRate * correction;
		}

	}else{
		for(unsigned int i = 0; i < inputs.size(); i++){
			lastDelta = -(target - weightedSum) * weightedSum * (1.0f - weightedSum);
			correction = lastDelta * inputs[i]->getOutput();
			weightBuffer[i] = weights[i] - learningRate * correction;
		}
	}
	
}

void Perceptron::correctWeights(){
	weights = weightBuffer;
}

void Perceptron::calculateWeightedSum(){
	weightedSum = 0.0f;

	if(directInput < 0.0f){
		for(unsigned int i = 0; i < inputs.size(); i++){
			weightedSum += inputs[i]->getOutput() * weights[i];
		}
		weightedSum = calculateActivationFunc();
	}else{
		weightedSum = directInput;
	}
}

float Perceptron::calculateActivationFunc(){
	return 1.0f / 1.0f + -std::pow(euler, weightedSum);
}

void Perceptron::initWeightsAtRandom(int num){
	for(int i = 0; i < num; i++){
		weights.push_back((float)rand() / ((float)RAND_MAX / 0.5f));
		weightBuffer.push_back(0.0f);
	}
}