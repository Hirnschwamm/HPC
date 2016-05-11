#include "stdafx.h"
#include "Perceptron.h"


Perceptron::Perceptron(void)
{
	weightedSum = 0.0;
}

Perceptron::Perceptron(std::vector<Perceptron*> inputs){
	weightedSum = 0.0;
	this->inputs = inputs;
	initWeightsAtRandom(inputs.size());
	directInput = -1.0;
	lastDelta = 0.0;
	biasWeight = 0.0;
}

Perceptron::Perceptron(double directInput){
	weightedSum = 0.0;
	this->directInput = directInput;
	lastDelta = 0.0;
	biasWeight = 0.0;
}

Perceptron::~Perceptron(void)
{
}

double Perceptron::getOutput(){
	calculateWeightedSum();
	if(calculateActivationFunc()){
		return weightedSum;
	}else{
		return 0.0;
	}
}

int Perceptron::getNumInputs(){
	return inputs.size();
}
	
double Perceptron::getWeight(int index){
	return weights[index];
}

double Perceptron::getLastDelta(){
	return lastDelta;
}

void Perceptron::setWeight(int index, double weight){
	weights[index] = weight;
}

void Perceptron::setBias(double bias){
	biasWeight = bias;
}

void Perceptron::setDirectInput(double input){
	directInput = input;
}

void Perceptron::calculateWeights(double learningRate, std::vector<Perceptron*> const * predecessors, double target){
	
	double correction = 0.0;
	if(predecessors){
		double deltaPredecessors = 0.0;
		double predecessorOutput = 0.0;
		for(unsigned int i = 0; i < predecessors->size(); i++){
			predecessorOutput = predecessors->at(i)->getOutput();
			deltaPredecessors += predecessors->at(i)->getLastDelta() * weights[i];
		}

		for(unsigned int i = 0; i < inputs.size(); i++){
			correction = deltaPredecessors * weightedSum * (1.0 - weightedSum) * inputs[i]->getOutput();
			weightBuffer[i] = clampWeight(weights[i] - learningRate * correction);
		}
	}else{
		for(unsigned int i = 0; i < inputs.size(); i++){
			lastDelta = -(target - weightedSum) * weightedSum * (1.0 - weightedSum);
			correction = lastDelta * inputs[i]->getOutput();
			weightBuffer[i] = clampWeight(weights[i] - learningRate * correction);
		}
	}
	
}

void Perceptron::correctWeights(){
	weights = weightBuffer;
}

void Perceptron::calculateWeightedSum(){
	weightedSum = 0.0;

	if(directInput < 0.0){
		for(unsigned int i = 0; i < inputs.size(); i++){
			weightedSum += inputs[i]->getOutput() * weights[i];
		}
		weightedSum += biasWeight * 1.0;
		weightedSum = calculateActivationFunc();
	}else{
		weightedSum = directInput;
	}
}

double Perceptron::calculateActivationFunc(){
	return 1.0 / (1.0 + std::pow(euler, -weightedSum));
}

void Perceptron::initWeightsAtRandom(int num){
	for(int i = 0; i < num; i++){
		weights.push_back((double)rand() / ((double)RAND_MAX / 0.5));
		weightBuffer.push_back(0.0);
	}
}

double Perceptron::clampWeight(double weight){
	if(weight > 1.0){
		return 1.0;
	}else if(weight < 0.0){
		return 0.0;
	}else{
		return weight;
	}
}