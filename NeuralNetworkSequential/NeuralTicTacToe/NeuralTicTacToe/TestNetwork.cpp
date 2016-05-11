#include "stdafx.h"
#include "TestNetwork.h"


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
