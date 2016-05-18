// NeuralXOR.cpp : Definiert den Einstiegspunkt für die Konsolenanwendung.
//

#include "stdafx.h"
#include "XORNetwork.h"
#include "TestNetwork.h"

int _tmain(int argc, _TCHAR* argv[])
{
	//TestNetwork();
	XORNetwork network("XORData.json");
	network.trainByBackpropagation(0.01, 0.7);

	int o1, o2, input;
	do{
		do{
			printf("\nOperand 1: ");
			std::cin >> input;
		}while(input != 1 && input != 0);
		o1 = input;

		do{
			printf("\nOperand 2: ");
			std::cin >> input;
		}while(input != 1 && input != 0);
		o2 = input;
		
		double result = network.xor(o1, o2);

		printf("\n%d XOR %d = %f\n", o1, o2, result);
	}while(true);
	system("pause");
}

