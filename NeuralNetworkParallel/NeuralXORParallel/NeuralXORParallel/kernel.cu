#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "XORNetwork.h"

#include <device_functions.h>
#include <iostream>
#include <stdio.h>

int main()
{
	XORNetwork network("XORData.json");
	network.trainByBackpropagation(100000, 0.5);

	int o1, o2, input;
	do{
		do{
			printf("\nOperand 1: ");
			std::cin >> input;
		}while(input != 1 && input != 0 && input != -1);
		o1 = input;

		do{
			printf("\nOperand 2: ");
			std::cin >> input;
		}while(input != 1 && input != 0 && input != -1);
		o2 = input;
		
		double result = network.xor(o1, o2);

		printf("\n%d XOR %d = %f\n", o1, o2, result);
	}while(input != -1);
	system("pause");

	cudaError cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}
