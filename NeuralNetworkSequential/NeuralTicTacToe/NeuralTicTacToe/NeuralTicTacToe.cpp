// NeuralTicTacToe.cpp : Definiert den Einstiegspunkt für die Konsolenanwendung.
//

#include "stdafx.h"
#include "TicTacToe.h"
#include "TicTacToeNetwork.h"


int _tmain(int argc, _TCHAR* argv[])
{
	TicTacToe ttt;
	TicTacToeNetwork ai;
	ai.trainByBackpropagation(1000, 0.5f);
	int playerIndex = 0;
	int aiIndex = 0;
	bool aiInput[9];
	printf("Welcome! Lets play a match of tic tac toe, shall we?\n");
	do{
		ttt.printField();
		
		printf("\nPlace your token!\n");
		std::cin >> playerIndex;
		ttt.placeToken(playerIndex, PLAYER);

		std::vector<Faction> field = ttt.getFormattedField();
		for(unsigned int i = 0; i < field.size(); i++){
			if(field[i] == PLAYER){
				aiInput[i] = true;
			}else{
				aiInput[i] = false;
			}
		}

		aiIndex = ai.getIndexforNextToken(aiInput);
		ttt.placeToken(aiIndex, AI);

		Faction winningFaction = ttt.checkForWin();
		if(winningFaction == PLAYER){
			ttt.printField();
			printf("Congrats! You won!\n");
			break;
		}else if(winningFaction == AI){
			ttt.printField();
			printf("You lost!\n");
			break;
		}

	}while(true);
	system("pause");
}

