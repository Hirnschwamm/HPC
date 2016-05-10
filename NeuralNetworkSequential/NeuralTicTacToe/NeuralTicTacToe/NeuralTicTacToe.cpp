// NeuralTicTacToe.cpp : Definiert den Einstiegspunkt für die Konsolenanwendung.
//

#include "stdafx.h"
#include "TicTacToe.h"
#include "TicTacToeNetwork.h"


int _tmain(int argc, _TCHAR* argv[])
{
	TicTacToe ttt;
	TicTacToeNetwork ai;
	ai.trainByBackpropagation(5000, 0.5f);
	int playerIndex = 0;
	int aiIndex = 0;
	printf("Welcome! Lets play a match of tic tac toe, shall we?\n");
	do{
		ttt.printField();
		
		printf("\nPlace your token!\n");
		std::cin >> playerIndex;
		ttt.placeToken(playerIndex, PLAYER);

		std::vector<Faction> field = ttt.getFormattedField();
		aiIndex = ai.getIndexforNextToken(field);
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

