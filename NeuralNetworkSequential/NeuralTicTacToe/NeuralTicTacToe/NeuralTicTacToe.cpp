// NeuralTicTacToe.cpp : Definiert den Einstiegspunkt für die Konsolenanwendung.
//

#include "stdafx.h"
#include "TicTacToe.h"
#include "TicTacToeNetwork.h"
#include "TestNetwork.h"

int _tmain(int argc, _TCHAR* argv[])
{
	//TestNetwork();
	TicTacToe ttt;
	TicTacToeNetwork ai;
	ai.trainByBackpropagation(10000, 1.0f);
	int playerIndex = 0;
	unsigned int aiIndex = 0;
	printf("Welcome! Lets play a match of tic tac toe, shall we?\n");
	do{
		ttt.printField();
		
		printf("\nPlace your token!\n");
		std::cin >> playerIndex;
		if(playerIndex >= 0 && playerIndex < 9){
			ttt.placeToken(playerIndex, PLAYER);
		}else if(playerIndex == 9){
			ttt.clearField();
			continue;
		}

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
		}else if(winningFaction == NONE && ttt.fieldFull()){
			ttt.printField();
			printf("It's a draw!\n");
			break;
		}

	}while(true);
	system("pause");
}

