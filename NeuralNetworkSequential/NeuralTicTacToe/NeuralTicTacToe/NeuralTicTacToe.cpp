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
	ai.trainByBackpropagation(5000, 0.5f);
	int playerIndex = 0;
	unsigned int aiIndex = 0;
	printf("Welcome! Lets play a match of tic tac toe, shall we?\n");
	do{
		ttt.printField();
		Faction winningFaction = ttt.checkForWin();
		if(winningFaction == PLAYER){
			printf("Congrats! You won!\n");
			ttt.clearField();
			continue;
		}else if(winningFaction == AI){
			printf("You lost!\n");
			ttt.clearField();
			continue;
		}else if(winningFaction == NONE && ttt.fieldFull()){
			printf("It's a draw!\n");
			ttt.clearField();
			continue;
		}
		
		printf("\nPlace your token!\n");
		std::cin >> playerIndex;
		if(playerIndex >= 0 && playerIndex < 9){
			ttt.placeToken(playerIndex, PLAYER);
		}else if(playerIndex == 9){
			ttt.clearField();
			continue;
		}else if(playerIndex == 10){
			break;
		}

		std::vector<Faction> field = ttt.getFormattedField();
		aiIndex = ai.getIndexforNextToken(field);
		ttt.placeToken(aiIndex, AI);

	}while(true);
	system("pause");
}

