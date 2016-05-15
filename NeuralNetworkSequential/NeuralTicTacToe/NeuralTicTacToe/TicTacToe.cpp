#include "stdafx.h"
#include "TicTacToe.h"


TicTacToe::TicTacToe(void)
{
	unsigned int dimension = dim;
	for(unsigned int i = 0; i < dimension; i++){
		field.push_back(std::vector<Faction>());
		for(unsigned int j = 0; j < dimension; j++){
			field[i].push_back(NONE);
		}
	}
}

TicTacToe::~TicTacToe(void)
{
}

void TicTacToe::placeToken(unsigned int fieldIndex, Faction faction){
	unsigned int row = fieldIndex / 3;
	unsigned int column = fieldIndex % 3;
	field[row][column] = faction;
}

Faction TicTacToe::checkForWin(){
	for(unsigned int i = 0; i < field.size(); i++){
		//check for horizontal win
		if(field[i][0] == field[i][1] && field[i][0] == field[i][2] && field[i][1] == field[i][2] &&
		   field[i][0] != NONE){
			return field[i][0];
		}

		//check for vertical win
		if(field[0][i] == field[1][i] && field[0][i] == field[2][i] && field[1][i] == field[2][i] &&
		   field[0][i] != NONE){
			return field[0][i];
		}
	}

	//check for diagonal win
	if((field[0][0] == field[1][1] && field[0][0] == field[2][2] && field[1][1] == field[2][2]) ||
       (field[0][2] == field[1][1]  && field[0][2] == field[2][0] && field[1][1] == field[2][0]) &&
	   field[1][1] != NONE){
		   return field[1][1];
	}

	return NONE;
}

bool TicTacToe::fieldFull(){
	bool full = true;
	
	unsigned int dimension = dim;
	for(unsigned int i = 0; i < dimension; i++){
		for(unsigned int j = 0; j < dimension; j++){
			full &= (field[i][j] == AI || field[i][j] == PLAYER);
		}
	}

	return full;
}

void TicTacToe::clearField(){
	unsigned int dimension = dim;
	for(unsigned int i = 0; i < dimension; i++){
		for(unsigned int j = 0; j < dimension; j++){
			field[i][j] = NONE;
		}
	}
}

std::vector<Faction> TicTacToe::getFormattedField(){
	std::vector<Faction> f;
	for(unsigned int i = 0; i < field.size(); i++){
		for(unsigned int j = 0; j < field[i].size(); j++){
			f.push_back(field[i][j]);
		}
	}
	return f;
}

void TicTacToe::printField(){
	for(unsigned int i = 0; i < field.size(); i++){
		for(unsigned int j = 0; j < field[i].size(); j++){
			switch(field[i][j]){
			case(NONE):
				printf(" . ");
				break;
			case(PLAYER):
				printf(" X ");
				break;
			case(AI):
				printf(" O ");
				break;
			}
		}
		printf("\n");
	}
}



