#pragma once

#define dim 3;

enum Faction
{
	NONE,
	PLAYER,
	AI
};

class TicTacToe
{
public:
	TicTacToe(void);
	~TicTacToe(void);

	void placeToken(unsigned int fieldIndex, Faction faction);
	std::vector<Faction> getFormattedField();
	Faction checkForWin();
	void clearField();
	void printField();

private:
	std::vector<std::vector<Faction>> field; 
};

