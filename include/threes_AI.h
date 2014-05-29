#ifndef threes_AI_threes_AI_h
#define threes_AI_threes_AI_h
#include <threes_IO.h>
#include <threes_Mechanics.h>

/*
 *******************************************************************************
 * function declaration *
 *******************************************************************************
*/

/* dfs on board for move path */
std::vector<Node> dfs(Board &);
Direction greedy_search2(Board, int);
Direction greedy_search(Board, int);
// int greedy_search(Board &, int, int);
int a_star(Board, std::vector<std::string> &, int, int *);
int i_aStar(Board &, std::vector<std::string> &);
std::vector<Direction> informed(Board, int);

#endif