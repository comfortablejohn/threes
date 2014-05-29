/**
 * 
 */
#ifndef threes_Mechanics_threes_Mechanics_h
#define threes_Mechanics_threes_Mechanics_h
/** Includes **/ 
#include <algorithm>

#include <vector>
#include <set>
#include <queue>
#include <stack>
#include <map>
#include <unordered_map>

#include <string> 

#include <iomanip>
#include <iostream>
#include <sstream>
#include <fstream> 
 
#include <cstdlib>
#include <cmath>
#include <ctime>

/** Defines **/ 
#define EMPTY 0
#define BOARD_SIZE 4

extern std::vector<int> inputSequence;
extern int tile_num;

/** 
 * move direction representation
 */
enum Direction {
  U, D, L, R
};

class comparator {
  bool reverse;
public:
  comparator(const bool & revparam=true){
    reverse=revparam;
  }
  bool operator() (const std::pair<int,Direction>& m1, const std::pair<int, Direction>& m2) {
    if (reverse) return m1.first > m2.first;
    return m1.first < m2.first;
  }
};

/** Board data structure **/ 
typedef std::vector< std::vector<int> > Board;


typedef std::priority_queue<std::pair<int, Direction>, std::vector<std::pair<int, Direction> >, comparator> PQ;
/** Node for DFS tree **/ 
struct Node {
  PQ poss_moves;
  Node *parent = nullptr;
  // the direction moved to get from parent-->node
  Direction moveMade;
  Board b;
  int depth;
  float f;
  int g;
  int h;
  int score;
  // unique identifier for comparing nodes
  int id;
  std::string str;
  bool isRoot;
};

class nodeComparator {
  bool reverse;
public:
  nodeComparator(const bool & revparam=false) {
    reverse=revparam;
  }

  bool operator() (const Node &n1, const Node &n2) {
    if (reverse) return n1.f < n2.f;
    return n2.f < n1.f;
  }
};

typedef std::priority_queue<Node, std::vector<Node>, nodeComparator> NodeQ;

/**
 * data structure to represent shift.
 */
struct Shift {
  int id; // describes the row/col that was shifted (id in [0-3])
  Direction m; // determines if id is row or col (D/U: col, L/R: row)
  // representation of shift row/col - allows lexicographic sorting of shifts
  std::vector<int> string_vec;

  // constructor
  Shift(Direction mm, int idid) : id(idid), m(mm) {}

  bool operator<(const Shift &a) const {
    int i = 0;
    while (i < string_vec.size() && string_vec[i] == a.string_vec[i]) {
      i++;
    }
    if (i == string_vec.size()) {
      // if move is U or R, we want to select lower IDs 
      // (left most col, top most row)
      if (m == U || m == R) return a.id > id;
      // if move is D or L, we want to select higher IDs 
      // (right most col, bottom most row)
      return a.id < id;
    }
    return string_vec[i] < a.string_vec[i];
  }

};


/*
 *******************************************************************************
 * function declaration *
 *******************************************************************************
*/

/* add tile to board */
void addTile(std::vector< std::vector<int> > *, std::vector<Shift> &, int);

/* 
 * check if two tiles can make a valid move (i.e add doubles, add 1 and 2
 * or simply move to empty space) and make the change.
 *
 * Returns 1 if move is made, 0 otherwise
 */
int tileMove(int *, int *);

/* Set the shift string of Shift s's columns/rows, so that we can sort them 
 * Not actually strings though, we build a vector of ints to sort with the
 * < operator specified in Shift struct
 */
void setShiftString(Board &, Shift &);

/* Get legal moves on board given tileNum to refer to tile in sequence */
std::vector<Direction> getPossibleMoves(const Board &, int);
PQ getPossibleMovesSorted(const Board &, int);

/* Make move specified by Direction on Board with input tile. Returns shifts */
std::vector<Shift> makeMove(Board *, Direction, int);

void print_board(const std::vector< std::vector<int> > &);
void read_in_file(std::vector< std::vector<int> > *, std::queue<int> *, char *);

/* Calculate score of tile */ 
int tileScore(int val);

/* Calculate score of board, based on sum of tileScore(tileVal) */
int score(const Board &);

bool inBounds(int, int);
int closestPair(Board &, int, int);

#endif