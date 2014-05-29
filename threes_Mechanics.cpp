#include <threes_Mechanics.h>

std::vector<int> inputSequence;
int tile_num = 0;

/**
 * check if two tiles can make a valid move (i.e add doubles, add 1 and 2
 * or simply move to empty space) and make the change
 */
int tileMove(int *lv, int *v) {
  if (*lv == EMPTY) {
    *lv = *v;
    *v = EMPTY;
  } else if (*lv == *v && *v > 2) {
    *lv *= 2;
    *v = EMPTY;
  } else if (*lv + *v == 3) {
    *lv = 3;
    *v = EMPTY;
  } else {
    return 0;
  }
  return 1;
}

/* Set the shift string of Shift s's columns/rows, so that we can sort them 
 * Not actually strings though, we build a vector of ints to sort with the
 * < operator specified in Shift struct (threes_Mechanics.h)
*/
void setShiftString(Board &board, Shift &s) {
  if (s.m == U) {
    for (int row = BOARD_SIZE - 1; row >= 0; row--) {
      s.string_vec.push_back(board[row][s.id]);
    }
  } else if (s.m == D) {
    for (int row = 0; row < BOARD_SIZE; row++) {
      s.string_vec.push_back(board[row][s.id]);
    }
  } else if (s.m == L) {
    for (int col = BOARD_SIZE - 1; col >= 0; col--) {
      s.string_vec.push_back(board[s.id][col]);
    }
  } else {
    for (int col = 0; col < BOARD_SIZE; col++) {
      s.string_vec.push_back(board[s.id][col]);
    }
  }
}

/* Get legal moves on board given input tile */
std::vector<Direction> getPossibleMoves(const Board &board, int tile) {
  std::vector<Direction> poss_moves;

  std::vector<Direction> move_space;
  move_space.push_back(U);
  move_space.push_back(D);
  move_space.push_back(L);
  move_space.push_back(R);

  for (Direction d : move_space) {
    std::vector< std::vector<int> > b_copy = board;
    std::vector<Shift> shifts = makeMove(&b_copy, d, tile);
    if (shifts.size() > 0) poss_moves.push_back(d);
  }

  return poss_moves;
}

PQ getPossibleMovesSorted(const Board &board, int tile) {
  PQ poss_moves;

  std::vector<Direction> move_space;
  move_space.push_back(U);
  move_space.push_back(D);
  move_space.push_back(L);
  move_space.push_back(R);

  for (Direction d : move_space) {
    std::vector< std::vector<int> > b_copy = board;
    std::vector<Shift> shifts = makeMove(&b_copy, d, tile);
    if (shifts.size() > 0) {
      poss_moves.push(std::pair<int, Direction>(shifts.size(), d));
    }
  }

  return poss_moves;
}

/* Make move specified by Direction on Board with input tile. Returns shifts. */
std::vector<Shift> makeMove(Board *board, Direction move, int tile) {
  std::vector<Shift> shifts;
  switch(move) {
    case U:
      for (int col = 0; col < BOARD_SIZE; col++) {
        int num_shifts = 0;
        for (int row = 1; row < BOARD_SIZE; row++) {
          int *lv = &(*board)[row - 1][col];
          int *v = &(*board)[row][col];
          if (*v == EMPTY) continue;
          num_shifts += tileMove(lv, v);
        }
        if (num_shifts > 0) {
          Shift s(U, col);
          setShiftString(*board, s);
          shifts.push_back(s);
        }
      }
      break;
    case D:
      for (int col = 0; col < BOARD_SIZE; col++) {
        int num_shifts = 0;
        for (int row = BOARD_SIZE - 2; row >= 0; row--) {
          int *lv = &(*board)[row + 1][col];
          int *v = &(*board)[row][col];
          if (*v == EMPTY) continue;
          num_shifts += tileMove(lv, v);
        }
        if (num_shifts > 0) {
          Shift s(D, col);
          setShiftString(*board, s);
          shifts.push_back(s);
        }
      }
      break;
    case L:
      for (int row = 0; row < BOARD_SIZE; row++) {
        int num_shifts = 0;
        for (int col = 1; col < BOARD_SIZE; col++) {
          int *lv = &(*board)[row][col - 1];
          int *v = &(*board)[row][col];
          if (*v == EMPTY) continue;
          num_shifts += tileMove(lv, v);
        }
        if (num_shifts > 0) {
          Shift s(L, row);
          setShiftString(*board, s);
          shifts.push_back(s);
        }
      }
      break;
    case R:
      for (int row = 0; row < BOARD_SIZE; row++) {
        int num_shifts = 0;
        for (int col = BOARD_SIZE - 2; col >= 0; col--) {
          int *lv = &(*board)[row][col + 1];
          int *v = &(*board)[row][col];
          if (*v == EMPTY) continue;
          num_shifts += tileMove(lv, v);
        }
        if (num_shifts > 0) {
          Shift s(R, row);
          setShiftString(*board, s);
          shifts.push_back(s);
        }
      }
      break;
    default:
      break;
  }
  
  if (shifts.size() > 0)
    addTile(board, shifts, inputSequence[tile]);
  // std::cout << "tile: " << tile << " = " << inputSequence[tile] << "\n";
  return shifts;
}

/* add tile to board */
std::pair<int, int> addTile(Board *board, std::vector<Shift> &shifts, int tile) {
  std::sort(shifts.begin(), shifts.end());
  // Shift min_shift = getMinShift(shifts);

  Direction shift_d = shifts[0].m;
  int shift_id = shifts[0].id;
  // Direction shift_d = min_shift.m;
  // int shift_id = min_shift.id;
  std::pair<int, int> p;
  if (shift_d == U) {
    (*board)[BOARD_SIZE - 1][shift_id] = tile;
    p = std::pair<int, int>(BOARD_SIZE - 1, shift_id);
  } else if (shift_d == D) {
    (*board)[0][shift_id] = tile;
    p = std::pair<int, int>(0, shift_id);
  } else if (shift_d == L) {
    (*board)[shift_id][BOARD_SIZE - 1] = tile;
    p = std::pair<int, int>(shift_id, BOARD_SIZE - 1);
  } else {
    (*board)[shift_id][0] = tile;
    p = std::pair<int, int>(shift_id, 0);
  }
  // std::cout << shift_d << " " << tile << "\n";

  return p;
}

/* Calculate score of tile */ 
int tileScore(int val) {
  if (val == 0) return 0;
  if (val == 1 || val == 2) return 1;
  return std::pow(3, (int)std::log2((double) val / 3.0) + 1);
}

/* Calculate score of board, based on sum of tileScore(tileVal) */
int score(const Board &board) {
  int score = 0;
  for (int row = 0; row < BOARD_SIZE; row++) {
    for (int col = 0; col < BOARD_SIZE; col++) {
      score += tileScore(board[row][col]);
    }
  }
  return score;
}