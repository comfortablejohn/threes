/**
 * Threes! player
 */
#include <threes.h>

/* 
 * Used for move iteration, such as user input or random move selection
 * Return 0 if game ends due to full board, 1 if game ends due to end of
 * tile input.
 */
int iterateMoves(Board &board, 
                  std::vector<std::string> &move_sequence, int playType) {

  // Use these maps to convert strings to Direction enums, and vice versa.
  std::map<std::string, Direction> move_parse;
  move_parse.insert(std::pair<std::string, Direction>("U", U));
  move_parse.insert(std::pair<std::string, Direction>("D", D));
  move_parse.insert(std::pair<std::string, Direction>("L", L));
  move_parse.insert(std::pair<std::string, Direction>("R", R));

  std::map<Direction, std::string> parse_move;
  parse_move.insert(std::pair<Direction, std::string>(U, "U"));
  parse_move.insert(std::pair<Direction, std::string>(D, "D"));
  parse_move.insert(std::pair<Direction, std::string>(L, "L"));
  parse_move.insert(std::pair<Direction, std::string>(R, "R"));

  std::vector<Direction> poss_moves = getPossibleMoves(board, tile_num);

  while (poss_moves.size() > 0) {
    printBoard(board);
    std::cout << "Possible moves: ";
    for (Direction d : poss_moves)
      std::cout << parse_move.find(d)->second << " ";
    std::cout << "\n";
    Direction m;

    std::string move;
    switch (playType) {
      case MAN: { // user defined input determines moves
        std::cin >> move;
        if (move == "Q") return -1;
        while (move_parse.count(move) == 0) {
          std::cout << "Move \"" << move 
                    << "\" invalid, please enter from {U, L, D, R}:\n";
          std::cin >> move;
          if (move == "Q") return -1;
          if (std::cin.eof()) {
            std::cout << "Read EOF for stdin.. Ending game.\n";
            return -1;
          }
        }
        m = move_parse.find(move)->second;
        break;
      }
      case RAND: { // random selection
        int rand_move = (int)(std::rand() % poss_moves.size());
        std::cout << "Chose poss[" << rand_move << "]: ";
        std::cout << parse_move.find(m)->second << "\n";
        m = poss_moves[rand_move];
        break;
      }
      case AI:{
        m = greedy_search2(board,tile_num);
        break;
      }
      default: {
        printf("Invalid playType used. Please use those defined in threes.h\n");
        exit(EXIT_FAILURE);
      }
    }

    std::vector<Shift> shifts = makeMove(&board, m, tile_num);

    std::cout << "Move #" << move_sequence.size() + 1 << " = " 
              << parse_move.find(m)->second << "\n";
    move_sequence.push_back(parse_move.find(m)->second);
    tile_num++;
    if (tile_num > inputSequence.size() - 1) {
      break;
    }

    poss_moves = getPossibleMoves(board, tile_num);
  }
  return poss_moves.size() > 0;
}
int help() {
  std::cout << "Usage: threes -i /dir/to/input_file_name [-m|-r|-a]\n";
  std::cout << "\t-i: Input tile sequence file name (mandatory arg)\n";
  std::cout << "\t-m: Manual moves (input U, D, L, or R for each turn)\n";
  std::cout << "\t-r: Random move selection\n";
  std::cout << "\t-a: AI move selection\n";  
  return 0;
}

/**
 * Play Threes!
 * Usage: threes -i /dir/to/input_file_name [-m|-r|-a]
 *     -i: Input tile sequence file name (mandatory arg)
 *     -m: Manual moves (input U, D, L, or R for each turn)
 *     -r: Random move selection
 *     -a: AI move selection
 */
int main(int argc, char *argv[]) {

  Board board (BOARD_SIZE, std::vector<int>(BOARD_SIZE, EMPTY));
  std::vector<std::string> move_sequence;
  char *fileName;
  fileName = (char *) malloc(SIZE_MAX);
  int playType;

  if (argc != 4) { // parse command line arguments
    return help();
  } else {
    int c;
    opterr = 0;
    while ((c = getopt(argc, argv, "i:mrat")) != -1) {
      switch(c) {
        case 'i': {
          fileName = optarg;
          break;
        }
        case 'm': {
          playType = MAN;
          break;
        }
        case 'r': {
          playType = RAND;
          break;
        }
        case 'a': {
          playType = AI;
          break;
        } 
        case 't': {
          playType = TEST;
          break;
        }
        default: {
          return help();   
        }
      }
    }
  }

  readInFile(&board, &inputSequence, fileName);

  std::srand(std::time(NULL));
  int endGame = -1;
  std::vector<Direction> moves;

  switch (playType) { // play the game how user wants to
    case (MAN):
      endGame = iterateMoves(board, move_sequence, MAN);
      break;
    case (RAND):
      endGame = iterateMoves(board, move_sequence, RAND);
      break;
    case (AI):
      // not implemented yet, call to AI algorithm goes here,
      // endGame = AI(); or something
      // endGame = dfs(board, move_sequence, 4);
      // int maxDepth;
      // endGame = a_star(board, move_sequence, &maxDepth);
      // endGame = i_aStar(board, move_sequence);
      endGame = iterateMoves(board, move_sequence, AI);
      // moves = greedy_search2(board, 0);
      // moves = informed(board, 0);
      for (Direction d : moves) {
        makeMove(&board, d, tile_num++);
      }
      break;
    case (TEST):
      std::cout << closestPair(board, 0, 0) << "\n";
      // return 1;
    default:
      return help();
  }

  switch (endGame) {
    case 0:
      std::cout << "No legal moves left (board full).\n";
      break;
    case 1:
      std::cout << "All input tiles have been used.\n";
      break;
    default:
      std::cout << "Not sure how the game ended.. (something went wrong, sorry)\n";
      break;
  }
  
  std::cout << "Printing final board...\n";
  printBoard(board);
  std::cout << "Game over. Score is: " << score(board) << ". Moves were: \n";
  for (std::string m : move_sequence) 
    std::cout << m;
  std::cout << "\n";

  return 0;
}
