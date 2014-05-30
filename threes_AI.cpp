#include <threes_AI.h>

/* Setup move parser maps, mapping Direction to string and vice versa */
void initMoveParsers(std::map<std::string, Direction> &move_parse,
                    std::map<Direction, std::string> &parse_move) {

  move_parse.insert(std::pair<std::string, Direction>("U", U));
  move_parse.insert(std::pair<std::string, Direction>("D", D));
  move_parse.insert(std::pair<std::string, Direction>("L", L));
  move_parse.insert(std::pair<std::string, Direction>("R", R));

  parse_move.insert(std::pair<Direction, std::string>(U, "U"));
  parse_move.insert(std::pair<Direction, std::string>(D, "D"));
  parse_move.insert(std::pair<Direction, std::string>(L, "L"));
  parse_move.insert(std::pair<Direction, std::string>(R, "R"));
}

std::string dToStr(Direction d) {
  switch (d) {
    case U:
      return "U";
    case D:
      return "D";
    case L:
      return "L";
    case R:
      return "R";
    default:
      std::cout << "Error in dToStr\n";
      exit(EXIT_FAILURE);
  }
}

Direction strToD(std::string str) {
  std::map<std::string, Direction> move_parse;
  move_parse.insert(std::pair<std::string, Direction>("U", U));
  move_parse.insert(std::pair<std::string, Direction>("D", D));
  move_parse.insert(std::pair<std::string, Direction>("L", L));
  move_parse.insert(std::pair<std::string, Direction>("R", R));

  return move_parse.find(str)->second;
}
/* Greedy best first search hill climb. Selects next move based on highest
 * board score
//  */
// void hillClimb(Board &board) {
//   PQ moveQueue;
//   // std::vector<Direction> parents = std::vector<Direction>(inputSequence.size());
//   std::vector<Direction> possMoves = getPossibleMoves(board, tile_num);

//   for (Direction d : possMoves) {
//     moveQueue.push()
//   }

//   while (!moveQueue.empty()) {

//   }
// }

/* calculates number of tiles that merged in move from board1 -> board2, returns
 * 0 if no tiles were consumed in the move (i.e. an input tile was added)
 */
int numberCollapsed(Board &b1, Board &b2) {

  int numNonZeroB1 = 0;
  int numNonZeroB2 = 0;

  for (int row = 0; row < BOARD_SIZE; row++) {
    for (int col = 0; col < BOARD_SIZE; col++) {
      numNonZeroB1 += b1[row][col] > 0 ? 1 : 0;
      numNonZeroB2 += b2[row][col] > 0 ? 1 : 0;
    }
  }

  return std::max(numNonZeroB1 - numNonZeroB2, 0);
}

int nonZeroTiles(Board &b) {
  int numNonZero = 0;

  for (int row = 0; row < BOARD_SIZE; row++) {
    for (int col = 0; col < BOARD_SIZE; col++) {
      numNonZero += b[row][col] > 0 ? 1 : 0;
    }
  }
  return numNonZero;
}

std::string boardToString(Board &b) {
  std::string str = "";

  for (int row = 0; row < BOARD_SIZE; row++) {
    for (int col = 0; col < BOARD_SIZE; col++) {
      str += std::to_string(b[row][col]);
    }
  }
  return str;
}

int THEORETICAL_HIGHSCORE;
int a_star(Board board, std::vector<std::string> &move_sequence, int depth, int *moves) {
  // goal state is tile_num = inputSequence.size() - 1
  // f(n) = 1 (cost of adding tile) + number of tiles collapsed?
  int maxDepth = 0;
  std::map<std::string, Direction> move_parse;
  std::map<Direction, std::string> parse_move;
  initMoveParsers(move_parse, parse_move);

  std::map<std::string, Node> parentMap;

  Node root;
  root.parent = NULL;
  root.b = board;
  root.depth = tile_num;
  root.f = 0;
  root.id = 0;
  root.score = score(root.b);
  root.str = boardToString(root.b);
  // getPossibleMoves(root.b, root.depth);

  parentMap.insert(std::pair<std::string, Node>(root.str, root));
  std::vector<Node> parents = std::vector<Node>(depth);
  NodeQ nq;
  nq.push(root);
  Node *maxNode;
  int maxScore = -1;
  while (!nq.empty()) {
    Node top = nq.top(); nq.pop();
    // std::cout << top.score << "\n"; 
    // std::cout << top.f << "\n";
    // if (top.depth == inputSequence.size() - 1) { // depth limit to # input tiles
    if (top.depth - root.depth == depth) { // depth limit to # input tiles
      if (top.score > maxScore) {
        // *moves = top.depth;
        maxScore = top.score;
        // move_sequence[top.depth] = parse_move.find(top.moveMade)->second;
        // maxDepth = top.depth;
        maxNode = &top;
        // std::cout << "Yo\n";
      }
      continue;
    }

    maxDepth = std::max(maxDepth, top.depth - tile_num);
    // store move sequence unless we have root node, which will 
    // have no previous move
    // if (top.id != root.id) { 
      // move_sequence[top.depth] = parse_move.find(top.moveMade)->second;
    // }

    // get frontier of current node
    std::vector<Direction> possMoves = getPossibleMoves(top.b, top.depth);

    if (possMoves.size() == 0) {
      // if (parentMap.size() > 0)
        // parentMap.erase(parentMap.find(top.str));
      continue;
    }
    // std::cout << top.depth << "\n";
    int id = 0;
    for (Direction d: possMoves) { // push moves as nodes to queue
      Node n;
      n.b = top.b;
      n.id = top.id + 1 + id++;
      n.depth = top.depth + 1;
      makeMove(&n.b, d, n.depth); // make move
      n.parent = &top;
      n.moveMade = d;
      n.score = score(n.b);
      n.g = top.g + 1; 
      // n.h = std::pow(numberCollapsed(top.b, n.b), 2); // this is where the heuristic matters
      // n.h = std::pow(nonZeroTiles(n.b), 2);
      // n.h = 0; // uniform cost search
      // n.h = n.score;
      // n.h  = nonZeroTiles(n.b);
      // n.h = score(n.b);
      // std::cout << n.h << "\n";
      // n.f = std::max(n.h + n.g, top.f);
      // n.f = std::min(n.h + n.g, top.f);
      n.f = score(n.b); // greedy best first search
      n.str = boardToString(n.b);
      // std::cout << n.depth - root.depth -1 << "\n";
      parents[n.depth - root.depth - 1] = top;

      // if (parentMap.count(n.str) == 0) 
      //   parentMap.insert(std::pair<std::string, Node>(n.str, top));
      // else {
      //   if (parentMap.find(n.str)->second.f > n.f) {
      //     parentMap.erase(parentMap.find(n.str));
      //     parentMap.insert(std::pair<std::string, Node>(n.str, top));
      //   }
      // }
      nq.push(n);
    }
    // std::cout << parentMap.size() << "\n";
  }
  *moves = maxDepth;
    // std::cout << "max: " << maxDepth << "\n";
  // make the actual moves on the initial board
  // for (int i = 1; i < *moves; i++) {
  //   std::cout << move_sequence[i];
  //   makeMove(&board, move_parse.find(move_sequence[i])->second, i);
  //   // printBoard(board);
  // }
  // Node p = *maxNode;
  int mm = maxDepth;
  // std::cout << "mm " << mm << "\n";
  while (mm > 0) {
    Node p = parents[mm--];
    move_sequence.push_back(parse_move.find(p.moveMade)->second);
  }
    
  // while (p.str != root.str) {
  //   // std::cout << parse_move.find(p.moveMade)->second;
  //   p = parentMap.find(p.str)->second;
  // }
  // parentMap.clear();
  std::reverse(move_sequence.begin(), move_sequence.end());
  std::cout <<"move_sequence.size(): " << move_sequence.size() << "\n";
  // std::vector<Direction> movess;
  // while (p->parent != NULL) {
  //   // printBoard(p.b);
  //   std::cout << parse_move.find(p->moveMade)->second;
  //   // makeMove(&board, p.moveMade)
  //   p = p->parent;
  // }

  std::cout << "\n";

  return !(tile_num < inputSequence.size() - 1);
}

int i_aStar(Board &board, std::vector<std::string> &move_sequence) {
  THEORETICAL_HIGHSCORE = inputSequence.size()*inputSequence.size();
  // THEORETICAL_HIGHSCORE = 1;

  std::cout << "Goal: " <<  THEORETICAL_HIGHSCORE << "\n";
  std::map<std::string, Direction> move_parse;
  std::map<Direction, std::string> parse_move;
  initMoveParsers(move_parse, parse_move);
  int numMoves = 0;
  int DEPTH_LIMIT = 8;

  // a_star(board, move_sequence, DEPTH_LIMIT, &numMoves);
  // int tt = 0;
  // for (std::string s: move_sequence) {
  //   makeMove(&board, move_parse.find(s)->second, tt++);
  // }
  // return 0;
  // std::vector<std::string> allMoves;
  while (numMoves < inputSequence.size()) {
    std::vector<std::string> ms;
    // std::cout << score(board) << "\n";
    int moves = 0;
    a_star(board, ms, DEPTH_LIMIT, &moves);
    numMoves += ms.size();
    if (ms.size() == 0) break;
    for (int i = 0; i < ms.size(); i++) {
      move_sequence.push_back(ms[i]);
      makeMove(&board, move_parse.find(ms[i])->second, tile_num++);
      printBoard(board);
    }
    // tile_num += moves;
  }
  std::cout << "inputSequence.size(): " << inputSequence.size() << "\n";
  std::cout << "moves: " << numMoves << "\n";
  return !(numMoves < inputSequence.size());
}

/*
 *
 * Design Possibilities:
 *    Select moves based on minimizing number of tiles on the board (consequently
 *     maximizing tile combinations)
 *
 *    Select next move based on maximizing board score (greedy best first search)
 * 
 */

// 
//  heuristic function h(b)
//  -> greedy local search, choose greatest h(b) until all h(b+1) < h(b), then
//      go back to h(b - 1) and choose next best h(b)
// h(b) = #shifts? board eval?
// 
// Node {
//  priorty_queue poss_moves
//  Node *parent
// }
int dfs(Board &board, std::vector<std::string> move_sequence, int depthLimit) {
    // Use these maps to convert strings to Direction enums, and vice versa.


  move_sequence = std::vector<std::string>(inputSequence.size());

  std::vector<Node> parents = std::vector<Node>(inputSequence.size());
  Node root; 
  root.b = board;
  // root.poss_moves = PQ();
  root.parent = NULL;
  root.depth = 0;

  // parents[0] = root;
  int max_score = -1;
  std::stack<Node> s;
  s.push(root);
  while (!s.empty()) {
    Node top = s.top(); s.pop();
    parents[top.depth] = top;
    if (top.depth == inputSequence.size() - 1) {
      // std::cout << "Score: " << score(top.b) << "\n";
      max_score = std::max(score(top.b), max_score);
      continue;
    }
    if (top.depth == depthLimit) continue;
    PQ poss_moves = getPossibleMovesSorted(top.b,  top.depth + 1);

    if (poss_moves.size() == 0) continue;

    for (int i = 0; i < poss_moves.size(); i++) {
      Direction d = poss_moves.top().second; poss_moves.pop();
      Node n;
      n.b = top.b;
      n.depth = top.depth + 1;
      makeMove(&n.b, d, top.depth + 1);
      s.push(n);
    }

  }
  return 0;
}

// int depthLimitedDFSSearch(Board board, int depth) {
//   std::stack<Node> path;
//   std::stack<Node> frontier;

//   Node root;
//   root.b = board;
//   root.
// }

// int greedy_search(Board &board, int depth, int tile) {
//   if (depth == 0) return score(board);
//   std::vector<Direction> poss_moves = getPossibleMoves(board, tile);
//   if (poss_moves.size() == 0) return score(board);
//   int best_val = -1;
//   for (Direction m : poss_moves) {
//     Board b = board;
//     makeMove(&b, m, tile); // possibly add shifts # to eval total
//     best_val = std::max(best_val, greedy_search(b, depth - 1, (tile + 1) % (inputSequence.size() - 1)));
//   }
//   return best_val;
// }
// 
// 

bool inBounds(int row, int col) {
  return row < BOARD_SIZE && row >= 0 && col < BOARD_SIZE && col >= 0;
}

int closestPair(Board &b, int row_0, int col_0) {
  std::queue<std::pair<int, int>> s;

  std::vector< std::pair<int,int> > adj;
  adj.push_back(std::pair<int, int>(-1, -1));
  adj.push_back(std::pair<int, int>(-1, 0)); 
  adj.push_back(std::pair<int, int>(-1, 1)); 
  adj.push_back(std::pair<int, int>(0, -1));
  adj.push_back(std::pair<int, int>(0, 1));
  adj.push_back(std::pair<int, int>(1, -1));
  adj.push_back(std::pair<int, int>(1, 0));
  adj.push_back(std::pair<int, int>(1, 1));

  std::vector< std::vector<int> > visited(BOARD_SIZE, std::vector<int>(BOARD_SIZE, 0));

  int row = row_0;
  int col = col_0;
  // std::cout << "closest pair for: " << b[row_0][col_0] << "\n";
  s.push(std::pair<int,int>(row_0, col_0));
  bool isstart = true;
  while (!s.empty()) {
    std::pair<int,int> top = s.front(); s.pop();
    if (!isstart) {
      if ((b[top.first][top.second] == b[row_0][col_0] &&
          b[top.first][top.second] != 1 && b[top.first][top.second] != 2) ||
          (b[top.first][top.second] == 1 && b[row_0][col_0] == 2) ||
          (b[top.first][top.second] == 2 && b[row_0][col_0] == 1)) {
        row = top.first;
        col = top.second;
        break;
      } 
    }
    visited[top.first][top.second] = 1;
    for (std::pair<int, int> p : adj) {
      std::pair<int, int> n(top.first + p.first, top.second + p.second);
      if (inBounds(n.first, n.second) && visited[n.first][n.second] == 0) {
        s.push(n);
      }
    }
    if (isstart) isstart = false;
  }
  if (s.size() == 0) return BOARD_SIZE + 1;
  // return manhattan distance row_0, col_0 -> row, col
  // std::cout << "row, col: " << row << " " << col << "\n";
  return abs(row_0 - row) + abs(col_0 - col);
}


const int sobel_y[3][3] = {
  {1, 2, 1},
  {0, 0, 0},
  {-1, -2, -1}
};

const int sobel_x[3][3] = {
  {-1, 0, 1},
  {-2, 0, 2},
  {-1, 0, 1}
};

/*
  Calculate variation within board by calculating gradient
 */
double placementScore(Board &b) {

  std::vector< std::pair<int,int> > adj;
  adj.push_back(std::pair<int, int>(-1, -1));
  adj.push_back(std::pair<int, int>(-1, 0)); 
  adj.push_back(std::pair<int, int>(-1, 1)); 
  adj.push_back(std::pair<int, int>(0, -1));
  adj.push_back(std::pair<int, int>(0, 1));
  adj.push_back(std::pair<int, int>(1, -1));
  adj.push_back(std::pair<int, int>(1, 0));
  adj.push_back(std::pair<int, int>(1, 1));

  int gradient[BOARD_SIZE][BOARD_SIZE];
  double gSum = 0.0;


  // convolve board matrix with sobel operator kernels
  for (int row = 0; row < BOARD_SIZE; row++) {
    for (int col = 0; col < BOARD_SIZE; col++) {
      int gy = 0;
      int gx = 0;
      for (int krow = 0; krow < 3; krow ++) {
        for (int kcol = 0; kcol < 3; kcol++) {

          if (inBounds(row + krow - 1, col + kcol - 1)) {
            gx += sobel_x[krow][kcol]*b[row + krow - 1][col + kcol - 1];
            gy += sobel_y[krow][kcol]*b[row + krow - 1][col + kcol - 1];
          }
        }
      }
      gradient[row][col] = std::sqrt(gx*gx + gy*gy);
      gSum += std::log((double)gradient[row][col]);
      // gSum += (double)gradient[row][col];
    }
  }

  // for (int row = 0; row < BOARD_SIZE; row++) {
    // for (int col = 0; col < BOARD_SIZE; col++) {
      // std::cout << gradient[row][col] << " ";
    // }
    // std::cout << "\n";
  // }
  // std::cout << "Gsum: " << gSum << "\n";
  return std::log(gSum);
  // return gSum;
}

float eval(Node &n, std::pair<int, int> loc) {
  // int s = n.score;
  // int nonZero = nonZeroTiles(n.b);
  // std::map<double, double> valWeights;
  // valWeights.insert(std::pair<double, double>(1.0, 10.0));
  // valWeights.insert(std::pair<double, double>(2.0, 10.0));
  // valWeights.insert(std::pair<double, double>(3.0, 5.0));
  // valWeights.insert(std::pair<double, double>(4.0, 2.5));
  // valWeights.insert(std::pair<double, double>(6.0, 1.5));
  // valWeights.insert(std::pair<double, double>(8.0, 1.0));
  // valWeights.insert(std::pair<double, double>(12.0, 1.0));

  // double distanceSum = 0;

  // // distance between every pair
  // for (int row = 0; row < BOARD_SIZE; row++) {
  //   for (int col = 0; col < BOARD_SIZE; col++) {
  //     if (valWeights.count(n.b[row][col]) > 0) 
  //       distanceSum += valWeights.find(n.b[row][col])->second*closestPair(n.b, row, col);
  //     else 
  //       distanceSum += closestPair(n.b, row, col);
  //   }
  // }
  

  //calc distance for added tile only
  // distanceSum += closestPair(n.b, loc.first, loc.second);
  // if (valWeights.count(n.b[loc.first][loc.second]) > 0) {
  //   distanceSum *= valWeights.find(n.b[loc.first][loc.second])->second;
  // }
  // std::cout << distanceSum - 2.0*std::log(n.score) << "\n";
  float scoreWeight = 2.0;
  // float nonZeroWeight = 2.0;
  // float distWeight = 10.0;
  float placementWeight = 100.0;
  // float 
  double eval = placementWeight*placementScore(n.b)
                // + (double)distanceSum*distWeight
                // - ((double)BOARD_SIZE*BOARD_SIZE - (double)nonZeroTiles(n.b)) 
                - scoreWeight*std::log((double)n.score);
                // - nonZeroWeight*std::log(nonZeroTiles(n.b));
  // std::cout << eval << "\n";
  return std::max(eval ,0.0);
  return eval;
}

std::vector<Direction> informed(Board board, int tile) {
  Node root;
  root.b = board;
  root.score = score(board);
  root.depth = 0;
  root.moveMade = U;

  std::vector<Direction> moves = std::vector<Direction>(inputSequence.size());
  
  // int depthCounter = root.depth;
  
  moves[0] = U;

  // queue nodes, so that we pop the best one according to heuristic first
  // order is defined in comparator in threes_Mechanics.h
  NodeQ nq;
  root.f = root.score;
  // std::vector<Node *> parentNodes(inputSequence.size());
  nq.push(root);

  while (!nq.empty()) {
    Node top = nq.top(); nq.pop();
    moves[top.depth + tile] =  top.moveMade;
    // parentNodes[top.depth + tile] = &top;
    // if (top.depth + tile == inputSequence.size() -1) { // goal state
    if (top.depth + tile == 5) {
      std::cout << top.score << ", Tile input done: \n";
      printBoard(top.b, false);
      // continue;
      break;
    }

    // if (top.depth == depthLimit) {

    // }

    std::vector<Direction> possMoves = getPossibleMoves(top.b, top.depth);

    if (possMoves.size() == 0) {
      continue;
    }

    for (Direction d: possMoves) {
      Node n;
      n.b = top.b;
      n.moveMade = d;
      n.depth = top.depth + 1;
      std::vector<Shift> shifts = makeMove(&n.b, d, tile + n.depth);
      Board bb = n.b;
      std::pair<int, int> loc = addTile(&bb, shifts, inputSequence[tile + n.depth]);
      n.score = score(n.b);
      n.f = eval(n, loc);
      // std::cout << n.f << "\n";
      // n.f = n.score;
      // n.f = 
      nq.push(n);
    }

    // tile++;
    if (tile > inputSequence.size()) {
      std::cout << "Tile out of bounds\n";
      exit(EXIT_FAILURE);
    }
    // std::cout << dToStr(top.moveMade);
  }
  std::cout << "\n ";

  for (int i = 1; i < moves.size(); i++) {
    std::cout << dToStr(moves[i]);
  }

  std::cout << "\n";
  return moves;
}


Direction greedy_search2(Board board, int tile) {
  int depthLim = 8;

  int maxScore = score(board);
  // int maxScore = eval(board)
  std::vector<Direction> originalFrontier = getPossibleMoves(board, tile);
  Direction maxD = originalFrontier[0];
  int minF = 1e9;
  if (originalFrontier.size() == 1) return maxD; // no point doing dfs
  // int maxScore = 1e6;
  for (Direction od : originalFrontier) {
    Node root;
    root.b = board;
    makeMove(&root.b, od, tile);
    // int tileID = tile + 1;
    root.score = score(board);
    root.depth = tile;
    // std::stack<Node> ns;
    NodeQ ns;
    ns.push(root);
    while (!ns.empty()) {
      Node top = ns.top(); ns.pop();

      std::vector<Direction> possMoves = getPossibleMoves(top.b, top.depth + 1);

      // there is simpler logic for the following abomination of if statements,
      // but i can't think properly.
      // 
      // basically just want to not look at a node if it has no more moves, 
      // unless it is the final tile (then we don't care if there's no more moves
      // , we just want the highest score).
      if (possMoves.size() == 0 && top.depth < inputSequence.size() - 1) {
        continue;
      }
      if (top.depth >= inputSequence.size() - 1 || top.depth - tile == depthLim) {
        // if (maxScore < top.score) {
          // maxScore = top.score;
        if (top.depth >= inputSequence.size() - 1) {
          if (maxScore < top.score) {
            maxD = od;
            maxScore = top.score;
          }
        } else if (minF >= top.f) {
          maxD = od;
          minF = top.f;   
          // break;
          // continue;
        } 
        continue;
      }
      if (possMoves.size() == 0) continue;

      for (Direction d : possMoves) {
        Node n;
        n.b = top.b;

        // std::pair<int, int> loc = std::pair<int, int>(0,0);
        n.moveMade = d;
        n.depth = top.depth + 1;
        n.score = score(n.b);
        n.parent = &top;
        std::vector<Shift> shifts = makeMove(&n.b, d, n.depth);
        Board bb = n.b;
        if (n.depth > inputSequence.size()) {
          std::cout << "TileID greater than input length\n";
          exit(EXIT_FAILURE);
        }
        std::pair<int, int> loc = addTile(&bb, shifts, inputSequence[n.depth]);
        n.f = eval(n, loc);
        ns.push(n);
        if (n.f < top.f) break;
      }

      // tileID++;
    }
    // tile++;
  }
  return maxD;
}

Direction greedy_search(Board board, int tile){
  std::vector<Direction> poss_moves = getPossibleMoves(board, tile);
  int sss = 0;
  int tile1 = tile + 1;
  int tile2 = tile1 + 1;
  int tile3 = tile2 + 1;
  int tile4 = tile3 + 1;
  int tile5 = tile4 + 1;
  int tile6 = tile5 + 1;
  int tile7 = tile6 + 1;
  Direction ddd;
  for (Direction m : poss_moves) {
    std::vector< std::vector<int> > b_copy = board;
    makeMove(&b_copy, m, tile);
    std::vector<Direction> poss_moves1 = getPossibleMoves(b_copy, tile1);
    if(poss_moves1.size()==0){
        if(score(b_copy)>sss){
          sss = score(b_copy);
          ddd = m;
          printf("Direction : %i \n", ddd);
        }
        continue;
    }
    for(Direction n : poss_moves1) {
      std::vector< std::vector<int> > c_copy = b_copy;
      makeMove(&c_copy, n, tile1);
      std::vector<Direction> poss_moves2 = getPossibleMoves(c_copy, tile2);
      if(poss_moves2.size()==0){
        if(score(c_copy)>sss){
          sss = score(c_copy);
          ddd = m;
          printf("Direction : %i \n", ddd);
        }
        continue;
      }
      for(Direction l : poss_moves2) {
        std::vector< std::vector<int> > d_copy = c_copy;
        makeMove(&d_copy, l, tile2);
        std::vector<Direction> poss_moves3 = getPossibleMoves(d_copy, tile3);
        if(poss_moves3.size()==0){
          if(score(d_copy)>sss){
            sss = score(d_copy);
            ddd = m;
            printf("Direction : %i \n", ddd);
          }
          continue;
        } 
        for(Direction k : poss_moves3) {
          std::vector< std::vector<int> > e_copy = d_copy;
          makeMove(&e_copy, k, tile3);
          std::vector<Direction> poss_moves4 = getPossibleMoves(e_copy, tile4);
          if(poss_moves4.size()==0){
            if(score(e_copy)>sss){
              sss = score(e_copy);
              ddd = m;
              printf("Direction : %i \n", ddd);
            }
            continue;
          } 
          for(Direction j : poss_moves4) {
            std::vector< std::vector<int> > f_copy = e_copy;
            makeMove(&f_copy, j, tile4);
            std::vector<Direction> poss_moves5 = getPossibleMoves(f_copy, tile5);
            if(poss_moves5.size()==0){
              if(score(f_copy)>sss){
                sss = score(f_copy);
                ddd = m;
                printf("Direction : %i \n", ddd);
              }
              continue;
            } 
            for(Direction o : poss_moves5) {
              std::vector< std::vector<int> > g_copy = f_copy;
              makeMove(&g_copy, o, tile5);
              std::vector<Direction> poss_moves6 = getPossibleMoves(g_copy, tile6);
              if(poss_moves6.size()==0){
                if(score(g_copy)>sss){
                  sss = score(g_copy);
                  ddd = m;
                  printf("Direction : %i \n", ddd);
                }
                continue;
              } 
              for(Direction p : poss_moves6) {
                std::vector< std::vector<int> > h_copy = g_copy;
                makeMove(&h_copy, p, tile6);
                std::vector<Direction> poss_moves7 = getPossibleMoves(h_copy, tile7);
                if(poss_moves7.size()==0){
                  if(score(h_copy)>sss){
                    sss = score(h_copy);
                    ddd = m;
                    printf("Direction : %i \n", ddd);
                  }
                  continue;
                } 
                for(Direction q : poss_moves7) {
                  std::vector< std::vector<int> > i_copy = h_copy;
                  makeMove(&i_copy, q, tile7);
                  if(score(i_copy)>sss){
                    sss = score(i_copy);
                    ddd = m;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  return ddd;
}
