#include <threes_IO.h>

/**
 * Read in the initial board values and input sequence for game of Threes!
 * @param board          Board container to initialize
 * @param inputSequence Game input sequence of tiles
 * @param file_name      File name for the input file
 */
void readInFile(Board *board, std::vector<int> *inputSequence, char *file_name) {
  std::ifstream in_file;
  in_file.open(file_name);
  if (in_file.is_open()) {
    std::string line;
    // skip first 2 lines for now
    std::getline(in_file, line);
    std::getline(in_file, line);

    // read board init values
    for (int row = 0; row < BOARD_SIZE; row++) {
      std::getline(in_file, line);
      std::istringstream iss(line);
      for (int col = 0; col < BOARD_SIZE; col++) {
        int val;
        iss >> val;
        (*board)[row][col] = val;
      }
    }

    // skip blank line 7
    std::getline(in_file, line);
    // read input sequence
    while (!in_file.eof()) {
      std::getline(in_file, line);
      std::istringstream iss(line);
      std::string string_val;
      while (std::getline(iss, string_val, ' ')) {
        if (!std::isdigit(string_val[0])) continue; // check for white space
        inputSequence->push_back(std::stoi(string_val));
      }
    }

  } else {
    std::cout << "Could not open file: \"" << file_name << "\"\n";
    std::exit(EXIT_FAILURE);
  }
}

int MOVE_NUM = 0;
void printSVGFile(const Board &board) {
  std::string SVG_DIR = "./SVGs";
  mode_t mode = 0777;
  struct stat st;
  // create SVG directory if it doesn't exist
  if (stat(SVG_DIR.c_str(), &st) != 0) { 
    if (errno == ENOENT) {
      mkdir(SVG_DIR.c_str(), mode);
    } else {
      printf("Something wrong with SVG directory. Please create ./SVGs manually\n");
      exit(EXIT_FAILURE);
    }
  }
  std::string SVG_FILE_NAME = SVG_DIR + "/" + std::to_string(MOVE_NUM) + ".svg";
  std::ofstream outFile;
  outFile.open(SVG_FILE_NAME);
  if (!outFile) {
    std::cout << "Couldn't open: " << SVG_FILE_NAME << "\n";
    exit(EXIT_FAILURE);
  }
  outFile << "<svg xmlns=\"http://www.w3.org/2000/svg\">\n";
  float size = 100.0;
  for (int row = 0; row < board.size(); row++) {
    for (int col = 0; col < board.size(); col++) {
      int val = board[row][col];
      std::string rgb = "";
      switch(val) {
        case 0: rgb = "rgb(230,230,230);"; break;
        case 1: rgb = "rgb(0,255,255);"; break;
        case 2: rgb = "rgb(255,150,255);"; break;
        default: rgb = "rgb(100,255,100);"; break;
      }
      float xPos = float(col + 1) * size;
      float yPos = float(row + 1) * size;
      outFile << "<polygon points=\"";
      outFile << xPos << "," << yPos << " ";
      outFile << xPos + size << "," << yPos << " ";
      outFile << xPos + size << "," << yPos + size << " ";
      outFile << xPos << "," << yPos + size << " ";
      outFile << xPos << "," << yPos;
      outFile <<"\" style=\"fill:" << rgb << "stroke:rgb(0,0,0);stroke-width:2\"/>\n";
      if (val > 0) {
        outFile << "<text x=\"" << xPos + 40.0 << "\" y=\"" << yPos + 65.0 << "\"";
        outFile << " font-family=\"Verdana\" font-size=\"40\" fill=\"black\">\n";
        outFile << val << "</text>\n";
      }
    }
  }
  outFile << "</svg>\n";
  outFile.close();
  MOVE_NUM++;
}

/* Print board to stdout */
void printBoard(const Board &board, bool printSVG) {
  if (printSVG)
    printSVGFile(board);
  for (std::vector<int> l : board) {
    for (int c : l)
      std::cout << std::setfill(' ') << std::setw(4) << c << " ";
    std::cout << "\n";
  }
  std::cout << "\n";
}
