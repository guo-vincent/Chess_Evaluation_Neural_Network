// Step 1: Remove stalemates. 
// These can be hard to distinguish from actual positions where both sides are equal, which may
// confuse the neural net. Hence we remove them.
// Outputs the FilteredchessData.csv file
// use -std=c++20

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include "../../chess.hpp"

int eliminated_boards = 0;

// Stalemates are hard to distinguish from actual positions that are equal. So remove them.
bool eliminate_stalemates(const std::string &FEN) {
    chess::Board board(FEN);
    if (board.isGameOver().second == chess::GameResult::DRAW) {
        eliminated_boards++;
        return false;
    }
    return true;
}

void process_csv(const std::string& input_filename, const std::string& output_filename) {
    std::ifstream infile(input_filename);
    std::ofstream outfile(output_filename);
    std::string line;

    if (!infile.is_open()) {
        std::cerr << "Unable to open input file " << input_filename << std::endl;
        return;
    }

    // Read the header line
    if (std::getline(infile, line)) {
        // Write the header to the output file
        outfile << line << std::endl;

        // Process each row
        while (std::getline(infile, line)) {
            std::stringstream line_stream(line);
            std::string board_value;
            std::string row_values;

            // Extract the first column value
            if (std::getline(line_stream, board_value, ',')) {
                // Apply the function to the board column value
                if (eliminate_stalemates(board_value)) {
                    row_values = line; // Keep the entire row if function returns true
                    outfile << row_values << std::endl;
                    outfile.flush();
                }
            }
        }
    } else {
        std::cerr << "Unable to read header from input file" << std::endl;
    }

    infile.close();
    outfile.close();
}

int main() {
    std::string input_filename = "CSVFiles/chessData.csv";
    std::string output_filename = "CSVFiles/FilteredchessData.csv";
    
    process_csv(input_filename, output_filename);
    std::cout << eliminated_boards;
    
    return 0;
}