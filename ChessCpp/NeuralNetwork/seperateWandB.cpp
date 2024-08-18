// Step 3: Isolate White and Black datasets. 
// Tempi is a huge factor in certain positions, but the way the boards will be encoded into tensors
// prevents the NN from accounting for that. And adding another parameter would complicate things.
// So instead, we'll just make 2 NN models.
// Outputs WhiteFinal.csv and BlackFinal.csv
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>

void process_csv(const std::string& input_filename, const std::string& white_filename, const std::string& black_filename) {
    std::ifstream infile(input_filename);
    std::ofstream white_outfile(white_filename);
    std::ofstream black_outfile(black_filename);
    std::string line;

    if (!infile.is_open()) {
        std::cerr << "Unable to open input file " << input_filename << std::endl;
        return;
    }

    // Read the header line
    if (std::getline(infile, line)) {
        // Write the header to both output files
        white_outfile << line << std::endl;
        black_outfile << line << std::endl;

        // Process each row
        while (std::getline(infile, line)) {
            std::stringstream line_stream(line);
            std::string fen;
            std::string eval_value;
            std::string row_values;

            // Extract the FEN and evaluation value
            if (std::getline(line_stream, fen, ',') && std::getline(line_stream, eval_value, ',')) {
                // Find the side to move in the FEN
                size_t space_pos = fen.find(' ');
                char side_to_move = (space_pos != std::string::npos) ? fen[space_pos + 1] : '\0';

                // Combine the modified values
                row_values = fen + "," + eval_value;

                // Write to the appropriate file based on side to move
                if (side_to_move == 'w') {
                    white_outfile << row_values << std::endl;
                } else if (side_to_move == 'b') {
                    black_outfile << row_values << std::endl;
                }
            }
        }
    } else {
        std::cerr << "Unable to read header from input file" << std::endl;
    }

    infile.close();
    white_outfile.close();
    black_outfile.close();
}

int main() {
    std::string input_filename = "C:/Chess_Engine/chess-engine/ChessCpp/NeuralNetwork/CSVFiles/Final.csv";
    std::string white_filename = "C:/Chess_Engine/chess-engine/ChessCpp/NeuralNetwork/CSVFiles/WhiteFinal.csv";
    std::string black_filename = "C:/Chess_Engine/chess-engine/ChessCpp/NeuralNetwork/CSVFiles/BlackFinal.csv";
    
    process_csv(input_filename, white_filename, black_filename);
    
    return 0;
}