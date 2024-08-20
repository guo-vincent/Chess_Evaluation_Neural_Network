// Step 2: Process Checkmates. 
// Checkmates in the original dataset are denoted as #number, where smaller numbers
// mean closer to checkmate. Neural nets won't understand that, so instead we replace it with 
// a function that gives ridiculosly high evaluations in those positions
// Outputs the Final.csv file (filename should be changed)
// Use -std=c++20 to compile (other compiler versions have not been tested)

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>

int removed = 0;

int scale_checkmate(const std::string &notation) {
    int mate_value = 0;
    const int extreme_value_positive = 14800;
    const int extreme_value_negative = -14800;
    const int max_mate_depth = 50; // Maximum depth

    // Scale factor to distribute mate values proportionally outside the range of non-checkmates
    double scale_factor_positive = (extreme_value_positive - 1000) / (double)max_mate_depth;
    double scale_factor_negative = (-1000 - extreme_value_negative) / (double)max_mate_depth;

    // Check for positive mate (e.g., #+3)
    if (notation[0] == '#' && notation[1] == '+') {
        removed++;
        int x = std::stoi(notation.substr(2));
        mate_value = 1000 + scale_factor_positive * x;
    }
    // Check for negative mate (e.g., #-3)
    else if (notation[0] == '#' && notation[1] == '-') {
        removed++;
        int x = std::stoi(notation.substr(2)); 
        mate_value = -1000 - scale_factor_negative * x;
    }
    
    return mate_value;
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
            std::string eval_value;
            std::string row_values;

            // Extract the first and second column values
            if (std::getline(line_stream, board_value, ',') && std::getline(line_stream, eval_value, ',')) {
                // Apply the function to the evaluation column value
                if (eval_value[0] == '+') {
                    int val = std::stof(eval_value.substr(1));
                    /*
                    if (val >= 1000) {
                        val = 1000.0 + (val - 1000.0) * (2000.0 - 1000.0) / (14800.0 - 1000.0);
                    }
                    */
                    eval_value = std::to_string(val);
                } else if (eval_value[0] == '#' && (eval_value[1] == '+' || eval_value[1] == '-')) {
                    int scaled_value = scale_checkmate(eval_value);
                    eval_value = std::to_string(scaled_value);
                } else {
                    int val = std::stof(eval_value);
                    /*
                    if (val <= -1000) {
                        val = -2000.0 + (val + 14800.0) * (-1000.0 + 2000.0) / (14800.0 - 1000.0);
                    }
                    */
                    eval_value = std::to_string(val);
                }

                // Combine the modified values
                row_values = board_value + "," + eval_value;
                outfile << row_values << std::endl;
            }
        }
    } else {
        std::cerr << "Unable to read header from input file" << std::endl;
    }

    infile.close();
    outfile.close();
}

int main() {
    std::string input_filename = "CSVFiles/FilteredchessData.csv";
    std::string output_filename = "CSVFiles/Final.csv";
    
    process_csv(input_filename, output_filename);
    std::cout << removed;
    
    return 0;
}
