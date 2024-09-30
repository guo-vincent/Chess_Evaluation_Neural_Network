// Step 4: Encode FENs. 
// FENs don't fit into CNNs so we have to transform them into tensors. Or at least something easy
// enough to convert into a tensor.
// Outputs White.csv and Black.csv
// use -std=c++20

// BUG: First matrix is a zero matrix, so one will need to go and edit it out. All other matrices are normal
#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <string>
#include "../../chess.hpp"

using BoardRow = std::array<int, 8>;

// Map for translating chess pieces to corresponding integer values
std::unordered_map<chess::Piece, int> piece_map = {
    {chess::Piece::WHITEPAWN, 1}, {chess::Piece::WHITEKNIGHT, 3}, {chess::Piece::WHITEBISHOP, 4}, {chess::Piece::WHITEROOK, 5}, {chess::Piece::WHITEQUEEN, 9}, {chess::Piece::WHITEKING, 100},
    {chess::Piece::BLACKPAWN, -1}, {chess::Piece::BLACKKNIGHT, -3}, {chess::Piece::BLACKBISHOP, -4}, {chess::Piece::BLACKROOK, -5}, {chess::Piece::BLACKQUEEN, -9}, {chess::Piece::BLACKKING, -100},
};

// Convert bitboard to a single row format
std::array<BoardRow, 8> bitboard_to_rows(const chess::Board &board) {
    chess::Bitboard bitboard = board.occ();
    std::array<BoardRow, 8> rows = {};
    for (int sq_index = 0; sq_index < 64; ++sq_index) {
        if (bitboard.check(sq_index)) {
            chess::Square sq(sq_index);
            int row = sq_index / 8;
            int col = sq_index % 8;
            rows[row][col] = piece_map[board.at<chess::Piece>(sq)];
        }
    }
    return rows;
}

// Modified function to include the side parameter and write to the CSV file
void write_rows_to_csv(const std::array<BoardRow, 8> &rows, const std::string &eval_str, const std::string &side, std::ofstream &ofs) {
    if (!ofs) {
        throw std::runtime_error("Failed to open file for writing");
    }

    // Write the matrix data to the file
    for (size_t row_idx = 0; row_idx < rows.size(); ++row_idx) {
        for (size_t col_idx = 0; col_idx < rows[row_idx].size(); ++col_idx) {
            ofs << rows[row_idx][col_idx];
            if (col_idx < rows[row_idx].size() - 1) {
                ofs << ",";
            }
        }
        ofs << std::endl;
    }

    // Write the evaluation value and side to the last row
    ofs << std::string(rows[0].size(), ',') << eval_str << "," << side << std::endl;
}

// Write information to the CSV file
void process_csv_and_write_to_csv(const std::string& input_filename, const std::string& output_filename, const std::string& side) {
    std::ifstream infile(input_filename);
    std::ofstream outfile(output_filename);
    std::string line;

    if (!infile.is_open()) {
        std::cerr << "Unable to open input file " << input_filename << std::endl;
        return;
    }

    if (!outfile.is_open()) {
        std::cerr << "Unable to open output file " << output_filename << std::endl;
        return;
    }

    // Write CSV header
    outfile << "col0,col1,col2,col3,col4,col5,col6,col7,Evaluation,Side" << std::endl;

    // Process rows
    while (std::getline(infile, line)) {
        std::stringstream line_stream(line);
        std::string fen;
        std::string eval_str;

        if (std::getline(line_stream, fen, ',') && std::getline(line_stream, eval_str, ',')) {
            chess::Board board(fen);

            // Convert board to rows
            std::array<BoardRow, 8> rows = bitboard_to_rows(board);
            write_rows_to_csv(rows, eval_str, side, outfile);
        }
    }

    infile.close();
    outfile.close();
}

int main() {
    // Uncomment to test single board
    chess::Board board(chess::constants::STARTPOS);
    std::array<BoardRow, 8> rows = bitboard_to_rows(board);
    // Header has to be manually specified when not using process_csv_and_write_to_csv.
    std::ofstream test("/Users/vincentguo/Chess_Engine/Chess_Evaluation_Neural_Network/ChessCpp/NeuralNetwork/CSVFiles/test.csv", std::ios::out);
    test << "col0,col1,col2,col3,col4,col5,col6,col7,Evaluation,Side" << std::endl;
    write_rows_to_csv(rows, "0", "White", test);

    // Process files
    // process_csv_and_write_to_csv("CSVFiles/WhiteFinal.csv", "CSVFiles/White2.csv", "White");
    // process_csv_and_write_to_csv("CSVFiles/BlackFinal.csv", "CSVFiles/Black2.csv", "Black");
    return 0;
}
