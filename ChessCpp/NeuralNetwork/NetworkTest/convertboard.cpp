#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <string>
#include "C:/Chess_Engine/chess-engine/ChessCpp/chess.hpp"

// Using floats to normalize values between 0 and 1
using BoardRow = std::array<int, 8>;

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

void write_rows_to_csv(const std::array<BoardRow, 8> &rows, const std::string &eval_str, const std::string &filename) {
    std::ofstream ofs(filename, std::ios::app); // Open in append mode
    if (!ofs.is_open()) {
        throw std::runtime_error("Failed to open file for writing");
    }

    // Write the matrix data to the file with row headers
    for (size_t row_idx = 0; row_idx < rows.size(); ++row_idx) {
        for (size_t col_idx = 0; col_idx < rows[row_idx].size(); ++col_idx) {
            ofs << rows[row_idx][col_idx];
            if (col_idx < rows[row_idx].size() - 1) {
                ofs << ",";
            }
        }
        ofs << std::endl;
    }

    // Write the evaluation value to the last row
    ofs << std::string(rows[0].size(), ',') << eval_str << std::endl;
}

void write_header(const std::string &filename) {
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        throw std::runtime_error("Failed to open file for writing");
    }
    ofs << "col0,col1,col2,col3,col4,col5,col6,col7,Evaluation" << std::endl;
}

int main() {
    try {
        const std::string filename = "C:/Chess_Engine/chess-engine/ChessCpp/NeuralNetwork/CSVFiles/test.csv";

        // Write the header (overwrite the file if it exists)
        write_header(filename);

        // Uncomment to test single board
        chess::Board board("rnbqkbnr/pppp1p1p/6p1/4p2Q/4P3/8/PPPP1PPP/RNB1KBNR w KQkq - 0 1");
        std::array<BoardRow, 8> rows = bitboard_to_rows(board);
        write_rows_to_csv(rows, "0", filename);
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
