// ProcessFile.cpp
// Combines all preprocessing steps into a single pass
// Compile with -std=c++20
// Before running this, make sure chessData under CSVFiles has been unzipped.

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <array>
#include <unordered_map>
#include <cmath>
#include "../../third_party/chess/chess.hpp"

using BoardRow = std::array<int, 8>;

// 1. Check for stalemate
bool is_not_stalemate(const std::string &FEN) {
    chess::Board board(FEN);
    return board.isGameOver().second != chess::GameResult::DRAW;
}

// 2. Qualify checkmates with validation
int scale_checkmate(const std::string &notation) {
    if (notation.length() < 3) return 0;
    
    const int extreme_value_positive = 14800;
    const int extreme_value_negative = -14800;
    const int max_mate_depth = 50;
    double scale_factor_positive = (extreme_value_positive - 1000) / static_cast<double>(max_mate_depth);
    double scale_factor_negative = (-1000 - extreme_value_negative) / static_cast<double>(max_mate_depth);

    try {
        if (notation[0] == '#' && notation[1] == '+') {
            int x = std::stoi(notation.substr(2));
            return 1000 + static_cast<int>(scale_factor_positive * x);
        } else if (notation[0] == '#' && notation[1] == '-') {
            int x = std::stoi(notation.substr(2));
            return -1000 - static_cast<int>(scale_factor_negative * x);
        }
    } catch (...) {
        // Invalid mate string
    }
    return 0;
}

// 3. Piece mapping
std::unordered_map<chess::Piece, int> piece_map = {
    {chess::Piece::WHITEPAWN, 1}, {chess::Piece::WHITEKNIGHT, 3}, 
    {chess::Piece::WHITEBISHOP, 4}, {chess::Piece::WHITEROOK, 5}, 
    {chess::Piece::WHITEQUEEN, 9}, {chess::Piece::WHITEKING, 100},
    {chess::Piece::BLACKPAWN, -1}, {chess::Piece::BLACKKNIGHT, -3}, 
    {chess::Piece::BLACKBISHOP, -4}, {chess::Piece::BLACKROOK, -5}, 
    {chess::Piece::BLACKQUEEN, -9}, {chess::Piece::BLACKKING, -100}
};

// 4. Convert board to matrix - FIXED: use original orientation (top to bottom)
std::array<BoardRow, 8> bitboard_to_rows(const chess::Board &board) {
    std::array<BoardRow, 8> rows = {};
    for (int sq_index = 0; sq_index < 64; ++sq_index) {
        chess::Square sq(sq_index);
        if (board.at(sq) != chess::Piece::NONE) {
            int row = sq.rank();  // Remove inversion: 0=top (black), 7=bottom (white)
            int col = sq.file();
            rows[row][col] = piece_map[board.at<chess::Piece>(sq)];
        }
    }
    return rows;
}

// 5. Write matrix to CSV
void write_rows_to_csv(const std::array<BoardRow, 8> &rows, 
                       const std::string &eval_str, 
                       const std::string &side, 
                       std::ofstream &ofs) {
    for (const auto &row : rows) {
        for (size_t col_idx = 0; col_idx < row.size(); ++col_idx) {
            ofs << row[col_idx];
            if (col_idx < row.size() - 1) ofs << ",";
        }
        ofs << std::endl;
    }
    
    ofs << ",,,,,,,," << eval_str << std::endl;
}

// --- Main Processing ---
int main() {
    // Open files
    std::ifstream infile("CSVFiles/chessData.csv");
    std::ofstream white_out("CSVFiles/White.csv");
    std::ofstream black_out("CSVFiles/Black.csv");

    if (!infile) {
        std::cerr << "Error opening input file!" << std::endl;
        return 1;
    }
    if (!white_out || !black_out) {
        std::cerr << "Error opening output files!" << std::endl;
        return 1;
    }

    // Write headers
    white_out << "col0,col1,col2,col3,col4,col5,col6,col7,Evaluation\n";
    black_out << "col0,col1,col2,col3,col4,col5,col6,col7,Evaluation\n";

    std::string line;
    std::getline(infile, line); // Skip header

    while (std::getline(infile, line)) {
        std::istringstream ss(line);
        std::string fen, eval_str;

        // Parse FEN and evaluation
        if (!std::getline(ss, fen, ',') || !std::getline(ss, eval_str, ',')) 
            continue;

        // 1. Remove stalemates first
        if (!is_not_stalemate(fen)) continue;

        // 2. Process evaluation
        try {
            if (eval_str.empty()) continue;

            if (eval_str[0] == '+' && eval_str.size() > 1) {
                int val = static_cast<int>(std::stof(eval_str.substr(1)));
                eval_str = std::to_string(val);
            } 
            else if (eval_str.size() >= 3 && eval_str[0] == '#' && 
                    (eval_str[1] == '+' || eval_str[1] == '-')) {
                int scaled = scale_checkmate(eval_str);
                eval_str = std::to_string(scaled);
            } 
            else {
                int val = static_cast<int>(std::stof(eval_str));
                eval_str = std::to_string(val);
            }
        } catch (...) {
            continue; // Skip invalid evaluations. Though we should never reach here
        }

        // 3. Determine side to move
        chess::Board board(fen);
        std::string side = (board.sideToMove() == chess::Color::WHITE) ? "White" : "Black";

        // 4. Convert and write board
        auto rows = bitboard_to_rows(board);
        if (side == "White") {
            write_rows_to_csv(rows, eval_str, side, white_out);
        } else {
            write_rows_to_csv(rows, eval_str, side, black_out);
        }
    }

    infile.close();
    white_out.close();
    black_out.close();
    return 0;
}