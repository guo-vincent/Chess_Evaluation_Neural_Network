#ifndef CHESS_ALG_H
#define CHESS_ALG_H

#include <limits>
#include "chess.hpp"

namespace chess {

int evaluate_pos(Board &board);
int negamax(Board &board, int depth, int alpha, int beta, Move &best_move);
int negamax(Board &board, int depth, int alpha, int beta);
Move find_best_move(Board board, int depth);

}

#endif // CHESS_ALG_H