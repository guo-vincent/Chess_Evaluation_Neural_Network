#include <limits>
#include <unordered_map>
#include "chess.hpp"
#include "evaluator.h"

namespace chess {

constexpr int INF = std::numeric_limits<int>::max();
constexpr int NegINF = std::numeric_limits<int>::min();

struct TTEntry {
    int score;
    int depth;
    enum Flag { EXACT, LOWERBOUND, UPPERBOUND } flag;
    chess::Move bestMove;
};

static std::unordered_map<uint64_t, TTEntry> trans_table;

int evaluate_pos(Board &board) {
    if (board.isGameOver().second == GameResult::LOSE) {
        return NegINF + 10000;
    } else if (board.isGameOver().second == GameResult::WIN) {
        return INF - 10000;
    } else if (board.isGameOver().second == GameResult::DRAW) {
        return 0;
    }
    
    static Evaluator evaluator(
        "Training/NeuralNetwork/onnx_models/ensemble_white.onnx",
        "Training/NeuralNetwork/onnx_models/ensemble_black.onnx"
    );

    int score = static_cast<int>(evaluator.evaluate(board)*1000.0f);
    return (board.sideToMove() == chess::Color::WHITE) ? score : -score;
}

int negamax(Board &board, int depth, int alpha, int beta, Move &best_move) {
    int alpha_orig = alpha;
    uint64_t key = board.hash();
    auto it = trans_table.find(key);
    if (it != trans_table.end() && it->second.depth >= depth) {
        const TTEntry &entry = it->second;
        if (entry.flag == TTEntry::EXACT) {
            best_move = entry.bestMove;
            return entry.score;
        } else if (entry.flag == TTEntry::LOWERBOUND) {
            alpha = std::max(alpha, entry.score);
        } else if (entry.flag == TTEntry::UPPERBOUND) {
            beta = std::min(beta, entry.score);
        }
        if (alpha >= beta) {
            best_move = entry.bestMove;
            return entry.score;
        }
    }

    if (depth == 0 || board.isGameOver().first != GameResultReason::NONE) {
        return evaluate_pos(board);
    }

    int max_score = -INF;
    Movelist moves;
    movegen::legalmoves(moves, board);

    std::vector<Move> ordered_moves;
    ordered_moves.reserve(moves.size());
    for (const Move &m : moves) {
        ordered_moves.push_back(m);
    }
    std::sort(ordered_moves.begin(), ordered_moves.end(), [&](const Move &a, const Move &b) {
        bool a_capture = board.isCapture(a);
        bool b_capture = board.isCapture(b);
        if (a_capture != b_capture) return a_capture > b_capture;
        return false;
    });

    for (const Move &m : ordered_moves) {
        board.makeMove(m);
        Move reply;
        int score = -negamax(board, depth - 1, -beta, -alpha, reply);
        board.unmakeMove(m);

        if (score > max_score) {
            max_score = score;
            best_move = m;
        }
        alpha = std::max(alpha, score);
        if (alpha >= beta) {
            break;
        }
    }

    TTEntry::Flag flag = TTEntry::EXACT;
    if (max_score <= alpha_orig) flag = TTEntry::UPPERBOUND;
    else if (max_score >= beta) flag = TTEntry::LOWERBOUND;
    trans_table[key] = TTEntry{max_score, depth, flag, best_move};

    return max_score;
}

int negamax(Board &board, int depth, int alpha, int beta) {
    Move dummy;
    return negamax(board, depth, alpha, beta, dummy);
}

Move find_best_move(Board board, int depth) {
    trans_table.clear();
    Move best_move{};
    int score = negamax(board, depth, -INF, INF, best_move);
    return best_move;
}

}

