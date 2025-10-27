#ifndef BOARD_WRAPPER_H
#define BOARD_WRAPPER_H

#define TILE_SIZE 120
#define BOARD_OFFSET_X 100
#define BOARD_OFFSET_Y 100

#include <unordered_map>
#include <thread>
#include <atomic>
#include <mutex>

#include "chess.hpp"
#include "rlImGui.h"
#include "imgui.h"
#include "raylib.h"

enum Player {
    HUMAN,
    BOT
};

class BoardWrapper {
    public:
        BoardWrapper(chess::Board &b, Player white_p = HUMAN, Player black_p = HUMAN) : board(b), white_player(white_p), black_player(black_p) {}
        ~BoardWrapper();
        void init(Player white_p = HUMAN, Player black_p = HUMAN);
        void DrawBoard();
        void UpdateClicks();
        Rectangle getPromotionBox(size_t i);
        std::string pieceName(chess::PieceType pt);
        void checkGameOver();

        void makeBotMoveAsync();
        void stopBotThread();

        void reset();

        void refreshMoveList() {
            moves.clear();
            chess::movegen::legalmoves(moves, board);
        }

        int get_selected_from() { return selected_from; }
        
    private:
        chess::Board &board;
        int selected_from;
        std::unordered_map<std::string, Texture2D> pieceTextures;
        chess::Movelist moves;
        bool promotion_pending;
        chess::Square promotion_from;
        chess::Square promotion_to;
        std::vector<chess::Move> promotion_options;
        chess::Color promotion_color;

        bool game_over;
        std::string result_text;

        Player white_player;
        Player black_player;

        int boardOffsetX = 0;
        int boardOffsetY = 0;

        int screenWidth;
        int screenHeight;

    std::thread botThread;
    std::atomic<bool> botThinking{false};
    std::atomic<bool> stopBot{false};
    std::mutex boardMutex;
};

#endif // BOARD_WRAPPER_H