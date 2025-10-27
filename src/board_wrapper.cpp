#include "board_wrapper.h"
#include "chess_alg.h"

#undef WHITE
#undef BLACK

Texture2D LoadPNGAsTexture(const char *filename) {
    Texture2D tex = LoadTexture(filename);
    if (tex.id == 0) {
        printf("Failed to load texture: %s\n", filename);
    }
    return tex;
}

void BoardWrapper::init(Player white_p, Player black_p) {
    promotion_pending = false;
    game_over = false;
    selected_from = -1;

    const std::vector<std::pair<std::string,std::string>> types = {
        {"pawn","white"}, {"knight","white"}, {"bishop","white"},
        {"rook","white"}, {"queen","white"}, {"king","white"},
        {"pawn","black"}, {"knight","black"}, {"bishop","black"},
        {"rook","black"}, {"queen","black"}, {"king","black"},
    };
    for (auto &p : types) {
        std::string key = p.second + "-" + p.first;
        std::string path = "pieces-basic-png/" + key + ".png";
        pieceTextures[key] = LoadPNGAsTexture(path.c_str());
    }

    screenWidth = GetScreenWidth();
    screenHeight = GetScreenHeight();
    int boardSize = TILE_SIZE * 8;

    boardOffsetX = (screenWidth - boardSize) / 2;
    boardOffsetY = (screenHeight - boardSize) / 2;

    chess::movegen::legalmoves(moves, board);

    white_player = white_p;
    black_player = black_p;
}

BoardWrapper::~BoardWrapper() {
    stopBotThread();
    for (auto &kv : pieceTextures) {
        UnloadTexture(kv.second);
    }
}

void BoardWrapper::DrawBoard() {
    for (int sq = 0; sq < 64; ++sq) {
        int file = sq % 8;
        int rank = sq / 8;
        int x = boardOffsetX + file * TILE_SIZE;
        int y = boardOffsetY + (7 - rank) * TILE_SIZE;

        // draw checkerboard tile
        Color bg = ((file + rank) % 2) ? DARKGRAY : LIGHTGRAY;
        DrawRectangle(x, y, TILE_SIZE, TILE_SIZE, bg);

        if (sq == selected_from) {
            DrawRectangle(x, y, TILE_SIZE, TILE_SIZE, YELLOW);
        }

        // look up piece on square
        chess::Square square(sq);
        chess::Piece piece = board.at(square);
        if (piece != chess::Piece::underlying::NONE) {
            std::string key;
            switch (piece.internal()) {
                case chess::Piece::underlying::WHITEPAWN:   key = "white-pawn";   break;
                case chess::Piece::underlying::WHITEKNIGHT: key = "white-knight"; break;
                case chess::Piece::underlying::WHITEBISHOP: key = "white-bishop"; break;
                case chess::Piece::underlying::WHITEROOK:   key = "white-rook";   break;
                case chess::Piece::underlying::WHITEQUEEN:  key = "white-queen";  break;
                case chess::Piece::underlying::WHITEKING:   key = "white-king";   break;
                case chess::Piece::underlying::BLACKPAWN:   key = "black-pawn";   break;
                case chess::Piece::underlying::BLACKKNIGHT: key = "black-knight"; break;
                case chess::Piece::underlying::BLACKBISHOP: key = "black-bishop"; break;
                case chess::Piece::underlying::BLACKROOK:   key = "black-rook";   break;
                case chess::Piece::underlying::BLACKQUEEN:  key = "black-queen";  break;
                case chess::Piece::underlying::BLACKKING:   key = "black-king";   break;
                default: continue;
            }

            // find preloaded textures
            auto it = pieceTextures.find(key);
            if (it != pieceTextures.end()) {
                Texture2D pieceTex = it->second;
                // Calculate center
                int drawX = x + (TILE_SIZE - pieceTex.width) / 2;
                int drawY = y + (TILE_SIZE - pieceTex.height) / 2;
                DrawTexture(pieceTex, drawX, drawY, CLITERAL(Color){ 255, 255, 255, 255 });
            }
        }
        if (promotion_pending) {
            // draw promotion boxes
            for (size_t i = 0; i < promotion_options.size(); ++i) {
                Rectangle box = getPromotionBox(i);
                Color transparentWhite = { 255, 255, 255, 50 };
                DrawRectangleRec(box, transparentWhite);
                chess::PieceType promoPiece = promotion_options[i].promotionType();
                std::string prefix;
                switch (promotion_color) {
                    case 0: prefix = "white-"; break;
                    case 1: prefix = "black-"; break;
                }
                std::string key = prefix + pieceName(promoPiece);
                auto it = pieceTextures.find(key);
                if (it != pieceTextures.end()) {
                    Texture2D tex = it->second;
                    DrawTexture(tex,
                                int(box.x + (box.width - tex.width)/2),
                                int(box.y + (box.height - tex.height)/2),
                                transparentWhite);
                }
            }
        }
    }

    if (game_over) {
        int boxWidth = 300;
        int boxHeight = 150;
        int boxX = (screenWidth - boxWidth) / 2;
        int boxY = (screenHeight - boxHeight) / 2;

        Color bg = { 0, 0, 0, 180 };
        DrawRectangle(boxX, boxY, boxWidth, boxHeight, bg);
        DrawRectangleLines(boxX, boxY, boxWidth, boxHeight, RAYWHITE);

        int fontSize = 30;
        int textWidth = MeasureText(result_text.c_str(), fontSize);
        int textX = boxX + (boxWidth - textWidth) / 2;
        int textY = boxY + (boxHeight - fontSize) / 2;

        DrawText(result_text.c_str(), textX, textY, fontSize, CLITERAL(Color){ 255, 255, 255, 255 });
    }
}

void BoardWrapper::UpdateClicks() {
    if (game_over || botThinking) {
        return;
    }

    if (!game_over && (
        (board.sideToMove() == chess::Color::WHITE && white_player == BOT) || 
        (board.sideToMove() == chess::Color::BLACK && black_player == BOT))) {
        makeBotMoveAsync();
        return;
    }

    if (!IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
        return;
    }

    Vector2 m = GetMousePosition();
    int mx = int(m.x) - boardOffsetX;
    int my = int(m.y) - boardOffsetY;

    // Translate to file/rank and square index
    int file = mx / TILE_SIZE;
    int rank = 7 - (my / TILE_SIZE);
    chess::Square clicked(rank * 8 + file);

    // Promotion mode
    if (promotion_pending) {
        // If user clicks one of the promo boxes, make move
        for (size_t i = 0; i < promotion_options.size(); ++i) {
            Rectangle box = getPromotionBox(i);
            if (CheckCollisionPointRec(m, box)) {
                board.makeMove(promotion_options[i]);
                promotion_pending = false;
                refreshMoveList();
                selected_from = -1;
                return;
            }
        }
        return;
    }

    // normal move select
    chess::Piece clicked_piece = board.at(clicked);

    if (mx < 0 || my < 0 || mx >= TILE_SIZE * 8 || my >= TILE_SIZE * 8)
        return;

    // initial piece selection
    if (selected_from < 0) {
        if (clicked_piece != chess::Piece::underlying::NONE) {
            selected_from = clicked.index();
        }
        return;
    }

    chess::Square from(selected_from);
    chess::Piece sel = board.at(from);

    // clicked same square: deselect
    if (clicked.index() == selected_from) {
        selected_from = -1;
        return;
    }

    std::vector<chess::Move> normal_or_castle;
    std::vector<chess::Move> promos;

    for (auto &m : moves) {
        if (m.from() == from && m.to() == clicked) {
            if (m.typeOf() == chess::Move::PROMOTION)
                promos.push_back(m);
            else
                normal_or_castle.push_back(m);
        }
    }

    // promotions
    if (!promos.empty()) {
        if (promos.size() == 1) {
            // one choice
            board.makeMove(promos[0]);
            refreshMoveList();
            selected_from = -1;
        } else {
            // Multiple choices
            promotion_pending  = true;
            promotion_from     = from;
            promotion_to       = clicked;
            promotion_options  = std::move(promos);
            promotion_color    = board.at(from).color();
        }
        return;
    }

    // normal/castling move
    if (!normal_or_castle.empty()) {
        board.makeMove(normal_or_castle.front());
        refreshMoveList();
        checkGameOver();
    }

    // Select another piece if we select a diff piece of same color. Must be at end to prevent short circuiting of castling
    if (clicked_piece != chess::Piece::underlying::NONE &&
        clicked_piece.color() == sel.color())
    {
        selected_from = clicked.index();
        return;
    }

    // Reset selection
    selected_from = -1;
}

Rectangle BoardWrapper::getPromotionBox(size_t i) {
    int sq = promotion_to.index();
    int file = sq % 8;
    int rank = sq / 8;

    int S = TILE_SIZE;

    int baseX = boardOffsetX + file * S;
    int baseY = boardOffsetY + (7 - rank) * S;

    bool isWhite = (static_cast<int>(promotion_color) == 0);

    // Total height of the stacked promotion boxes
    int stackHeight = 4 * S;

    // Default top-left y of the stack
    int stackTopY = isWhite ? baseY - stackHeight : baseY + S;

    // Clamp to screen
    if (stackTopY < 0) {
        stackTopY = 0;
    } else if (stackTopY + stackHeight > screenHeight) {
        stackTopY = screenHeight - stackHeight;
    }

    // Final Y for box
    int boxY = stackTopY + int(i) * S;

    return {
        float(baseX),
        float(boxY),
        float(S),
        float(S)
    };
}

std::string BoardWrapper::pieceName(chess::PieceType pt) {
    switch (pt.internal()) {
        case chess::PieceType::PAWN:   return "pawn";
        case chess::PieceType::KNIGHT: return "knight";
        case chess::PieceType::BISHOP: return "bishop";
        case chess::PieceType::ROOK:   return "rook";
        case chess::PieceType::QUEEN:  return "queen";
        default:                       return "king";
    }
}

void BoardWrapper::checkGameOver() {
    refreshMoveList();
    if (moves.empty()) {
        game_over = true;
        if (board.inCheck()) {
            result_text = static_cast<int>(board.sideToMove()) == 0 ? "Black wins!" : "White wins!";
        } else {
            result_text = "Stalemate!";
        }
    }
}

void BoardWrapper::makeBotMoveAsync() {
    if (botThinking) return;
    stopBot = false;
    botThinking = true;
    if (botThread.joinable()) {
        botThread.join();
    }
    botThread = std::thread([this]() {
        chess::Move bestMove;
        {
            std::lock_guard<std::mutex> lock(boardMutex);
            if (stopBot) {
                botThinking = false;
                return;
            }
            bestMove = find_best_move(board, 3);
        }
        {
            std::lock_guard<std::mutex> lock(boardMutex);
            if (!stopBot) {
                board.makeMove(bestMove);
                refreshMoveList();
            }
        }
        checkGameOver();
        botThinking = false;
    });
}

void BoardWrapper::stopBotThread() {
    stopBot = true;
    if (botThread.joinable()) {
        botThread.join();
    }
    botThinking = false;
}

void BoardWrapper::reset() {
    stopBotThread();
    // Reset board and state variables
    {
        std::lock_guard<std::mutex> lock(boardMutex);
        board.setFen(chess::constants::STARTPOS);
        promotion_pending = false;
        game_over = false;
        selected_from = -1;
        chess::movegen::legalmoves(moves, board);
    }
}

#define WHITE CLITERAL(Color){ 255, 255, 255, 255 }
#define BLACK CLITERAL(Color){ 0, 0, 0, 255 }