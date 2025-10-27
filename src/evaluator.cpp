#include <vector>
#include "chess.hpp"
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <cassert>
#include <locale>
#include <codecvt>
#include <string>
#include <algorithm>

#include "evaluator.h"

namespace chess {

static std::vector<float> init_ranks() {
    std::vector<float> r(64);
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            r[i*8 + j] = (i / 3.5f) - 1.0f;
        }
    }
    return r;
}

static std::vector<float> init_files() {
    std::vector<float> f(64);
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            f[i*8 + j] = (j / 3.5f) - 1.0f;
        }
    }
    return f;
}

std::vector<float> board_to_tensor(const chess::Board& board) {
    std::vector<float> tensor(8 * 8 * 16, 0.0f);
    board_to_tensor(board, tensor.data());
    return tensor;
}

void board_to_tensor(const chess::Board& board, float* buffer) {
    const static auto ranks = init_ranks();
    const static auto files = init_files();
    
    // Zero out buffer
    std::fill(buffer, buffer + 8*8*16, 0.0f);

    // Piece channels 0–11
    const std::vector<std::pair<Color, PieceType>> piece_channels = {
        {Color::WHITE, PieceType::KING}, {Color::WHITE, PieceType::QUEEN},
        {Color::WHITE, PieceType::ROOK}, {Color::WHITE, PieceType::BISHOP},
        {Color::WHITE, PieceType::KNIGHT}, {Color::WHITE, PieceType::PAWN},
        {Color::BLACK, PieceType::KING}, {Color::BLACK, PieceType::QUEEN},
        {Color::BLACK, PieceType::ROOK}, {Color::BLACK, PieceType::BISHOP},
        {Color::BLACK, PieceType::KNIGHT}, {Color::BLACK, PieceType::PAWN}
    };

    for (int square = 0; square < 64; ++square) {
        Square sq(square);
        Piece piece = board.at(sq);
        int row = square / 8;
        int col = square % 8;

        for (int ch = 0; ch < 12; ++ch) {
            auto [color, piece_type] = piece_channels[ch];
            if (piece != Piece::NONE && piece.color() == color && piece.type() == piece_type) {
                buffer[(row * 8 + col) * 16 + ch] = 1.0f;
            }
        }

        buffer[(row * 8 + col) * 16 + 12] = ranks[square];  // rank
        buffer[(row * 8 + col) * 16 + 13] = files[square];  // file
    }

    // King zones (channels 14–15)
    auto set_king_zone = [&](Color color, int channel) {
        Square ksq = board.kingSq(color);
        if (ksq.index() < 64) {
            int r0 = ksq.index() / 8;
            int c0 = ksq.index() % 8;
            for (int dr = -1; dr <= 1; ++dr) {
                for (int dc = -1; dc <= 1; ++dc) {
                    int r = r0 + dr;
                    int c = c0 + dc;
                    if (r >= 0 && r < 8 && c >= 0 && c < 8) {
                        buffer[(r * 8 + c) * 16 + channel] = 1.0f;
                    }
                }
            }
        }
    };

    set_king_zone(Color::WHITE, 14);
    set_king_zone(Color::BLACK, 15);
}

Evaluator::ModelSession Evaluator::createModelSession(const std::string& model_path) {
    #if defined(_WIN32)
        std::wstring wide_path = std::filesystem::path(model_path).wstring();
        Ort::Session session(env_, wide_path.c_str(), session_options_);
    #else
        Ort::Session session(env_, model_path.c_str(), session_options_);
    #endif

    ModelSession model_session(std::move(session));

    size_t in_count = model_session.session.GetInputCount();
    model_session.input_names_str.reserve(in_count);
    model_session.input_names_c.reserve(in_count);
    for (size_t i = 0; i < in_count; ++i) {
        auto name_ptr = model_session.session.GetInputNameAllocated(i, allocator_);
        model_session.input_names_str.emplace_back(name_ptr.get());
        model_session.input_names_c.push_back(model_session.input_names_str.back().c_str());
    }

    size_t out_count = model_session.session.GetOutputCount();
    model_session.output_names_str.reserve(out_count);
    model_session.output_names_c.reserve(out_count);
    for (size_t i = 0; i < out_count; ++i) {
        auto name_ptr = model_session.session.GetOutputNameAllocated(i, allocator_);
        model_session.output_names_str.emplace_back(name_ptr.get());
        model_session.output_names_c.push_back(model_session.output_names_str.back().c_str());
    }

    return model_session;
}

Evaluator::Evaluator(const std::string& white_model_path, const std::string& black_model_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "chess_eval"),
      session_options_(),
      allocator_(),
      white_session_(createModelSession(white_model_path)),
      black_session_(createModelSession(black_model_path)),
      input_buffer_(8*8*16)
{
    session_options_.SetIntraOpNumThreads(1);
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
}

float Evaluator::evaluate(const chess::Board &board) {
    board_to_tensor(board, input_buffer_.data());

    ModelSession* current_session = nullptr;
    if (board.sideToMove() == chess::Color::WHITE) {
        current_session = &white_session_;
    } else {
        current_session = &black_session_;
    }

    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info,
        input_buffer_.data(),
        input_buffer_.size(),
        shape_.data(), shape_.size());

    std::vector<Ort::Value> input_values;
    input_values.push_back(std::move(input_tensor));

    Ort::RunOptions run_options;
    auto outputs = current_session->session.Run(
        run_options,
        current_session->input_names_c.data(),
        input_values.data(),
        input_values.size(),
        current_session->output_names_c.data(),
        current_session->output_names_c.size()
    );

    float* out = outputs.front().GetTensorMutableData<float>();
    return out[0];
}

}