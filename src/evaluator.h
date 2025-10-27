#ifndef EVALUATOR_H
#define EVALUATOR_H

#include <vector>
#include <array>
#include "chess.hpp"
#include <onnxruntime_cxx_api.h>
#include <filesystem>
#include <stdexcept>
namespace chess {

static std::vector<float> init_ranks();
static std::vector<float> init_files();
std::vector<float> board_to_tensor(const chess::Board& board);
void board_to_tensor(const chess::Board& board, float* buffer);

class Evaluator {
    public:
        Evaluator(const std::string& white_model_path, const std::string& black_model_path);
        float evaluate(const chess::Board& board);

    private:
        struct ModelSession {
            ModelSession(Ort::Session&& s) : session(std::move(s)) {}
            
            Ort::Session session;
            std::vector<std::string> input_names_str;
            std::vector<const char*> input_names_c;
            std::vector<std::string> output_names_str;
            std::vector<const char*> output_names_c;
        };

        ModelSession createModelSession(const std::string& model_path);

        Ort::Env env_;
        Ort::SessionOptions session_options_;
        Ort::AllocatorWithDefaultOptions allocator_;

        ModelSession white_session_;
        ModelSession black_session_;

        std::vector<float> input_buffer_; 
        const std::array<int64_t,4> shape_{{1,8,8,16}};
};

}

#endif // EVALUATOR_H