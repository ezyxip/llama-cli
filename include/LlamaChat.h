#pragma once

#include <string>
#include <vector>
#include <llama.h>

class LlamaChat {
public:
    explicit LlamaChat(const std::string& model_path);
    ~LlamaChat();

    LlamaChat(const LlamaChat&) = delete;
    LlamaChat& operator=(const LlamaChat&) = delete;

    void start_chat();

private:
    llama_model* m_model = nullptr;
    const llama_vocab* m_vocab = nullptr; // Добавлен указатель на словарь
    llama_context* m_ctx = nullptr;
    llama_sampler* m_sampler = nullptr;

    std::vector<llama_token> m_history_tokens;

    // Вспомогательные методы
    std::vector<llama_token> tokenize(const std::string& text, bool add_special, bool parse_special);
    std::string token_to_string(llama_token token);
    void generate_response(const std::string& user_input);

    // Вспомогательные методы для работы с llama_batch (замена удаленным из API)
    void batch_add(llama_batch& batch, llama_token id, llama_pos pos, bool logits);
    void batch_clear(llama_batch& batch);
};