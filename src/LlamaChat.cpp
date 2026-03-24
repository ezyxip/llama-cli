#include "LlamaChat.h"
#include <iostream>
#include <stdexcept>
#include <cstdint>

LlamaChat::LlamaChat(const std::string &model_path)
{
    llama_backend_init();

    // 1. Загрузка модели (теперь параметры передаются по значению, без &)
    llama_model_params model_params = llama_model_default_params();
    m_model = llama_load_model_from_file(model_path.c_str(), model_params);
    if (!m_model)
    {
        throw std::runtime_error("Ошибка: Не удалось загрузить модель: " + model_path);
    }

    // 2. Получение словаря (API update)
    m_vocab = llama_model_get_vocab(m_model);

    // 3. Создание контекста (параметры по значению, без &)
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 2048;
    m_ctx = llama_new_context_with_model(m_model, ctx_params);
    if (!m_ctx)
    {
        throw std::runtime_error("Ошибка: Не удалось создать контекст llama.cpp");
    }

    // 4. Инициализация семплера (API update)
    llama_sampler_chain_params sampler_params = llama_sampler_chain_default_params();
    m_sampler = llama_sampler_chain_init(sampler_params);

    // Добавляем алгоритмы семплирования по новому стандарту API
    llama_sampler_chain_add(m_sampler, llama_sampler_init_temp(0.7f));
    llama_sampler_chain_add(m_sampler, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(m_sampler, llama_sampler_init_top_p(0.9f, 1));
    llama_sampler_chain_add(m_sampler, llama_sampler_init_dist(1234));
}

LlamaChat::~LlamaChat()
{
    if (m_sampler)
        llama_sampler_free(m_sampler);
    if (m_ctx)
        llama_free(m_ctx);
    if (m_model)
        llama_free_model(m_model);
    llama_backend_free();
}

// Замена старым llama_batch_add
void LlamaChat::batch_add(llama_batch &batch, llama_token id, llama_pos pos, bool logits)
{
    batch.token[batch.n_tokens] = id;
    batch.pos[batch.n_tokens] = pos;
    batch.n_seq_id[batch.n_tokens] = 1;
    batch.seq_id[batch.n_tokens][0] = 0;
    batch.logits[batch.n_tokens] = logits;
    batch.n_tokens++;
}

// Замена старым llama_batch_clear
void LlamaChat::batch_clear(llama_batch &batch)
{
    batch.n_tokens = 0;
}

std::vector<llama_token> LlamaChat::tokenize(const std::string &text, bool add_special, bool parse_special)
{
    int32_t max_tokens = static_cast<int32_t>(text.length()) + (add_special ? 1 : 0);
    std::vector<llama_token> result(static_cast<size_t>(max_tokens));

    // Передаем m_vocab вместо m_model
    int32_t n_tokens = llama_tokenize(
        m_vocab,
        text.c_str(),
        static_cast<int32_t>(text.length()),
        result.data(),
        max_tokens,
        add_special,
        parse_special);

    if (n_tokens < 0)
    {
        result.resize(static_cast<size_t>(-n_tokens));
        n_tokens = llama_tokenize(
            m_vocab,
            text.c_str(),
            static_cast<int32_t>(text.length()),
            result.data(),
            static_cast<int32_t>(result.size()),
            add_special,
            parse_special);
    }
    result.resize(static_cast<size_t>(n_tokens));
    return result;
}

std::string LlamaChat::token_to_string(llama_token token)
{
    std::vector<char> buf(8, 0);
    // Передаем m_vocab вместо m_model
    int32_t n_chars = llama_token_to_piece(m_vocab, token, buf.data(), static_cast<int32_t>(buf.size()), 0, true);
    if (n_chars < 0)
    {
        buf.resize(static_cast<size_t>(-n_chars));
        n_chars = llama_token_to_piece(m_vocab, token, buf.data(), static_cast<int32_t>(buf.size()), 0, true);
    }
    return std::string(buf.data(), static_cast<size_t>(n_chars));
}

void LlamaChat::generate_response(const std::string &user_input)
{
    std::string prompt = "<|im_start|>user\n" + user_input + "<|im_end|>\n<|im_start|>assistant\n";

    std::vector<llama_token> prompt_tokens = tokenize(prompt, false, true);
    m_history_tokens.insert(m_history_tokens.end(), prompt_tokens.begin(), prompt_tokens.end());

    llama_batch batch = llama_batch_init(512, 0, 1);

    for (size_t i = 0; i < prompt_tokens.size(); ++i)
    {
        bool is_last = (i == prompt_tokens.size() - 1);
        batch_add(batch, prompt_tokens[i], static_cast<llama_pos>(m_history_tokens.size() - prompt_tokens.size() + i), is_last);
    }

    std::cout << "Assistant: ";
    std::cout.flush();

    // Передаем m_vocab вместо m_model
    llama_token eos_token = llama_token_eos(m_vocab);
    llama_token eot_token = llama_token_eot(m_vocab);

    while (true)
    {
        if (llama_decode(m_ctx, batch) != 0)
        {
            std::cerr << "\n[Ошибка: llama_decode не удался]" << std::endl;
            break;
        }

        llama_token new_token_id = llama_sampler_sample(m_sampler, m_ctx, batch.n_tokens - 1);
        llama_sampler_accept(m_sampler, new_token_id);

        if (new_token_id == eos_token || new_token_id == eot_token)
        {
            break;
        }

        std::string token_str = token_to_string(new_token_id);
        std::cout << token_str;
        std::cout.flush();

        m_history_tokens.push_back(new_token_id);

        batch_clear(batch);
        batch_add(batch, new_token_id, static_cast<llama_pos>(m_history_tokens.size() - 1), true);
    }

    std::cout << std::endl;
    llama_batch_free(batch);
}

void LlamaChat::start_chat()
{
    std::cout << "=== Чат инициализирован. Для выхода введите 'exit' или 'quit'. ===" << std::endl;

    // Используем специальные теги <|im_start|> и <|im_end|>
    std::string system_prompt = "<|im_start|>system\nТы полезный, вежливый и умный ИИ-ассистент.<|im_end|>\n";
    std::vector<llama_token> sys_tokens = tokenize(system_prompt, true, true);
    m_history_tokens.insert(m_history_tokens.end(), sys_tokens.begin(), sys_tokens.end());

    std::string user_input;
    while (true)
    {
        std::cout << "\nUser: ";
        if (!std::getline(std::cin, user_input))
            break;
        if (user_input == "exit" || user_input == "quit")
            break;
        if (user_input.empty())
            continue;

        generate_response(user_input);
    }
}