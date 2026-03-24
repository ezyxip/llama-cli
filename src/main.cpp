#include "LlamaChat.h"
#include <iostream>

int main(int argc, char** argv) {
    // В идеале можно использовать парсер аргументов (например CLI11 или argp),
    // но для простоты берем первый аргумент.
    if (argc < 2) {
        std::cerr << "Использование: " << argv[0] << " <путь_к_модели.gguf>" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];

    try {
        std::cout << "Загрузка модели из " << model_path << "..." << std::endl;
        
        LlamaChat chat(model_path);
        chat.start_chat();

    } catch (const std::exception& e) {
        std::cerr << "Критическая ошибка: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}