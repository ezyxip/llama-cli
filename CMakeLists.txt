cmake_minimum_required(VERSION 3.5)

project(llama-cli
    VERSION 0.1.0
    DESCRIPTION "Llama-wrapper"
    LANGUAGES CXX
)

option(ENABLE_WARNINGS "Включить строгие предупреждения" ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

if(ENABLE_WARNINGS)
    if(MSVC)
        add_compile_options(/W4 /WX)
    else()
        add_compile_options(-Wall -Wextra -Wpedantic -Wshadow -Wconversion)
        if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            add_compile_options(-Weverything -Wno-c++98-compat)
        endif()
    endif()
endif()

file(GLOB_RECURSE SOURCES 
    src/*.cpp
    src/**/*.cpp
)

# add_subdirectory(llama.cpp)

add_executable(${PROJECT_NAME} ${SOURCES})

target_link_libraries(${PROJECT_NAME} PRIVATE llama)

# target_include_directories(${PROJECT_NAME} 
#     PRIVATE 
#         ${CMAKE_CURRENT_SOURCE_DIR}/include
#         ${CMAKE_CURRENT_SOURCE_DIR}/src
# )

include(GNUInstallDirs)

install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# install(DIRECTORY model DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/models)