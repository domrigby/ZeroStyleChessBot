cmake_minimum_required(VERSION 3.15...3.31)
project(chess_moves LANGUAGES CXX)

# Structured bindings in new_engine.cpp need C++17.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# pybind11 is provided by the build environment (see [build-system] in pyproject.toml).
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(chess_moves cpp_chess_env/new_engine.cpp)

target_compile_options(chess_moves PRIVATE -Wall -Wextra -O3)

install(TARGETS chess_moves DESTINATION .)
