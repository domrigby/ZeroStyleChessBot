cmake_minimum_required(VERSION 3.12)
project(chess_moves)

set(CMAKE_CXX_STANDARD 14)

set(pybind11_DIR "/home/dom/.local/lib/python3.11/site-packages/pybind11/share/cmake/pybind11") # Adjust this path
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

pybind11_add_module(chess_moves cpp_chess_env/new_engine.cpp)

# Set compiler flags for this target
target_compile_options(chess_moves PRIVATE -Wall -Wextra -O3)