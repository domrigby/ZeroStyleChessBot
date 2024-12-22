cmake_minimum_required(VERSION 3.12)
project(chess_moves)

set(CMAKE_CXX_STANDARD 14)

# Find Pybind11
set(pybind11_DIR "/home/dom/.local/lib/python3.11/site-packages/pybind11/share/cmake/pybind11") # Adjust this path
find_package(pybind11 REQUIRED)


# Add the target
pybind11_add_module(chess_moves engine.cpp)
