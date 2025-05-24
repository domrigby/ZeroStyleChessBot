cmake_minimum_required(VERSION 3.12)
project(chess_moves)

set(CMAKE_CXX_STANDARD 14)

# bring in pybind11 as a “subproject”
include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.11.1
)
FetchContent_MakeAvailable(pybind11)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# now pybind11 is available
pybind11_add_module(chess_moves cpp_chess_env/new_engine.cpp)

target_compile_options(chess_moves PRIVATE -Wall -Wextra -O3)
