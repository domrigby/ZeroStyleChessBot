#!/bin/bash
mkdir build && cd build
cmake ..
make

## Move the final executable to folder above
mv chess_moves*.so ..