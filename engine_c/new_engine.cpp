#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <string>
#include <cctype>
#include <sstream>
#include <unordered_map>
#include <algorithm>
#include <iostream>

namespace py = pybind11;

class ChessEngine {
public:
    void set_position(const std::string& fen) {
        parse_fen(fen);
    }

    std::vector<std::string> get_legal_moves() {
        std::vector<std::string> moves;
        generate_moves(moves);
        return moves;
    }

    std::string make_move(const std::string& fen, const std::string& move) {
        set_position(fen);
        apply_move(move);
        return generate_fen();
    }

    bool is_game_over() {
        std::vector<std::string> moves;
        generate_moves(moves);
        return moves.empty();
    }

private:
    char board[8][8] = {{'.'}};
    bool white_to_move = true;
    int halfmove_clock = 0;
    int fullmove_number = 1;

    void parse_fen(const std::string& fen) {
        std::istringstream iss(fen);
        std::string board_fen, turn;
        iss >> board_fen >> turn;

        int row = 0, col = 0;
        for (char c : board_fen) {
            if (c == '/') {
                row++;
                col = 0;
            } else if (std::isdigit(c)) {
                int empty_squares = c - '0';
                for (int i = 0; i < empty_squares; ++i) {
                    board[row][col++] = '.'; // Use '.' for empty squares
                }
            } else {
                board[row][col++] = c;
            }
        }

        white_to_move = (turn == "w");
    }


    void generate_moves(std::vector<std::string>& moves) {
        for (int row = 0; row < 8; ++row) {
            for (int col = 0; col < 8; ++col) {
                char piece = board[row][col];
                if (piece == '.') continue;
                if (white_to_move && islower(piece)) continue;
                if (!white_to_move && isupper(piece)) continue;

                switch (tolower(piece)) {
                    case 'p': generate_pawn_moves(row, col, moves); break;
                    case 'n': generate_knight_moves(row, col, moves); break;
                    case 'b': generate_bishop_moves(row, col, moves); break;
                    case 'r': generate_rook_moves(row, col, moves); break;
                    case 'q': generate_queen_moves(row, col, moves); break;
                    case 'k': generate_king_moves(row, col, moves); break;
                }
            }
        }
    }

void generate_pawn_moves(int row, int col, std::vector<std::string>& moves) {
        int direction = white_to_move ? -1 : 1; // White moves up (-1), Black moves down (+1)
        int start_row = white_to_move ? 6 : 1;  // Starting row for white (6) and black (1)

        // Single forward move
        int next_row = row + direction;

//        std::cout << "Attempting single forward move for pawn at (" << row << "," << col << ")\n";
//        std::cout << "Calculated next_row: " << next_row << ", col: " << col << "\n";
//
//        std::cout << "Calling is_within_bounds for (" << next_row << "," << col << "): ";
//        bool within_bounds_single = is_within_bounds(next_row, col);
//        std::cout << (within_bounds_single ? "true" : "false") << "\n";
//
//        std::cout << "Calling is_empty for (" << next_row << "," << col << "): ";
//        bool empty_single = is_empty(next_row, col);
//        std::cout << (empty_single ? "true" : "false") << "\n";


        if (is_within_bounds(next_row, col) && is_empty(next_row, col)) {
            add_move(row, col, next_row, col, moves);

            // Double forward move (only if single forward is valid and on start row)
            int double_row = next_row + direction;
            if (row == start_row && is_within_bounds(double_row, col) && is_empty(double_row, col)) {
                add_move(row, col, double_row, col, moves);
            }
        }

        // Diagonal captures
        int left_diag = col - 1;
        int right_diag = col + 1;

        // Capture to the left diagonal
        if (is_within_bounds(next_row, left_diag) && is_enemy(next_row, left_diag)) {
            add_move(row, col, next_row, left_diag, moves);
        }

        // Capture to the right diagonal
        if (is_within_bounds(next_row, right_diag) && is_enemy(next_row, right_diag)) {
            add_move(row, col, next_row, right_diag, moves);
        }
    }



    void generate_knight_moves(int row, int col, std::vector<std::string>& moves) {
        static const int offsets[8][2] = {
            {2, 1}, {2, -1}, {-2, 1}, {-2, -1},
            {1, 2}, {1, -2}, {-1, 2}, {-1, -2}
        };
        for (const auto& offset : offsets) {
            int new_row = row + offset[0], new_col = col + offset[1];
            if (is_within_bounds(new_row, new_col) && !is_friendly(new_row, new_col)) {
                add_move(row, col, new_row, new_col, moves);
            }
        }
    }

    void generate_bishop_moves(int row, int col, std::vector<std::string>& moves) {
        static const int directions[4][2] = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
        generate_sliding_moves(row, col, directions, 4, moves);
    }

    void generate_rook_moves(int row, int col, std::vector<std::string>& moves) {
        static const int directions[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        generate_sliding_moves(row, col, directions, 4, moves);
    }

    void generate_queen_moves(int row, int col, std::vector<std::string>& moves) {
        static const int directions[8][2] = {
            {1, 1}, {1, -1}, {-1, 1}, {-1, -1},
            {1, 0}, {-1, 0}, {0, 1}, {0, -1}
        };
        generate_sliding_moves(row, col, directions, 8, moves);
    }

    void generate_king_moves(int row, int col, std::vector<std::string>& moves) {
        static const int offsets[8][2] = {
            {1, 1}, {1, -1}, {-1, 1}, {-1, -1},
            {1, 0}, {-1, 0}, {0, 1}, {0, -1}
        };
        for (const auto& offset : offsets) {
            int new_row = row + offset[0], new_col = col + offset[1];
            if (is_within_bounds(new_row, new_col) && !is_friendly(new_row, new_col)) {
                add_move(row, col, new_row, new_col, moves);
            }
        }
    }

    void generate_sliding_moves(int row, int col, const int directions[][2], int count, std::vector<std::string>& moves) {
        for (int i = 0; i < count; ++i) {
            int new_row = row, new_col = col;
            while (true) {
                new_row += directions[i][0];
                new_col += directions[i][1];
                if (!is_within_bounds(new_row, new_col) || is_friendly(new_row, new_col)) break;
                add_move(row, col, new_row, new_col, moves);
                if (!is_empty(new_row, new_col)) break;
            }
        }
    }

    void apply_move(const std::string& move) {
        int from_col = move[0] - 'a';
        int from_row = '8' - move[1];
        int to_col = move[2] - 'a';
        int to_row = '8' - move[3];

        board[to_row][to_col] = board[from_row][from_col];
        board[from_row][from_col] = '.';

        if (!white_to_move) fullmove_number++;
        white_to_move = !white_to_move;
    }

    std::string generate_fen() const {
        std::ostringstream fen;
        for (int row = 0; row < 8; ++row) {
            int empty_count = 0;
            for (int col = 0; col < 8; ++col) {
                if (board[row][col] == '.') {
                    empty_count++;
                } else {
                    if (empty_count > 0) {
                        fen << empty_count;
                        empty_count = 0;
                    }
                    fen << board[row][col];
                }
            }
            if (empty_count > 0) fen << empty_count;
            if (row < 7) fen << '/';
        }
        fen << ' ' << (white_to_move ? 'w' : 'b') << " - - 0 " << fullmove_number;
        return fen.str();
    }

    bool is_within_bounds(int row, int col) const {
        return row >= 0 && row < 8 && col >= 0 && col < 8;
    }


    bool is_empty(int row, int col) const {
//        std::cout << "board[" << row << "][" << col << "] = '" << board[row][col] << "'\n";
        return is_within_bounds(row, col) && board[row][col] == '.';
    }


    bool is_friendly(int row, int col) const {
        if (!is_within_bounds(row, col)) return false;
        return white_to_move ? isupper(board[row][col]) : islower(board[row][col]);
    }

    bool is_enemy(int row, int col) const {
        if (!is_within_bounds(row, col)) return false;
        return white_to_move ? islower(board[row][col]) : isupper(board[row][col]);
    }


    void add_move(int from_row, int from_col, int to_row, int to_col, std::vector<std::string>& moves) {
        char from_file = 'a' + from_col; // Column to file
        char from_rank = '8' - from_row; // Row to rank
        char to_file = 'a' + to_col;
        char to_rank = '8' - to_row;
        moves.emplace_back(std::string() + from_file + from_rank + to_file + to_rank);
    }


public:
    // Add this method to expose the board as a 2D vector
    std::vector<std::vector<char>> get_board() const {
        std::vector<std::vector<char>> python_board(8, std::vector<char>(8, '.'));
        for (int row = 0; row < 8; ++row) {
            for (int col = 0; col < 8; ++col) {
                python_board[row][col] = board[row][col];
            }
        }
        return python_board;
    }
};

// Pybind11 bindings
PYBIND11_MODULE(chess_moves, m) {
    py::class_<ChessEngine>(m, "ChessEngine")
        .def(py::init<>())
        .def("set_fen", &ChessEngine::set_position, py::arg("fen"))
        .def("legal_moves", &ChessEngine::get_legal_moves)
        .def("push", &ChessEngine::make_move, py::arg("fen"), py::arg("move"))
        .def("is_game_over", &ChessEngine::is_game_over)
        .def("get_board", &ChessEngine::get_board);
}
