#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <string>
#include <cctype>
#include <sstream>
#include <unordered_map>
#include <algorithm>

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

        if (moves.empty()) {
            return is_king_in_check() ? true : true; // Checkmate or Stalemate
        }
        return false; // The game is not over
    }

private:
    char board[8][8] = {{'.'}};
    bool white_to_move = true;
    int halfmove_clock = 0;
    int fullmove_number = 1;

    void parse_fen(const std::string& fen) {
        std::istringstream iss(fen);
        std::string board_fen, turn, castling, en_passant;

        iss >> board_fen >> turn >> castling >> en_passant >> halfmove_clock >> fullmove_number;

        int row = 0, col = 0;
        for (char c : board_fen) {
            if (c == '/') {
                row++;
                col = 0;
            } else if (std::isdigit(c)) {
                col += c - '0';
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
        int direction = white_to_move ? -1 : 1;
        int start_row = white_to_move ? 6 : 1;

        if (is_empty(row + direction, col)) {
            add_move(row, col, row + direction, col, moves);
            if (row == start_row && is_empty(row + 2 * direction, col)) {
                add_move(row, col, row + 2 * direction, col, moves);
            }
        }
        if (is_enemy(row + direction, col - 1)) {
            add_move(row, col, row + direction, col - 1, moves);
        }
        if (is_enemy(row + direction, col + 1)) {
            add_move(row, col, row + direction, col + 1, moves);
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
        static const int directions[4][2] = {
            {1, 1}, {1, -1}, {-1, 1}, {-1, -1}
        };
        generate_sliding_moves(row, col, directions, 4, moves);
    }

    void generate_rook_moves(int row, int col, std::vector<std::string>& moves) {
        static const int directions[4][2] = {
            {1, 0}, {-1, 0}, {0, 1}, {0, -1}
        };
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

    void generate_sliding_moves(int row, int col, const int directions[][2], int direction_count, std::vector<std::string>& moves) {
        for (int i = 0; i < direction_count; ++i) {
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

        char piece = board[from_row][from_col];
        board[from_row][from_col] = '.';
        board[to_row][to_col] = piece;

        // Update move details
        if (tolower(piece) == 'p' || board[to_row][to_col] != '.') {
            halfmove_clock = 0;
        } else {
            halfmove_clock++;
        }

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

        fen << ' ' << (white_to_move ? 'w' : 'b')
            << " - - "
            << halfmove_clock << ' ' << fullmove_number;

        return fen.str();
    }

    bool is_king_in_check() {
        std::vector<std::string> moves;
        generate_moves(moves);
        return std::any_of(moves.begin(), moves.end(), [&](const std::string& move) {
            return board[move[2] - 'a'][move[3] - '8'] == (white_to_move ? 'K' : 'k');
        });
    }

    bool is_within_bounds(int row, int col) const {
        return row >= 0 && row < 8 && col >= 0 && col < 8;
    }

    bool is_empty(int row, int col) const {
        return is_within_bounds(row, col) && board[row][col] == '.';
    }

    bool is_friendly(int row, int col) const {
        if (!is_within_bounds(row, col)) return false;
        return white_to_move ? isupper(board[row][col]) : islower(board[row][col]);
    }

    bool is_enemy(int row, int col) const {
        return !is_empty(row, col) && !is_friendly(row, col);
    }

    void add_move(int from_row, int from_col, int to_row, int to_col, std::vector<std::string>& moves) {
        char from_file = 'a' + from_col;
        char from_rank = '8' - from_row;
        char to_file = 'a' + to_col;
        char to_rank = '8' - to_row;
        moves.emplace_back(std::string() + from_file + from_rank + to_file + to_rank);
    }
};

// Pybind11 bindings
PYBIND11_MODULE(chess_moves, m) {
    py::class_<ChessEngine>(m, "ChessEngine")
        .def(py::init<>())
        .def("set_position", &ChessEngine::set_position, py::arg("fen"))
        .def("get_legal_moves", &ChessEngine::get_legal_moves)
        .def("make_move", &ChessEngine::make_move, py::arg("fen"), py::arg("move"))
        .def("is_game_over", &ChessEngine::is_game_over);
}
