#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
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
        // Generate all pseudo-legal moves
        std::vector<std::string> pseudo_moves;
        generate_moves(pseudo_moves);

        std::vector<std::string> legal_moves;
        bool sideToMove = white_to_move; // Side whose king we are checking

        // Check each pseudo-legal move
        for (const auto& mv : pseudo_moves) {
            char captured = apply_move_with_capture(mv);

            // Only add the move if it doesn't leave the king in check
            if (!is_king_in_check(sideToMove)) {
                legal_moves.push_back(mv);
            }

            undo_move(mv, captured);
        }

        return legal_moves;
    }

    bool is_king_in_check(bool sideToCheck) {
        // Determine which king to check for
        char kingSymbol = sideToCheck ? 'K' : 'k';

        // Find the king's position
        int kingRow = -1, kingCol = -1;
        for (int r = 0; r < 8; ++r) {
            for (int c = 0; c < 8; ++c) {
                if (board[r][c] == kingSymbol) {
                    kingRow = r;
                    kingCol = c;
                    break;
                }
            }
            if (kingRow != -1) break;
        }

        // If the king is not found (invalid board state), assume not in check
        if (kingRow == -1) {
            return false;
        }

        // Temporarily switch to the opponent's side to generate their moves
        bool savedSide = white_to_move;
        white_to_move = !sideToCheck;
        std::vector<std::string> opponentMoves;
        generate_moves(opponentMoves);
        white_to_move = savedSide;

        // Check if any opponent move attacks the king's position
        for (const auto& mv : opponentMoves) {
            int toRow = '8' - mv[3];
            int toCol = mv[2] - 'a';
            if (toRow == kingRow && toCol == kingCol) {
                return true;
            }
        }

        return false;
    }

    char apply_move_with_capture(const std::string& move) {
        int fromCol = move[0] - 'a';
        int fromRow = '8' - move[1];
        int toCol   = move[2] - 'a';
        int toRow   = '8' - move[3];

        // Save the captured piece
        char captured = board[toRow][toCol];

        // Apply the move
        board[toRow][toCol] = board[fromRow][fromCol];
        board[fromRow][fromCol] = '.';

        // Update game state
        if (!white_to_move) {
            fullmove_number++;
        }
        white_to_move = !white_to_move;

        return captured;
    }

    void undo_move(const std::string& move, char captured) {
        int fromCol = move[0] - 'a';
        int fromRow = '8' - move[1];
        int toCol   = move[2] - 'a';
        int toRow   = '8' - move[3];

        // Revert the move
        board[fromRow][fromCol] = board[toRow][toCol];
        board[toRow][toCol] = captured;

        // Restore game state
        white_to_move = !white_to_move;
        if (!white_to_move) {
            fullmove_number--;
        }
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

    py::array_t<float> fen_to_tensor(const std::string& fen) {
            set_position(fen);

            // Create a 3D tensor with shape (12, 8, 8)
            std::vector<float> tensor(12 * 8 * 8, 0.0f);

            auto encode_piece = [&](char piece, int row, int col) {
                int plane_index = -1;
                bool is_white = isupper(piece);
                char lower_piece = tolower(piece);

                if (lower_piece == 'k') plane_index = 0;
                else if (lower_piece == 'q') plane_index = 1;
                else if (lower_piece == 'b') plane_index = 2;
                else if (lower_piece == 'n') plane_index = 3;
                else if (lower_piece == 'r') plane_index = 4;
                else if (lower_piece == 'p') plane_index = 5;

                if (plane_index != -1) {
                    if (!is_white) plane_index += 6; // Opponent pieces are in the next 6 planes
                    tensor[plane_index * 64 + row * 8 + col] = 1.0f;
                }
            };

            // Encode the board into the tensor
            for (int row = 0; row < 8; ++row) {
                for (int col = 0; col < 8; ++col) {
                    char piece = board[row][col];
                    if (piece != '.') {
                        encode_piece(piece, row, col);
                    }
                }
            }

            return py::array_t<float>({12, 8, 8}, tensor.data());
        }

    py::array_t<float> move_to_target(const std::string& move) {
        // Ensure the move string is valid
        if ((move.size() != 4 && move != "O-O" && move != "O-O-O") ||
            (move != "O-O" && move != "O-O-O" &&
             (move[0] < 'a' || move[0] > 'h' || move[2] < 'a' || move[2] > 'h' ||
              move[1] < '1' || move[1] > '8' || move[3] < '1' || move[3] > '8'))) {
            throw std::invalid_argument("Invalid move format");
        }

        // Create a 3D tensor with shape (66, 8, 8), initialized to zero
        std::vector<float> target(66 * 8 * 8, 0.0f);

        int channel = -1;
        int from_row = 0, from_col = 0; // Default initialization

        if (move == "O-O") {
            // Kingside castling
            channel = 64; // Assign channel 64 for kingside castling
        } else if (move == "O-O-O") {
            // Queenside castling
            channel = 65; // Assign channel 65 for queenside castling
        } else {
            // Extract move details
            from_col = move[0] - 'a';
            from_row = '8' - move[1];
            int to_col = move[2] - 'a';
            int to_row = '8' - move[3];

            // Determine the channel using helper
            channel = calculate_channel(from_row, from_col, to_row, to_col);
        }

        if (channel == -1) {
            throw std::invalid_argument("Invalid move");
        }

        // One-hot encode the move in the tensor
        target[channel * 64 + from_row * 8 + from_col] = 1.0f;

        return py::array_t<float>({66, 8, 8}, target.data());
    }

    std::tuple<int, int, int> move_to_target_indices(const std::string& move) {
        // Ensure the move string is valid
        if ((move.size() != 4 && move != "O-O" && move != "O-O-O") ||
            (move != "O-O" && move != "O-O-O" &&
             (move[0] < 'a' || move[0] > 'h' || move[2] < 'a' || move[2] > 'h' ||
              move[1] < '1' || move[1] > '8' || move[3] < '1' || move[3] > '8'))) {
            throw std::invalid_argument("Invalid move format");
        }

        int channel = -1;
        int from_row = 0, from_col = 0; // Default initialization

        if (move == "O-O") {
            // Kingside castling
            channel = 64; // Assign channel 64 for kingside castling
        } else if (move == "O-O-O") {
            // Queenside castling
            channel = 65; // Assign channel 65 for queenside castling
        } else {
            // Extract move details
            from_col = move[0] - 'a';
            from_row = '8' - move[1];
            int to_col = move[2] - 'a';
            int to_row = '8' - move[3];

            // Determine the channel using helper
            channel = calculate_channel(from_row, from_col, to_row, to_col);
        }

        if (channel == -1) {
            throw std::invalid_argument("Invalid move");
        }

        // Return the channel, row, and column indices
        return std::make_tuple(channel, from_row, from_col);
    }

    // Helper function to calculate the channel
    int calculate_channel(int from_row, int from_col, int to_row, int to_col) {
        int row_diff = to_row - from_row;
        int col_diff = to_col - from_col;

        int channel = -1;

        // Handle queen-like moves
        if (row_diff != 0 && col_diff == 0) channel = (row_diff > 0) ? 0 : 1;  // Up/Down
        else if (row_diff == 0 && col_diff != 0) channel = (col_diff > 0) ? 2 : 3;  // Right/Left
        else if (abs(row_diff) == abs(col_diff)) {
            channel = (row_diff > 0) ? ((col_diff > 0) ? 4 : 5) : ((col_diff > 0) ? 6 : 7);  // Diagonals
        }

        // Check valid queen move distance
        int distance = std::max(abs(row_diff), abs(col_diff));
        if (distance <= 7 && channel != -1) {
            channel += (distance - 1) * 8;
        } else {
            // Handle knight-like moves
            static const int knight_offsets[8][2] = {
                {2, 1}, {2, -1}, {-2, 1}, {-2, -1},
                {1, 2}, {1, -2}, {-1, 2}, {-1, -2}
            };

            for (int i = 0; i < 8; ++i) {
                if (row_diff == knight_offsets[i][0] && col_diff == knight_offsets[i][1]) {
                    channel = 56 + i; // Knight channels start at 56
                    break;
                }
            }
        }

        if (channel == -1) {
            throw std::invalid_argument("Invalid move");
        }

        return channel;
    }

std::string indices_to_move(int channel, int from_row, int from_col) {
    if (channel == 64) {
        return "O-O";   // Kingside castling
    } else if (channel == 65) {
        return "O-O-O"; // Queenside castling
    }

    int to_row = from_row;
    int to_col = from_col;

    if (channel >= 0 && channel < 56) {
        // Queen-like moves
        int direction = channel % 8;
        int distance = (channel / 8) + 1;

        switch (direction) {
            case 0: // "Up" (increase row index)
                to_row = from_row + distance;
                break;
            case 1: // "Down" (decrease row index)
                to_row = from_row - distance;
                break;
            case 2: // "Right"
                to_col = from_col + distance;
                break;
            case 3: // "Left"
                to_col = from_col - distance;
                break;
            case 4: // "Up-Right"
                to_row = from_row + distance;
                to_col = from_col + distance;
                break;
            case 5: // "Up-Left"
                to_row = from_row + distance;
                to_col = from_col - distance;
                break;
            case 6: // "Down-Right"
                to_row = from_row - distance;
                to_col = from_col + distance;
                break;
            case 7: // "Down-Left"
                to_row = from_row - distance;
                to_col = from_col - distance;
                break;
            default:
                throw std::invalid_argument("Invalid direction for queen-like move");
        }
    } else if (channel >= 56 && channel < 64) {
        // Knight moves
        static const int knight_offsets[8][2] = {
            { 2,  1}, { 2, -1}, {-2,  1}, {-2, -1},
            { 1,  2}, { 1, -2}, {-1,  2}, {-1, -2}
        };
        int knight_index = channel - 56;
        to_row = from_row + knight_offsets[knight_index][0];
        to_col = from_col + knight_offsets[knight_index][1];
    } else {
        throw std::invalid_argument("Invalid channel");
    }

    // Validate the resulting indices are within bounds
    if (to_row < 0 || to_row > 7 || to_col < 0 || to_col > 7) {
        throw std::invalid_argument("Calculated move is out of bounds");
    }

    // Convert the from and to indices into algebraic notation
    char from_col_char = 'a' + from_col;
    char from_row_char = '8' - from_row;
    char to_col_char   = 'a' + to_col;
    char to_row_char   = '8' - to_row;

    return std::string({from_col_char, from_row_char, to_col_char, to_row_char});
}


    py::array_t<float> moves_to_board_tensor(const std::vector<std::string>& moves, bool white_to_play) {
        // Reset the board to the starting position
        set_position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

        // Apply moves sequentially
        for (const auto& move : moves) {
            apply_move(move);
        }

        // Create a 3D tensor with shape (12, 8, 8)
        py::array_t<float> tensor_array({12, 8, 8});
        auto tensor = tensor_array.mutable_data();

        auto encode_piece = [&](char piece, int row, int col) {
            int plane_index = -1;
            bool is_white = isupper(piece);
            char lower_piece = tolower(piece);

            if (lower_piece == 'k') plane_index = 0;
            else if (lower_piece == 'q') plane_index = 1;
            else if (lower_piece == 'b') plane_index = 2;
            else if (lower_piece == 'n') plane_index = 3;
            else if (lower_piece == 'r') plane_index = 4;
            else if (lower_piece == 'p') plane_index = 5;

            if (plane_index != -1) {
                if (!is_white == white_to_play) plane_index += 6; // Opponent pieces are in the next 6 planes
                tensor[plane_index * 64 + row * 8 + col] = 1.0f;
            }
        };

        // Encode the board into the tensor
        for (int row = 0; row < 8; ++row) {
            for (int col = 0; col < 8; ++col) {
                char piece = board[row][col];
                if (piece != '.') {
                    encode_piece(piece, row, col);
                }
            }
        }

        return tensor_array;
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
        .def("get_board", &ChessEngine::get_board)
        .def("fen_to_tensor", &ChessEngine::fen_to_tensor, py::arg("fen"))
        .def("move_to_target", &ChessEngine::move_to_target, py::arg("move"))
        .def("move_to_target_indices", &ChessEngine::move_to_target_indices, py::arg("move"))
        .def("moves_to_board_tensor", &ChessEngine::moves_to_board_tensor, py::arg("moves"), py::arg("white_to_play"))
        .def("indices_to_move", &ChessEngine::indices_to_move, py::arg("channel"), py::arg("from_row"), py::arg("from_col"));
}
