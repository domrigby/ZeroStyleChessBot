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
#include <array>

constexpr int MAX_MOVES = 218
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
        int from_col = move[0] - 'a';
        int from_row = '8' - move[1];
        int to_col = move[2] - 'a';
        int to_row = '8' - move[3];

        char captured = board[to_row][to_col];
        char promotion_piece = (move.size() == 5)
            ? (white_to_move ? toupper(move[4]) : tolower(move[4]))
            : '.';

        // Apply the move
        board[to_row][to_col] = (promotion_piece != '.') ? promotion_piece : board[from_row][from_col];
        board[from_row][from_col] = '.';

        if (!white_to_move) fullmove_number++;
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

    std::pair<bool, int> is_game_over() {
        // Check for the 50-move rule
        if (halfmove_clock >= 100) {
            return {true, 2}; // 2 represents a draw due to the 50-move rule
        }

        // Get all legal moves for the current side to move
        std::vector<std::string> moves = get_legal_moves();

        // If there are no legal moves, it’s either stalemate or checkmate
        if (moves.empty()) {
            if (is_king_in_check(white_to_move)) {
                return {true, 1}; // Checkmate
            } else {
                return {true, 0}; // Stalemate
            }
        }

        // Otherwise, the game is not over
        return {false, -1};
    }


    py::array_t<float> fen_to_tensor(const std::string &fen)
    {
        // 1. Parse the FEN to detect the side to move.
        //    A typical FEN has fields like "rnbqkbnr/8/8/8/8/8/8/RNBQKBNR w KQkq - 0 1"
        //    The side-to-move is right after the first space-separated field.
        std::stringstream ss(fen);
        std::string board_part, side_to_move_str;
        ss >> board_part >> side_to_move_str;
        bool black_to_move = (side_to_move_str == "b");

        // 2. Set up your internal board data structure from the entire FEN.
        //    This presumably fills board[8][8].
        set_position(fen);

        // 3. If Black is to move, we will flip the board in our internal representation
        //    so that from the tensor’s perspective, Black is "on bottom."
        //    (Alternatively, you can fill the tensor with reversed indices;
        //     flipping the board array first is often simpler.)
        if (black_to_move)
        {
            // Flip board in-place 180 degrees
            // The top-left corner becomes bottom-right, etc.
            for (int row = 0; row < 4; ++row) {
                for (int col = 0; col < 8; ++col) {
                    std::swap(board[row][col], board[7 - row][7 - col]);
                }
            }
        }

        // 4. Create a 3D tensor with shape (12, 8, 8).
        //    We’ll store them in row-major for convenience, then construct the PyArray.
        std::vector<float> tensor(12 * 8 * 8, 0.0f);

        // This lambda encodes a single piece into the correct “channel”.
        auto encode_piece = [&](char piece, int row, int col) {
            if (piece == '.') return;

            bool is_white = std::isupper(piece);
            char lower_piece = std::tolower(piece);

            // Assign piece_type_index from 0..5 (king->0, queen->1, bishop->2, knight->3, rook->4, pawn->5)
            int piece_type_index = -1;
            if      (lower_piece == 'k') piece_type_index = 0;
            else if (lower_piece == 'q') piece_type_index = 1;
            else if (lower_piece == 'b') piece_type_index = 2;
            else if (lower_piece == 'n') piece_type_index = 3;
            else if (lower_piece == 'r') piece_type_index = 4;
            else if (lower_piece == 'p') piece_type_index = 5;
            if (piece_type_index < 0) return; // unknown

            // Now determine which side is “friendly” in channels 0..5 and “enemy” in channels 6..11.
            //
            // If black_to_move is true, that means black is “friendly,” so black’s pieces go in channels 0..5.
            // If black_to_move is false, white is “friendly,” so white’s pieces go in channels 0..5.
            bool piece_is_friendly = black_to_move ? !is_white : is_white;

            int plane_index = piece_type_index + (piece_is_friendly ? 0 : 6);
            // plane_index is 0..5 if friendly, 6..11 if enemy.

            // Fill the tensor
            // Flattened index = plane_index * 64 + (row * 8 + col)
            // row, col are [0..7].
            tensor[plane_index * 64 + row * 8 + col] = 1.0f;
        };

        // 5. Encode the board into the tensor
        for (int row = 0; row < 8; ++row) {
            for (int col = 0; col < 8; ++col) {
                char piece = board[row][col];
                encode_piece(piece, row, col);
            }
        }

        // 6. Return a py::array_t<float> with shape [12, 8, 8].
        return py::array_t<float>({12, 8, 8}, tensor.data());
    }


    std::string unflip_move(const std::string& move) {
        if (move == "O-O" || move == "O-O-O") {
            return move; // Castling moves do not change when flipped
        }

        if (move.size() < 4) {
            throw std::runtime_error("Move string too short");
        }

        char from_file = move[0];
        char from_rank = move[1];
        char to_file = move[2];
        char to_rank = move[3];

        auto algebraic_to_rc = [&](char file, char rank) {
            int c = file - 'a';
            int r = 8 - (rank - '0');
            return std::make_pair(r, c);
        };

        auto [from_r, from_c] = algebraic_to_rc(from_file, from_rank);
        auto [to_r, to_c] = algebraic_to_rc(to_file, to_rank);

        from_r = 7 - from_r;
        from_c = 7 - from_c;
        to_r = 7 - to_r;
        to_c = 7 - to_c;

        auto rc_to_algebraic = [&](int r, int c) {
            char file = 'a' + c;
            char rank = '0' + (8 - r);
            return std::string{file, rank};
        };

        std::string new_from = rc_to_algebraic(from_r, from_c);
        std::string new_to = rc_to_algebraic(to_r, to_c);
        std::string promotion_part = (move.size() == 5) ? move.substr(4) : "";

        return new_from + new_to + promotion_part;
    }


    py::array_t<float> move_to_target(const std::string& move) {
        if (move.size() < 4 || move.size() > 5 ||
            (move != "O-O" && move != "O-O-O" &&
             (move[0] < 'a' || move[0] > 'h' || move[2] < 'a' || move[2] > 'h' ||
              move[1] < '1' || move[1] > '8' || move[3] < '1' || move[3] > '8'))) {
            throw std::invalid_argument("Invalid move format");
        }

        std::vector<float> target(70 * 8 * 8, 0.0f); // Extra 4 channels for promotions
        int channel = -1;
        int from_row = 0, from_col = 0;

        if (move == "O-O") {
            channel = 64; // Kingside castling
        } else if (move == "O-O-O") {
            channel = 65; // Queenside castling
        } else {
            from_row = '8' - move[1];
            from_col = move[0] - 'a';

            if (move.size() == 5) {
                switch (move[4]) {
                    case 'q': channel = 66; break; // Promotion to Queen
                    case 'r': channel = 67; break; // Promotion to Rook
                    case 'b': channel = 68; break; // Promotion to Bishop
                    case 'n': channel = 69; break; // Promotion to Knight
                    default: throw std::invalid_argument("Invalid promotion piece");
                }
            } else {
                int to_row = '8' - move[3];
                int to_col = move[2] - 'a';
                channel = calculate_channel(from_row, from_col, to_row, to_col);
            }
        }

        if (channel == -1) {
            throw std::invalid_argument("Invalid move");
        }

        target[channel * 64 + from_row * 8 + from_col] = 1.0f;
        return py::array_t<float>({70, 8, 8}, target.data());
    }

     std::tuple<int, int, int> move_to_target_indices(const std::string& move) {
        if ((move.size() != 4 && move.size() != 5 && move != "O-O" && move != "O-O-O") ||
            (move != "O-O" && move != "O-O-O" &&
             (move[0] < 'a' || move[0] > 'h' || move[2] < 'a' || move[2] > 'h' ||
              move[1] < '1' || move[1] > '8' || move[3] < '1' || move[3] > '8'))) {
            throw std::invalid_argument("Invalid move format");
        }

        int channel = -1;
        int from_row = 0, from_col = 0;

        if (move == "O-O") {
            channel = 64; // Kingside castling
        } else if (move == "O-O-O") {
            channel = 65; // Queenside castling
        } else {
            from_col = move[0] - 'a';
            from_row = '8' - move[1];

            if (move.size() == 5) {
                switch (move[4]) {
                    case 'q': channel = 66; break;
                    case 'r': channel = 67; break;
                    case 'b': channel = 68; break;
                    case 'n': channel = 69; break;
                    default: throw std::invalid_argument("Invalid promotion piece");
                }
            } else {
                int to_col = move[2] - 'a';
                int to_row = '8' - move[3];
                channel = calculate_channel(from_row, from_col, to_row, to_col);
            }
        }

        if (channel == -1) {
            throw std::invalid_argument("Invalid move");
        }

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
        } else if (channel >= 66 && channel <= 69) {
            char promotion_piece = "qrbn"[channel - 66];
            char from_file = 'a' + from_col;
            char from_rank = '8' - from_row;
            char to_file = from_file;
            char to_rank = (white_to_move ? '1' : '8'); // Promotion rank depends on side to move
            return std::string({from_file, from_rank, to_file, to_rank, promotion_piece});
        }

        int to_row = from_row;
        int to_col = from_col;

        if (channel >= 0 && channel < 56) {
            int direction = channel % 8;
            int distance = (channel / 8) + 1;
            switch (direction) {
                case 0: to_row += distance; break;
                case 1: to_row -= distance; break;
                case 2: to_col += distance; break;
                case 3: to_col -= distance; break;
                case 4: to_row += distance; to_col += distance; break;
                case 5: to_row += distance; to_col -= distance; break;
                case 6: to_row -= distance; to_col += distance; break;
                case 7: to_row -= distance; to_col -= distance; break;
            }
        } else if (channel >= 56 && channel < 64) {
            static const int knight_offsets[8][2] = {
                {2, 1}, {2, -1}, {-2, 1}, {-2, -1}, {1, 2}, {1, -2}, {-1, 2}, {-1, -2}
            };
            int knight_index = channel - 56;
            to_row += knight_offsets[knight_index][0];
            to_col += knight_offsets[knight_index][1];
        } else {
            throw std::invalid_argument("Invalid channel");
        }

        char from_file = 'a' + from_col;
        char from_rank = '8' - from_row;
        char to_file = 'a' + to_col;
        char to_rank = '8' - to_row;
        return std::string({from_file, from_rank, to_file, to_rank});
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

        // We’ll read at least 6 tokens from the FEN (some may be unused):
        //   1) board_fen
        //   2) turn
        //   3) castling rights (unused here, but we must read it)
        //   4) en_passant (unused here, but we must read it)
        //   5) halfmove clock
        //   6) fullmove number
        std::string board_fen, turn, castling, en_passant, halfmove_s, fullmove_s;
        iss >> board_fen >> turn >> castling >> en_passant >> halfmove_s >> fullmove_s;

        // Clear the board
        for (int r = 0; r < 8; ++r) {
            for (int c = 0; c < 8; ++c) {
                board[r][c] = '.';
            }
        }

        // Parse the board portion (board_fen)
        int row = 0, col = 0;
        for (char c : board_fen) {
            if (c == '/') {
                row++;
                col = 0;
            } else if (std::isdigit(static_cast<unsigned char>(c))) {
                int empty_squares = c - '0';
                for (int i = 0; i < empty_squares; ++i) {
                    board[row][col++] = '.';
                }
            } else {
                board[row][col++] = c;
            }
        }

        // Which side to move
        white_to_move = (turn == "w");

        // Parse halfmove clock
        if (!halfmove_s.empty()) {
            halfmove_clock = std::stoi(halfmove_s);
        } else {
            halfmove_clock = 0;
        }

        // Parse fullmove number
        if (!fullmove_s.empty()) {
            fullmove_number = std::stoi(fullmove_s);
        } else {
            fullmove_number = 1;
        }
    }


    void handle_castling(const std::string& move) {
        if (move == "O-O") {
            int row = white_to_move ? 7 : 0;
            board[row][6] = board[row][4]; // Move king
            board[row][5] = board[row][7]; // Move rook
            board[row][4] = '.';
            board[row][7] = '.';
        } else if (move == "O-O-O") {
            int row = white_to_move ? 7 : 0;
            board[row][2] = board[row][4]; // Move king
            board[row][3] = board[row][0]; // Move rook
            board[row][4] = '.';
            board[row][0] = '.';
        }

        if (!white_to_move) fullmove_number++;
        white_to_move = !white_to_move;
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
        int promotion_row = white_to_move ? 0 : 7; // Row where promotion occurs

        // Single forward move
        int next_row = row + direction;
        if (is_within_bounds(next_row, col) && is_empty(next_row, col)) {
            if (next_row == promotion_row) {
                // Generate promotion moves
                for (char promo : { 'q', 'r', 'b', 'n' }) {
                    add_move(row, col, next_row, col, moves, promo);
                }
            } else {
                add_move(row, col, next_row, col, moves);

                // Double forward move (only if single forward is valid and on start row)
                int double_row = next_row + direction;
                if (row == start_row && is_within_bounds(double_row, col) && is_empty(double_row, col)) {
                    add_move(row, col, double_row, col, moves);
                }
            }
        }

        // Diagonal captures
        for (int offset : { -1, 1 }) {
            int diag_col = col + offset;
            if (is_within_bounds(next_row, diag_col) && is_enemy(next_row, diag_col)) {
                if (next_row == promotion_row) {
                    for (char promo : { 'q', 'r', 'b', 'n' }) {
                        add_move(row, col, next_row, diag_col, moves, promo);
                    }
                } else {
                    add_move(row, col, next_row, diag_col, moves);
                }
            }
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
        // If castling, handle that separately
        if (move == "O-O" || move == "O-O-O") {
            handle_castling(move);
            return;
        }

        // Get from/to squares
        int from_col = move[0] - 'a';
        int from_row = '8' - move[1];
        int to_col   = move[2] - 'a';
        int to_row   = '8' - move[3];

        // Check for promotion piece
        char promotion_piece = '.';
        if (move.size() == 5) {
            promotion_piece = white_to_move ? toupper(move[4]) : tolower(move[4]);
        }

        // Check if it's a capture
        bool is_capture = (board[to_row][to_col] != '.');

        // Check if it's a pawn move
        char moving_piece = board[from_row][from_col];
        bool is_pawn_move = (std::tolower(moving_piece) == 'p');

        // Apply the move
        board[to_row][to_col] = (promotion_piece != '.') ? promotion_piece : moving_piece;
        board[from_row][from_col] = '.';

        // Update halfmove clock
        if (is_capture || is_pawn_move) {
            halfmove_clock = 0;
        } else {
            halfmove_clock++;
        }

        // Update fullmove number and side to move
        if (!white_to_move) {
            fullmove_number++;
        }
        white_to_move = !white_to_move;
    }


    std::string generate_fen() const {
        std::ostringstream fen;

        // 1) Board layout
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

        // 2) Side to move
        fen << ' ' << (white_to_move ? 'w' : 'b');

        // 3) Castling rights – ignoring for now, so output '-'
        fen << " -";

        // 4) En passant square – ignoring for now, so output '-'
        fen << " -";

        // 5) Halfmove clock
        fen << " " << halfmove_clock;

        // 6) Fullmove number
        fen << " " << fullmove_number;

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


    void add_move(int from_row, int from_col, int to_row, int to_col, std::vector<std::string>& moves, char promotion = '\0') {
        char from_file = 'a' + from_col; // Column to file
        char from_rank = '8' - from_row; // Row to rank
        char to_file = 'a' + to_col;
        char to_rank = '8' - to_row;
        std::string move = std::string() + from_file + from_rank + to_file + to_rank;
        if (promotion) {
            move += promotion;
        }
        moves.push_back(move);
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
        .def("indices_to_move", &ChessEngine::indices_to_move, py::arg("channel"), py::arg("from_row"), py::arg("from_col"))
        .def("unflip_move", &ChessEngine::unflip_move, py::arg("move"));
}
