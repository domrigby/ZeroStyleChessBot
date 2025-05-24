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
#include <cstdint>
using Bitboard = uint64_t;

namespace py = pybind11;

enum PieceIndex {
    WHITE_KING = 0,
    WHITE_QUEEN,
    WHITE_BISHOP,
    WHITE_KNIGHT,
    WHITE_ROOK,
    WHITE_PAWN,
    BLACK_KING,
    BLACK_QUEEN,
    BLACK_BISHOP,
    BLACK_KNIGHT,
    BLACK_ROOK,
    BLACK_PAWN,
    PIECE_NB // total count (should be 12)
};

static const char pieceSymbols[PIECE_NB] = {
    'K','Q','B','N','R','P',  // white pieces
    'k','q','b','n','r','p'   // black pieces
};


class ChessEngine {
public:

    void set_position(const std::string& fen) {
        parse_fen(fen);
    }

    std::vector<std::string> get_legal_moves() {
        // Generate all pseudo-legal moves based on the bitboard state.
        std::vector<std::string> pseudo_moves;
        generate_moves(pseudo_moves);

        std::vector<std::string> legal_moves;
        bool sideToMove = white_to_move;  // Save the current side to check king safety later

        // For each pseudo-legal move, apply it, check if the king is safe, then undo it.
        for (const auto& mv : pseudo_moves) {
            // Apply the move using bitboards; this returns the captured piece (if any)
            char captured = apply_move_with_capture(mv);
            // If the king of the side that just moved is not in check, the move is legal.
            if (!is_king_in_check(sideToMove)) {
                legal_moves.push_back(mv);
            }
            // Undo the move so we can test the next one
            undo_move(mv, captured);
        }

        return legal_moves;
    }

    bool is_king_in_check(bool sideToCheck) {
        // Determine which king we are checking.
        int kingIndex = sideToCheck ? WHITE_KING : BLACK_KING;
        Bitboard kingBB = bitboards[kingIndex];
        if (kingBB == 0) {
            // King not found—invalid state—assume safe.
            return false;
        }
        // Assume exactly one king exists; use a built-in to find its bit index.
        int kingSquare = __builtin_ctzll(kingBB); // index in [0, 63]
        int kingRow = kingSquare / 8;
        int kingCol = kingSquare % 8;

        // Temporarily switch side: generate all moves for the opponent.
        bool savedSide = white_to_move;
        white_to_move = !sideToCheck;
        std::vector<std::string> opponentMoves;
        generate_moves(opponentMoves);
        white_to_move = savedSide;

        // If any opponent move lands on the king's square, then the king is in check.
        for (const auto& mv : opponentMoves) {
            int toRow = mv[3] - '1';
            int toCol = mv[2] - 'a';
            if (toRow == kingRow && toCol == kingCol) {
                return true;
            }
        }

        return false;
    }


    char apply_move_with_capture(const std::string& move) {
        // Convert the move string into board coordinates.
        int from_col = move[0] - 'a';
        int from_row = move[1] - '1';
        int to_col   = move[2] - 'a';
        int to_row   = move[3] - '1';

        Bitboard from_mask = square_mask(from_row, from_col);
        Bitboard to_mask   = square_mask(to_row, to_col);

        // Determine if there is a captured piece at the destination.
        char captured = '.';
        int capturedIndex = -1;
        for (int i = 0; i < PIECE_NB; ++i) {
            if (bitboards[i] & to_mask) {
                captured = pieceSymbols[i];
                capturedIndex = i;
                break;
            }
        }

//        std::cout << "col: "<< from_col << "row: "  << from_row << "col: "  << to_col << "row: "  << to_row << std::endl;
//        std::cout << move << std::endl;

        // Identify the moving piece by checking the friendly bitboards.
        int movingPieceIndex = -1;
        if (white_to_move) {
            for (int i = WHITE_KING; i <= WHITE_PAWN; ++i) {
                if (bitboards[i] & from_mask) {
                    movingPieceIndex = i;
                    break;
                }
            }
        } else {
            for (int i = BLACK_KING; i <= BLACK_PAWN; ++i) {
                if (bitboards[i] & from_mask) {
                    movingPieceIndex = i;
                    break;
                }
            }
        }
        if (movingPieceIndex == -1)
            throw std::runtime_error("No moving piece found on from-square");

        // Check for promotion.
        char promotion_piece = '.';
        if (move.size() == 5) {
            promotion_piece = white_to_move ? toupper(move[4]) : tolower(move[4]);
        }

        // Remove the moving piece from its origin.
        bitboards[movingPieceIndex] &= ~from_mask;

        // If promotion, replace the pawn with the promoted piece.
        if (promotion_piece != '.') {
            int promotedIndex = -1;
            if (white_to_move) {
                switch (promotion_piece) {
                    case 'Q': promotedIndex = WHITE_QUEEN; break;
                    case 'R': promotedIndex = WHITE_ROOK; break;
                    case 'B': promotedIndex = WHITE_BISHOP; break;
                    case 'N': promotedIndex = WHITE_KNIGHT; break;
                    default: throw std::invalid_argument("Invalid promotion piece");
                }
            } else {
                switch (promotion_piece) {
                    case 'q': promotedIndex = BLACK_QUEEN; break;
                    case 'r': promotedIndex = BLACK_ROOK; break;
                    case 'b': promotedIndex = BLACK_BISHOP; break;
                    case 'n': promotedIndex = BLACK_KNIGHT; break;
                    default: throw std::invalid_argument("Invalid promotion piece");
                }
            }
            bitboards[promotedIndex] |= to_mask;
        } else {
            // Normal move: simply place the moving piece on the destination.
            bitboards[movingPieceIndex] |= to_mask;
        }

        // Remove any captured piece.
        if (capturedIndex != -1) {
            bitboards[capturedIndex] &= ~to_mask;
        }

        // Update game state.
        if (!white_to_move) fullmove_number++;
        white_to_move = !white_to_move;

        return captured;
    }


    void undo_move(const std::string& move, char captured) {
        int from_col = move[0] - 'a';
        int from_row = move[1] - '1';
        int to_col   = move[2] - 'a';
        int to_row   = move[3] - '1';

//        std::cout << "Inside undo move: "<< move << std::endl;

        Bitboard from_mask = square_mask(from_row, from_col);
        Bitboard to_mask   = square_mask(to_row, to_col);

        // After apply_move_with_capture, the moving piece is on the destination square.
        // Because we toggled white_to_move, the moving piece now belongs to the opposite side.
        int movedPieceIndex = -1;
        if (white_to_move) {
            // Now white is to move; the last move was by black.
            for (int i = BLACK_KING; i <= BLACK_PAWN; ++i) {
                if (bitboards[i] & to_mask) {
                    movedPieceIndex = i;
                    break;
                }
            }
        } else {
            for (int i = WHITE_KING; i <= WHITE_PAWN; ++i) {
                if (bitboards[i] & to_mask) {
                    movedPieceIndex = i;
                    break;
                }
            }
        }
        if (movedPieceIndex == -1)
            throw std::runtime_error("No moved piece found during undo");

        // Remove the piece from the destination square and restore it to the origin.
        bitboards[movedPieceIndex] &= ~to_mask;
        bitboards[movedPieceIndex] |= from_mask;

        // If a piece was captured, restore it on the destination square.
        if (captured != '.') {
            int capturedIndex = -1;
            if (isupper(captured)) {
                switch (captured) {
                    case 'K': capturedIndex = WHITE_KING; break;
                    case 'Q': capturedIndex = WHITE_QUEEN; break;
                    case 'B': capturedIndex = WHITE_BISHOP; break;
                    case 'N': capturedIndex = WHITE_KNIGHT; break;
                    case 'R': capturedIndex = WHITE_ROOK; break;
                    case 'P': capturedIndex = WHITE_PAWN; break;
                }
            } else {
                switch (captured) {
                    case 'k': capturedIndex = BLACK_KING; break;
                    case 'q': capturedIndex = BLACK_QUEEN; break;
                    case 'b': capturedIndex = BLACK_BISHOP; break;
                    case 'n': capturedIndex = BLACK_KNIGHT; break;
                    case 'r': capturedIndex = BLACK_ROOK; break;
                    case 'p': capturedIndex = BLACK_PAWN; break;
                }
            }
            if (capturedIndex == -1)
                throw std::runtime_error("Invalid captured piece symbol during undo");
            bitboards[capturedIndex] |= to_mask;
        }

        // Restore game state.
        white_to_move = !white_to_move;
        if (!white_to_move) {
            fullmove_number--;
        }
    }


    std::string make_move(const std::string& fen, const std::string& move) {
        set_position(fen);  // set_position should now initialize bitboards from a FEN string
        apply_move(move);   // apply_move (which likely calls apply_move_with_capture internally) works with bitboards
        return generate_fen();
    }

    std::pair<bool, int> is_game_over() {
        // 50-move rule check.
        if (halfmove_clock >= 25) {
            return {true, 2}; // 2 means draw by 50-move rule.
        }

        std::vector<std::string> moves = get_legal_moves();

//        std::cout  << " (" << moves.size() << " moves): ";
//        for (const auto& move : moves) {
//            std::cout << move << " ";
//        }
//        std::cout << std::endl;

        if (moves.empty()) {
                if (is_king_in_check(white_to_move))
                    return {true, 1}; // Checkmate.
                else
                    return {true, 0}; // Stalemate.
            }
            return {false, -1};
    }

    py::array_t<float> fen_to_tensor(const std::string &fen)
    {
        // 1. Parse the FEN to detect the side to move.
        //    For example: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        std::stringstream ss(fen);
        std::string board_part, side_to_move_str;
        ss >> board_part >> side_to_move_str;
        bool black_to_move = (side_to_move_str == "b");

        // 2. Set up your internal board from the FEN. This updates the bitboards.
        set_position(fen);

        // 3. (Instead of flipping a 2D board, we will flip coordinates when encoding if Black is to move.)

        // 4. Create a tensor vector of size (12 * 8 * 8) and initialize to 0.
        std::vector<float> tensor(12 * 8 * 8, 0.0f);

        // 5. For each piece type, iterate over its bitboard.
        for (int pieceType = 0; pieceType < PIECE_NB; ++pieceType) {
            Bitboard bb = bitboards[pieceType];
            while (bb) {
                // Get index of least-significant 1 bit.
                int sq = __builtin_ctzll(bb);
                int row = sq / 8;
                int col = sq % 8;
                // If Black is to move, flip the coordinates.
                if (black_to_move) {
                    row = 7 - row;
                    col = 7 - col;
                }

                // Determine the piece type index:
                //   king -> 0, queen -> 1, bishop -> 2, knight -> 3, rook -> 4, pawn -> 5
                char pieceChar = pieceSymbols[pieceType];
                char lower_piece = std::tolower(pieceChar);
                int piece_type_index = -1;
                if      (lower_piece == 'k') piece_type_index = 0;
                else if (lower_piece == 'q') piece_type_index = 1;
                else if (lower_piece == 'b') piece_type_index = 2;
                else if (lower_piece == 'n') piece_type_index = 3;
                else if (lower_piece == 'r') piece_type_index = 4;
                else if (lower_piece == 'p') piece_type_index = 5;
                if (piece_type_index < 0) {
                    bb &= bb - 1;
                    continue;
                }

                // Determine which side is “friendly” from the tensor’s perspective.
                // If Black is to move, then Black's pieces are friendly.
                bool is_white = std::isupper(pieceChar);
                bool piece_is_friendly = black_to_move ? (!is_white) : is_white;
                // Friendly pieces go in channels 0..5, enemy pieces in channels 6..11.
                int plane_index = piece_type_index + (piece_is_friendly ? 0 : 6);

                // Compute the flattened index for tensor: plane_index * 64 + (row * 8 + col)
                int index = plane_index * 64 + row * 8 + col;
                tensor[index] = 1.0f;

                // Clear the least-significant bit.
                bb &= bb - 1;
            }
        }

        // 6. Return the tensor as a py::array_t<float> with shape [12, 8, 8]
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

        std::vector<float> target(70 * 8 * 8, 0.0f); // 70 channels: 0-63 for sliding/knight, 64 for kingside, 65 for queenside, 66-69 for promotions
        int channel = -1;
        int from_row = 0, from_col = 0;

        if (move == "O-O") {
            channel = 64; // Kingside castling
        } else if (move == "O-O-O") {
            channel = 65; // Queenside castling
        } else {
            // Use corrected coordinate mapping: '1' -> row 0, '8' -> row 7.
            from_row = move[1] - '1';
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
                int to_row = move[3] - '1';
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
            from_row = move[1] - '1';  // Corrected mapping: '1' -> row 0

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
                int to_row = move[3] - '1';  // Corrected mapping: '1' -> row 0
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
            // Promotion moves
            char promotion_piece = "qrbn"[channel - 66];
            char from_file = 'a' + from_col;
            char from_rank = '1' + from_row;  // Corrected: row 0 -> '1', row 7 -> '8'
            // For promotion, if white is moving, pawn promotes on rank 8 (row 7); if black, on rank 1 (row 0).
            char to_file = from_file;
            char to_rank = white_to_move ? '8' : '1';
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

        char from_file_final = 'a' + from_col;
        char from_rank_final = '1' + from_row;  // Corrected mapping
        char to_file_final = 'a' + to_col;
        char to_rank_final = '1' + to_row;        // Corrected mapping
        return std::string({from_file_final, from_rank_final, to_file_final, to_rank_final});
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

    Bitboard bitboards[PIECE_NB] = {0};  // all initialized to 0
    bool white_to_move = true;
    int halfmove_clock = 0;
    int fullmove_number = 1;

    int ep_square = -1;
    // parsed from the FEN’s third field (e.g. “KQkq”)
    bool white_can_castle_kingside  = false;
    bool white_can_castle_queenside = false;
    bool black_can_castle_kingside  = false;
    bool black_can_castle_queenside = false;


    inline void set_piece(Bitboard& board, int rank, int file) {
        board |= square_mask(rank, file);
    }

    inline void clear_piece(Bitboard& board, int rank, int file) {
        board &= ~square_mask(rank, file);
    }

    inline bool is_set(Bitboard board, int rank, int file) {
        return board & square_mask(rank, file);
    }


    void parse_fen(const std::string& fen) {
        // Assume fen is in the standard format.
        std::istringstream iss(fen);
        std::string board_fen, turn, castling, en_passant, halfmove_s, fullmove_s;
        iss >> board_fen >> turn >> castling >> en_passant >> halfmove_s >> fullmove_s;

        white_to_move   = (turn == "w");
        halfmove_clock  = std::stoi(halfmove_s);
        fullmove_number = std::stoi(fullmove_s);

        // --- new ep_square parsing ---
        if (en_passant != "-") {
            // e.g. en_passant == "e3"
            int file = en_passant[0] - 'a';
            int rank = en_passant[1] - '1';   // FEN rank '1' -> row 0
            ep_square = rank * 8 + file;
        } else {
            ep_square = -1;
        }

        white_can_castle_kingside  = (castling.find('K') != std::string::npos);
        white_can_castle_queenside = (castling.find('Q') != std::string::npos);
        black_can_castle_kingside  = (castling.find('k') != std::string::npos);
        black_can_castle_queenside = (castling.find('q') != std::string::npos);

        // Clear bitboards
        for (int i = 0; i < PIECE_NB; ++i) {
            bitboards[i] = 0;
        }

        int rank = 0; // starting from top rank (a8)
        int file = 0;
        for (char c : board_fen) {
            if (c == '/') {
                ++rank;
                file = 0;
            } else if (std::isdigit(c)) {
                file += c - '0'; // skip empty squares
            } else {
                // Map FEN character to PieceIndex:
                PieceIndex idx;
                switch (c) {
                    case 'K': idx = WHITE_KING; break;
                    case 'Q': idx = WHITE_QUEEN; break;
                    case 'R': idx = WHITE_ROOK; break;
                    case 'B': idx = WHITE_BISHOP; break;
                    case 'N': idx = WHITE_KNIGHT; break;
                    case 'P': idx = WHITE_PAWN; break;
                    case 'k': idx = BLACK_KING; break;
                    case 'q': idx = BLACK_QUEEN; break;
                    case 'r': idx = BLACK_ROOK; break;
                    case 'b': idx = BLACK_BISHOP; break;
                    case 'n': idx = BLACK_KNIGHT; break;
                    case 'p': idx = BLACK_PAWN; break;
                    default: continue;  // unknown character
                }
                // Convert (rank, file) from FEN order to our mapping.
                // For example, if you decide that rank 0 is a1, you might do:
                int our_rank = 7 - rank; // if FEN rank 0 is a8, then a8 -> our_rank 7
                set_piece(bitboards[idx], our_rank, file);
                ++file;
            }
        }

        // Also update turn, halfmove, and fullmove as needed...
        white_to_move = (turn == "w");
        halfmove_clock = std::stoi(halfmove_s);
        fullmove_number = std::stoi(fullmove_s);
    }

    inline Bitboard square_mask(int rank, int file) const {
        return Bitboard(1) << (rank * 8 + file);
    }

    /// Returns true if side `attacker_is_white` attacks (r,c)
    bool is_square_attacked(int r, int c, bool attacker_is_white) {
        // temporarily switch side and generate all pseudo‐legal moves
        bool saved = white_to_move;
        white_to_move = attacker_is_white;
        std::vector<std::string> opp;
        generate_moves(opp);
        white_to_move = saved;

        for (auto &mv : opp) {
            int tr = mv[3] - '1';
            int tc = mv[2] - 'a';
            if (tr == r && tc == c)
                return true;
        }
        return false;
    }


    void handle_castling(const std::string& move) {
        // Determine the back rank for the side to move.
        // For bitboards, we assume white's back rank is 0 and black's is 7.
        int rank = white_to_move ? 0 : 7;

        if (move == "O-O") {
            // Kingside castling:
            // King: from e1/e8 (file 4) -> g1/g8 (file 6)
            // Rook: from h1/h8 (file 7) -> f1/f8 (file 5)
            if (white_to_move) {
                // Update white king:
                bitboards[WHITE_KING] &= ~square_mask(rank, 4);  // Remove king from e1
                bitboards[WHITE_KING] |= square_mask(rank, 6);   // Place king on g1

                // Update white rook:
                bitboards[WHITE_ROOK] &= ~square_mask(rank, 7);   // Remove rook from h1
                bitboards[WHITE_ROOK] |= square_mask(rank, 5);    // Place rook on f1
            } else {
                // Update black king:
                bitboards[BLACK_KING] &= ~square_mask(rank, 4);  // Remove king from e8
                bitboards[BLACK_KING] |= square_mask(rank, 6);   // Place king on g8

                // Update black rook:
                bitboards[BLACK_ROOK] &= ~square_mask(rank, 7);   // Remove rook from h8
                bitboards[BLACK_ROOK] |= square_mask(rank, 5);    // Place rook on f8
            }
        } else if (move == "O-O-O") {
            // Queenside castling:
            // King: from e1/e8 (file 4) -> c1/c8 (file 2)
            // Rook: from a1/a8 (file 0) -> d1/d8 (file 3)
            if (white_to_move) {
                // Update white king:
                bitboards[WHITE_KING] &= ~square_mask(rank, 4);  // Remove king from e1
                bitboards[WHITE_KING] |= square_mask(rank, 2);   // Place king on c1

                // Update white rook:
                bitboards[WHITE_ROOK] &= ~square_mask(rank, 0);   // Remove rook from a1
                bitboards[WHITE_ROOK] |= square_mask(rank, 3);    // Place rook on d1
            } else {
                // Update black king:
                bitboards[BLACK_KING] &= ~square_mask(rank, 4);  // Remove king from e8
                bitboards[BLACK_KING] |= square_mask(rank, 2);   // Place king on c8

                // Update black rook:
                bitboards[BLACK_ROOK] &= ~square_mask(rank, 0);   // Remove rook from a8
                bitboards[BLACK_ROOK] |= square_mask(rank, 3);    // Place rook on d8
            }
        }

        // Update move counters and toggle side to move.
        if (!white_to_move)
            fullmove_number++;
        white_to_move = !white_to_move;
    }

    void generate_moves(std::vector<std::string>& moves) {
//        std::cout << "\n=== Generating moves for: " << (white_to_move ? "White" : "Black") << " ===\n";

        int move_count_before = moves.size();

        if (white_to_move) {
            for (int pieceType = WHITE_KING; pieceType <= WHITE_PAWN; ++pieceType) {
                Bitboard bb = bitboards[pieceType];
//                std::cout << pieceSymbols[pieceType] << " bitboard: " << std::hex << bb << std::dec << std::endl;

                while (bb) {
                    int sq = __builtin_ctzll(bb);
                    int row = sq / 8;
                    int col = sq % 8;

                    switch (pieceType) {
                        case WHITE_PAWN:   generate_pawn_moves(row, col, moves); break;
                        case WHITE_KNIGHT: generate_knight_moves(row, col, moves); break;
                        case WHITE_BISHOP: generate_bishop_moves(row, col, moves); break;
                        case WHITE_ROOK:   generate_rook_moves(row, col, moves); break;
                        case WHITE_QUEEN:  generate_queen_moves(row, col, moves); break;
                        case WHITE_KING:   generate_king_moves(row, col, moves); break;
                    }
                    bb &= bb - 1;  // Remove the lowest set bit
                }
            }
        } else {
            for (int pieceType = BLACK_KING; pieceType <= BLACK_PAWN; ++pieceType) {
                Bitboard bb = bitboards[pieceType];
//                std::cout << pieceSymbols[pieceType] << " bitboard: " << std::hex << bb << std::dec << std::endl;

                while (bb) {
                    int sq = __builtin_ctzll(bb);
                    int row = sq / 8;
                    int col = sq % 8;

                    switch (pieceType) {
                        case BLACK_PAWN:   generate_pawn_moves(row, col, moves); break;
                        case BLACK_KNIGHT: generate_knight_moves(row, col, moves); break;
                        case BLACK_BISHOP: generate_bishop_moves(row, col, moves); break;
                        case BLACK_ROOK:   generate_rook_moves(row, col, moves); break;
                        case BLACK_QUEEN:  generate_queen_moves(row, col, moves); break;
                        case BLACK_KING:   generate_king_moves(row, col, moves); break;
                    }
                    bb &= bb - 1;
                }
            }
        }

        int move_count_after = moves.size();
//        std::cout << "Total moves generated: " << (move_count_after - move_count_before) << "\n";
    }


    void generate_pawn_moves(int row, int col, std::vector<std::string>& moves) {
        // Direction, starting rank and promotion rank depend on side to move
        int direction      = white_to_move ? 1 : -1;
        int start_row      = white_to_move ? 1 : 6;
        int promotion_row  = white_to_move ? 7 : 0;

        // Build occupancy and enemy occupancy bitboards
        Bitboard occupancy = 0, enemyOccupancy = 0;
        for (int i = 0; i < PIECE_NB; ++i) {
            occupancy |= bitboards[i];
            // Pieces with index >= BLACK_KING are black
            if (white_to_move ? i >= BLACK_KING : i < WHITE_PAWN)
                enemyOccupancy |= bitboards[i];
        }

        // Ensure there's a pawn of the correct color on (row,col)
        Bitboard from_bb = square_mask(row, col);
        Bitboard pawn_bb = white_to_move ? bitboards[WHITE_PAWN] : bitboards[BLACK_PAWN];
        if (!(pawn_bb & from_bb))
            return;

        // Single forward move
        int nr = row + direction;
        int nc = col;
        if (nr >= 0 && nr < 8) {
            Bitboard to_bb = square_mask(nr, nc);
            // Empty square
            if (!(occupancy & to_bb)) {
                // Promotion
                if (nr == promotion_row) {
                    for (char promo : {'q', 'r', 'b', 'n'}) {
                        std::string mv;
                        mv += char('a' + col);
                        mv += char('1' + row);
                        mv += char('a' + nc);
                        mv += char('1' + nr);
                        mv += promo;
                        moves.push_back(mv);
                    }
                } else {
                    // Non-promotion single push
                    std::string mv = {char('a' + col), char('1' + row), char('a' + nc), char('1' + nr)};
                    moves.push_back(mv);
                    // Double push from starting rank
                    if (row == start_row) {
                        int nn = nr + direction;
                        if (nn >= 0 && nn < 8) {
                            Bitboard dbl_bb = square_mask(nn, nc);
                            if (!(occupancy & dbl_bb)) {
                                std::string mv2 = {char('a' + col), char('1' + row), char('a' + nc), char('1' + nn)};
                                moves.push_back(mv2);
                            }
                        }
                    }
                }
            }
        }

        // Captures and promotion-captures
        for (int dc : {-1, +1}) {
            int cr = row + direction;
            int cc = col + dc;
            if (cr >= 0 && cr < 8 && cc >= 0 && cc < 8) {
                Bitboard cap_bb = square_mask(cr, cc);
                // Normal capture
                if (enemyOccupancy & cap_bb) {
                    if (cr == promotion_row) {
                        // Promotion capture
                        for (char promo : {'q', 'r', 'b', 'n'}) {
                            std::string mv;
                            mv += char('a' + col);
                            mv += char('1' + row);
                            mv += char('a' + cc);
                            mv += char('1' + cr);
                            mv += promo;
                            moves.push_back(mv);
                        }
                    } else {
                        // Non-promotion capture
                        std::string mv = {char('a' + col), char('1' + row), char('a' + cc), char('1' + cr)};
                        moves.push_back(mv);
                    }
                }
                // En-passant capture (if ep_square is set correctly)
                else if (ep_square != -1 && cr * 8 + cc == ep_square) {
                    std::string mv = {
                        char('a' + col),
                        char('1' + row),
                        char('a' + cc),
                        char('1' + cr)
                    };
                    moves.push_back(mv);
                }
            }
        }
    }

    void generate_knight_moves(int row, int col, std::vector<std::string>& moves) {
        static const int offsets[8][2] = {
            {2, 1}, {2, -1}, {-2, 1}, {-2, -1},
            {1, 2}, {1, -2}, {-1, 2}, {-1, -2}
        };

        // Compute the occupancy for friendly pieces.
        Bitboard friendlyOccupancy = 0;
        if (white_to_move) {
            for (int i = WHITE_KING; i <= WHITE_PAWN; ++i)
                friendlyOccupancy |= bitboards[i];
        } else {
            for (int i = BLACK_KING; i <= BLACK_PAWN; ++i)
                friendlyOccupancy |= bitboards[i];
        }

        // Try each knight offset.
        for (int i = 0; i < 8; ++i) {
            int new_row = row + offsets[i][0];
            int new_col = col + offsets[i][1];
            if (new_row < 0 || new_row >= 8 || new_col < 0 || new_col >= 8)
                continue;

            Bitboard destSquare = square_mask(new_row, new_col);
            if (friendlyOccupancy & destSquare)
                continue;  // Skip if a friendly piece occupies this square.

            add_move(row, col, new_row, new_col, moves);
        }
    }

    // Generate bishop moves using sliding directions for diagonals.
    void generate_bishop_moves(int row, int col, std::vector<std::string>& moves) {
        static const int directions[4][2] = { {1, 1}, {1, -1}, {-1, 1}, {-1, -1} };
        generate_sliding_moves(row, col, directions, 4, moves);
    }

    // Generate rook moves using sliding directions for ranks and files.
    void generate_rook_moves(int row, int col, std::vector<std::string>& moves) {
        static const int directions[4][2] = { {1, 0}, {-1, 0}, {0, 1}, {0, -1} };
        generate_sliding_moves(row, col, directions, 4, moves);
    }

    // Generate queen moves by combining rook and bishop directions.
    void generate_queen_moves(int row, int col, std::vector<std::string>& moves) {
        static const int directions[8][2] = {
            {1, 1}, {1, -1}, {-1, 1}, {-1, -1},
            {1, 0}, {-1, 0}, {0, 1}, {0, -1}
        };
        generate_sliding_moves(row, col, directions, 8, moves);
    }

    // Generate king moves (one square in any direction).
    void generate_king_moves(int row, int col, std::vector<std::string>& moves) {
        static const int offsets[8][2] = {
            {1, 1}, {1, -1}, {-1, 1}, {-1, -1},
            {1, 0}, {-1, 0}, {0, 1}, {0, -1}
        };

        Bitboard friendlyOccupancy = 0;
        if (white_to_move) {
            for (int i = WHITE_KING; i <= WHITE_PAWN; ++i)
                friendlyOccupancy |= bitboards[i];
        } else {
            for (int i = BLACK_KING; i <= BLACK_PAWN; ++i)
                friendlyOccupancy |= bitboards[i];
        }

        for (int i = 0; i < 8; ++i) {
            int new_row = row + offsets[i][0];
            int new_col = col + offsets[i][1];
            if (new_row < 0 || new_row >= 8 || new_col < 0 || new_col >= 8)
                continue;

            Bitboard sq = square_mask(new_row, new_col);
            if (!(friendlyOccupancy & sq)) {  // Only move if the square is not occupied by a friendly piece
                add_move(row, col, new_row, new_col, moves);
            }
        }
    }


    // Generate sliding moves (used for bishops, rooks, and queens).
    void generate_sliding_moves(int row, int col, const int directions[][2], int count, std::vector<std::string>& moves) {
        // Compute full occupancy: union of all pieces.
        Bitboard occupancy = 0;
        for (int i = 0; i < PIECE_NB; ++i)
            occupancy |= bitboards[i];

        // Compute occupancy for friendly pieces.
        Bitboard friendlyOccupancy = 0;
        if (white_to_move) {
            for (int i = WHITE_KING; i <= WHITE_PAWN; ++i)
                friendlyOccupancy |= bitboards[i];
        } else {
            for (int i = BLACK_KING; i <= BLACK_PAWN; ++i)
                friendlyOccupancy |= bitboards[i];
        }

        // For each sliding direction...
        for (int i = 0; i < count; ++i) {
            int new_row = row;
            int new_col = col;
            // Move step by step in the current direction.
            while (true) {
                new_row += directions[i][0];
                new_col += directions[i][1];
                if (new_row < 0 || new_row >= 8 || new_col < 0 || new_col >= 8)
                    break;  // Off the board.

                Bitboard sq = square_mask(new_row, new_col);
                if (friendlyOccupancy & sq)
                    break;  // Blocked by a friendly piece.

                // Add the move (even if the square is occupied by an enemy, add it then break).
                add_move(row, col, new_row, new_col, moves);

                if (occupancy & sq)
                    break;  // Stop sliding if any piece is present.
            }
        }
    }


    void apply_move(const std::string& move) {
        // If castling, handle that separately.
        if (move == "O-O" || move == "O-O-O") {
            handle_castling(move);
            return;
        }

        // Convert move string into board coordinates.
        // Using the same conversion as before:
        // e.g., for move "e2e4": from: ('e','2'), to: ('e','4')
        int from_col = move[0] - 'a';
        int from_row = move[1] - '1';
        int to_col   = move[2] - 'a';
        int to_row   = move[3] - '1';

        // Determine promotion piece, if any.
        char promotion_piece = '.';
        if (move.size() == 5) {
            promotion_piece = white_to_move ? toupper(move[4]) : tolower(move[4]);
        }

        Bitboard from_mask = square_mask(from_row, from_col);
        Bitboard to_mask   = square_mask(to_row, to_col);

        // Find the moving piece among the friendly pieces.
        int moving_piece_index = -1;
        if (white_to_move) {
            for (int i = WHITE_KING; i <= WHITE_PAWN; ++i) {
                if (bitboards[i] & from_mask) {
                    moving_piece_index = i;
                    break;
                }
            }
        } else {
            for (int i = BLACK_KING; i <= BLACK_PAWN; ++i) {
                if (bitboards[i] & from_mask) {
                    moving_piece_index = i;
                    break;
                }
            }
        }

        if (moving_piece_index == -1) {
            throw std::runtime_error("No piece found on from-square");
        }

        // Determine if a capture occurs by checking if any piece occupies the destination.
        bool is_capture = false;
        for (int i = 0; i < PIECE_NB; ++i) {
            if (bitboards[i] & to_mask) {
                is_capture = true;
                // Remove the captured piece.
                bitboards[i] &= ~to_mask;
                break;
            }
        }

        // Check if the moving piece is a pawn.
        bool is_pawn_move = false;
        if ((white_to_move && moving_piece_index == WHITE_PAWN) ||
            (!white_to_move && moving_piece_index == BLACK_PAWN))
        {
            is_pawn_move = true;
        }

        // Remove the moving piece from its source square.
        bitboards[moving_piece_index] &= ~from_mask;

        // If a promotion is specified, replace the pawn with the promoted piece.
        if (promotion_piece != '.') {
            int promoted_index = -1;
            if (white_to_move) {
                switch (promotion_piece) {
                    case 'Q': promoted_index = WHITE_QUEEN; break;
                    case 'R': promoted_index = WHITE_ROOK; break;
                    case 'B': promoted_index = WHITE_BISHOP; break;
                    case 'N': promoted_index = WHITE_KNIGHT; break;
                    default: throw std::invalid_argument("Invalid promotion piece");
                }
            } else {
                switch (promotion_piece) {
                    case 'q': promoted_index = BLACK_QUEEN; break;
                    case 'r': promoted_index = BLACK_ROOK; break;
                    case 'b': promoted_index = BLACK_BISHOP; break;
                    case 'n': promoted_index = BLACK_KNIGHT; break;
                    default: throw std::invalid_argument("Invalid promotion piece");
                }
            }
            // Place the promoted piece on the destination square.
            bitboards[promoted_index] |= to_mask;
        } else {
            // Normal move: place the moving piece on the destination.
            bitboards[moving_piece_index] |= to_mask;
        }

        // Update halfmove clock.
        if (is_capture || is_pawn_move) {
            halfmove_clock = 0;
        } else {
            halfmove_clock++;
        }

        // Update fullmove number and toggle side to move.
        if (!white_to_move) {
            fullmove_number++;
        }
        white_to_move = !white_to_move;
    }

    std::string generate_fen() const {
        std::ostringstream fen;
        // FEN requires board rows from rank 8 down to rank 1.
        // Our mapping: row 0 = rank 1, row 7 = rank 8.
        for (int row = 7; row >= 0; --row) {
            int empty_count = 0;
            for (int col = 0; col < 8; ++col) {
                Bitboard sq = square_mask(row, col);
                char pieceChar = '.';
                // Mapping from bitboard index to piece symbol.
                static const char pieceSymbols[PIECE_NB] = {
                    'K', 'Q', 'B', 'N', 'R', 'P',  // white pieces
                    'k', 'q', 'b', 'n', 'r', 'p'   // black pieces
                };
                bool found = false;
                for (int i = 0; i < PIECE_NB; ++i) {
                    if (bitboards[i] & sq) {
                        pieceChar = pieceSymbols[i];
                        found = true;
                        break;
                    }
                }
                if (pieceChar == '.') {
                    empty_count++;
                } else {
                    if (empty_count > 0) {
                        fen << empty_count;
                        empty_count = 0;
                    }
                    fen << pieceChar;
                }
            }
            if (empty_count > 0)
                fen << empty_count;
            if (row > 0)
                fen << '/';
        }
        // Side to move.
        fen << ' ' << (white_to_move ? 'w' : 'b');
        // Castling rights – ignored, so output '-'
        fen << " -";
        // En passant square – ignored, so output '-'
        fen << " -";
        // Halfmove clock.
        fen << " " << halfmove_clock;
        // Fullmove number.
        fen << " " << fullmove_number;
        return fen.str();
    }

    bool is_within_bounds(int row, int col) const {
        return row >= 0 && row < 8 && col >= 0 && col < 8;
    }


    bool is_empty(int row, int col) const {
        if (!is_within_bounds(row, col))
            return false;
        Bitboard sq = square_mask(row, col);
        for (int i = 0; i < PIECE_NB; ++i) {
            if (bitboards[i] & sq)
                return false;
        }
        return true;
    }

    bool is_friendly(int row, int col) const {
        if (!is_within_bounds(row, col))
            return false;
        Bitboard sq = square_mask(row, col);
        if (white_to_move) {
            for (int i = WHITE_KING; i <= WHITE_PAWN; ++i)
                if (bitboards[i] & sq)
                    return true;
        } else {
            for (int i = BLACK_KING; i <= BLACK_PAWN; ++i)
                if (bitboards[i] & sq)
                    return true;
        }
        return false;
    }


    bool is_enemy(int row, int col) const {
        if (!is_within_bounds(row, col))
            return false;
        Bitboard sq = square_mask(row, col);
        if (white_to_move) {
            // Enemy pieces are black.
            for (int i = BLACK_KING; i <= BLACK_PAWN; ++i)
                if (bitboards[i] & sq)
                    return true;
        } else {
            // Enemy pieces are white.
            for (int i = WHITE_KING; i <= WHITE_PAWN; ++i)
                if (bitboards[i] & sq)
                    return true;
        }
        return false;
    }



    void add_move(int from_row, int from_col, int to_row, int to_col, std::vector<std::string>& moves, char promotion = '\0') {
        char from_file = 'a' + from_col; // Column to file
        char from_rank = '1' + from_row; // Row to rank
        char to_file = 'a' + to_col;
        char to_rank = '1' + to_row;

//        std::cout << from_col << from_row << to_col << to_row << std::endl;
//        std::cout << from_file << from_rank << to_file << to_rank << std::endl;
        std::string move = std::string() + from_file + from_rank + to_file + to_rank;
        if (promotion) {
            move += promotion;
        }
        moves.push_back(move);
    }

public:
    // Add this method to expose the board as a 2D vector
    std::vector<std::vector<char>> get_board() const {
        // Initialize an 8x8 board filled with empty squares.
        std::vector<std::vector<char>> python_board(8, std::vector<char>(8, '.'));

        // Define a mapping from our piece index to the character symbol.
        // Adjust these symbols if needed.
        static const char pieceSymbols[PIECE_NB] = {
            'K', 'Q', 'B', 'N', 'R', 'P',  // white pieces
            'k', 'q', 'b', 'n', 'r', 'p'   // black pieces
        };

        // Loop over all squares.
        for (int row = 0; row < 8; ++row) {
            for (int col = 0; col < 8; ++col) {
                Bitboard sq = square_mask(row, col);
                // Check each piece's bitboard.
                for (int i = 0; i < PIECE_NB; ++i) {
                    if (bitboards[i] & sq) {
                        python_board[row][col] = pieceSymbols[i];
                        break;  // Only one piece should be on any square.
                    }
                }
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
