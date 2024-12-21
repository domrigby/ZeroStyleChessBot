#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define BOARD_SIZE 64

// Piece constants
#define EMPTY '.'
#define PAWN 'P'
#define KNIGHT 'N'
#define BISHOP 'B'
#define ROOK 'R'
#define QUEEN 'Q'
#define KING 'K'

// Side to move constants
#define WHITE 0
#define BLACK 1

// Algebraic notation for board squares
const char *square_to_algebraic(int square) {
    static char notation[3];
    notation[0] = 'a' + (square % 8);
    notation[1] = '8' - (square / 8);
    notation[2] = '\0';
    return notation;
}

typedef struct {
    char board[BOARD_SIZE];
    int side_to_move; // WHITE or BLACK
} ChessState;

// Converts a FEN string to a ChessState
int load_fen(const char *fen, ChessState *state) {
    memset(state->board, EMPTY, BOARD_SIZE);
    state->side_to_move = WHITE;

    int square = 0;
    const char *ptr = fen;

    // Parse board state
    while (*ptr && *ptr != ' ') {
        if (isdigit(*ptr)) {
            square += *ptr - '0';
        } else if (*ptr == '/') {
            continue;
        } else {
            state->board[square++] = *ptr;
        }
        ptr++;
    }

    if (square != BOARD_SIZE) return -1; // Invalid board size

    // Skip space and parse side to move
    if (*ptr == ' ') ptr++;
    state->side_to_move = (*ptr == 'w') ? WHITE : BLACK;

    return 0;
}

// Generates pseudo-legal moves for pawns and returns them in algebraic notation
void generate_legal_moves(const ChessState *state, char *output, size_t output_size) {
    char moves[1024] = "";
    for (int i = 0; i < BOARD_SIZE; i++) {
        if (state->board[i] == (state->side_to_move == WHITE ? PAWN : tolower(PAWN))) {
            int forward = state->side_to_move == WHITE ? -8 : 8;
            int target = i + forward;
            if (target >= 0 && target < BOARD_SIZE && state->board[target] == EMPTY) {
                char move[6];
                snprintf(move, sizeof(move), "%s%s ", square_to_algebraic(i), square_to_algebraic(target));
                strcat(moves, move);
            }
        }
    }
    strncpy(output, moves, output_size - 1);
    output[output_size - 1] = '\0';
}

// Prints the board
void print_board(const ChessState *state) {
    for (int rank = 0; rank < 8; rank++) {
        for (int file = 0; file < 8; file++) {
            printf("%c ", state->board[rank * 8 + file]);
        }
        printf("\n");
    }
    printf("Side to move: %s\n", state->side_to_move == WHITE ? "White" : "Black");
}

int main() {
    const char *fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    ChessState state;

    if (load_fen(fen, &state) != 0) {
        printf("Invalid FEN string!\n");
        return 1;
    }

    print_board(&state);

    char moves[1024];
    generate_legal_moves(&state, moves, sizeof(moves));
    printf("Legal moves: %s\n", moves);

    return 0;
}
