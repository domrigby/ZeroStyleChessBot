/* Include necessary headers */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <ctype.h>

/* Define constants for the board */
#define BOARD_SIZE 64
#define MOVE_BUFFER_SIZE 16

/* Helper function to parse the FEN and initialize the board */
void parse_fen(const char* fen, char board[8][8]) {
    int row = 0, col = 0;

    for (int i = 0; fen[i] != ' ' && fen[i] != '\0'; i++) {
        char c = fen[i];

        if (c == '/') {
            row++;
            col = 0;
        } else if (c >= '1' && c <= '8') {
            int empty_squares = c - '0';
            for (int j = 0; j < empty_squares; j++) {
                board[row][col++] = '.'; // Use '.' to represent empty squares
            }
        } else {
            board[row][col++] = c;
        }
    }
}

/* Function to generate pseudo-legal moves for pawns */
void generate_pawn_moves(char board[8][8], int row, int col, char** moves, int* move_count, char color) {
    int direction = (color == 'w') ? -1 : 1;

    if (row + direction >= 0 && row + direction < 8 && board[row + direction][col] == '.') {
        char move[MOVE_BUFFER_SIZE];
        snprintf(move, MOVE_BUFFER_SIZE, "%c%d%c%d", col + 'a', 8 - row, col + 'a', 8 - (row + direction));
        moves[*move_count] = strdup(move);
        (*move_count)++;
    }
}

/* Main function to generate all legal moves from a FEN string */
char** generate_legal_moves(const char* fen, int* move_count) {
    char board[8][8];
    memset(board, '.', sizeof(board));
    parse_fen(fen, board);

    char** moves = malloc(500 * sizeof(char*)); // Allocate memory for moves
    if (moves == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return NULL;
    }

    *move_count = 0;
    for (int row = 0; row < 8; row++) {
        for (int col = 0; col < 8; col++) {
            char piece = board[row][col];
            if (piece == 'P') {
                generate_pawn_moves(board, row, col, moves, move_count, 'w');
            } else if (piece == 'p') {
                generate_pawn_moves(board, row, col, moves, move_count, 'b');
            }
        }
    }

    return moves;
}

/* Freeing the allocated moves */
void free_moves(char** moves, int move_count) {
    if (moves == NULL) {
        return;
    }
    for (int i = 0; i < move_count; ++i) {
        if (moves[i] != NULL) {
            free(moves[i]);
        }
    }
    free(moves);
}

/* Python-callable wrapper */
__attribute__((visibility("default")))
__attribute__((used))
char** generate_legal_moves_py(const char* fen, int* move_count) {
    return generate_legal_moves(fen, move_count);
}

/* Free individual string pointers */
__attribute__((visibility("default")))
__attribute__((used))
void free_allocated_memory(void* ptr) {
    if (ptr != NULL) {
        free(ptr);
    }
}
