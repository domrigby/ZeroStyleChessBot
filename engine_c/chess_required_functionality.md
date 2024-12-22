# Python-Chess Used Functionality

1. Takes a fen state and return the allowed moves
2. Get FEN state and perform a move. Get the whether the game is over or if a piece has been captured

## C - functions
1. fen_state_to_moves(fen_state) -> List of allowed moves in chess format (e.g. f4f5)
2. do_move_in_fen_state(fen_state, move) -> [New fen state, done, piece captured_bool]