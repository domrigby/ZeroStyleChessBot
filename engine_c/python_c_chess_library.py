import ctypes

# Load the shared library
chess_engine = ctypes.CDLL('./chess_engine.so')  # Replace with the correct path to your .so or .dll file

# Define the return and argument types for the function
chess_engine.generate_legal_moves_py.restype = ctypes.POINTER(ctypes.c_char_p)
chess_engine.generate_legal_moves_py.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)]

# Define the function to use in Python
def generate_legal_moves(fen):
    move_count = ctypes.c_int()
    moves_ptr = chess_engine.generate_legal_moves_py(ctypes.c_char_p(fen.encode('utf-8')), ctypes.byref(move_count))
    move_count_value = move_count.value

    # Extract the moves from the pointer
    moves = []
    for i in range(move_count_value):
        move = ctypes.cast(moves_ptr[i], ctypes.c_char_p).value.decode('utf-8')
        moves.append(move)

    # Free the allocated memory in the C code
    for i in range(move_count_value):
        chess_engine.free(moves_ptr[i])
    chess_engine.free(moves_ptr)

    return moves

# Example usage
fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
legal_moves = generate_legal_moves(fen)
print(f"Legal moves for FEN '{fen}':")
print(legal_moves)
