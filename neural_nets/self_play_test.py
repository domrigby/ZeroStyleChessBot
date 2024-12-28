import chess
import chess.engine
import numpy as np
import torch
import chess_moves
from cpp_chess_env.chess_funcs import get_move_from_tensor, board_to_tensor

from neural_nets.conv_net import ChessNet

# Initialize chess engine and neural network
chess_engine = chess_moves.ChessEngine()

chess_net = ChessNet(input_size=[12, 8, 8], output_size=[66, 8, 8], num_repeats=16)
chess_net.load_network(r"/home/dom/Code/chess_bot/neural_nets/session/best_model_120.pt")
chess_net.eval()

# Initialize the chess board
board = chess.Board()

counter = 0

# Play a game where the network plays itself
while not board.is_game_over():

    counter += 1
    print(f"Move number: {counter}")

    # Convert the board to tensor
    board_tensor = board_to_tensor(board)

    legal_moves = board.legal_moves
    legal_move_mask = torch.zeros([66, 8, 8], dtype=torch.int)
    for legal_move in legal_moves:
        # Need to add pawn promotion
        if len(str(legal_move)) < 5:
            legal_move_mask[chess_engine.move_to_target_indices(str(legal_move))] = 1

    # Predict the move
    with torch.no_grad():
        output_tensor = chess_net(torch.tensor(board_tensor, dtype=torch.float32, device='cuda').unsqueeze(0),
                                  legal_move_mask)
    move_str = get_move_from_tensor(output_tensor[1].cpu().numpy().squeeze())

    try:
        # Apply the move to the board
        move = chess.Move.from_uci(move_str)
        if move in board.legal_moves:
            print(f"Move: {move}")
            print(f"\tIs capture {board.is_capture(move)}")
            board.push(move)
            print(board)
            print('\n')
        else:
            print(f"Illegal move predicted: {move_str}")
            break
    except Exception as e:
        print(f"Error processing move {move_str}: {e}")
        break

# Print the final result
print("Game Over")
print(f"Result: {board.result()}\n")
print(f"\t is stalemate: {board.is_stalemate()}\n")
