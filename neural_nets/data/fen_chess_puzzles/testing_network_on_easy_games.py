import torch
from neural_nets.conv_net import ChessNet
import chess_moves

# Initialize the network
chess_net = ChessNet(input_size=[12, 8, 8], output_size=[70, 8, 8], num_repeats=32, num_filters=64, init_lr=0.001)
chess_net.load_network("/home/dom/Code/chess_bot/networks/best_model_293.pt")
chess_net.eval()

engine = chess_moves.ChessEngine()


# Case 1 :
fen = "4r3/1k6/pp3r2/1b2P2p/3R1p2/P1R2P2/1P4PP/6K1 w - - 0 35"
input_tens = engine.fen_to_tensor(fen)

print("Current player 2 moves from victory:")
print("\t Value =", chess_net(torch.tensor(input_tens).unsqueeze(0))[0].item())

fen = "4r3/1k6/pp3r2/1b2P2p/3R1p2/P1R2P2/1P4PP/6K1 b - - 0 35"
input_tens = engine.fen_to_tensor(fen)
print("\t Value =", chess_net(torch.tensor(input_tens).unsqueeze(0))[0].item())

# Case 2: Out of dist, definite black win
fen = "K7/6r1/5r2/8/8/8/8/7k w - - 0 1"
print("Simple definite loss")
input_tens = engine.fen_to_tensor(fen)
print("\t White value =", chess_net(torch.tensor(input_tens).unsqueeze(0))[0].item())
fen = "K7/6r1/5r2/8/8/8/8/7k b - - 0 1"
input_tens = engine.fen_to_tensor(fen)
print("\t Black value =", chess_net(torch.tensor(input_tens).unsqueeze(0))[0].item())

# Case 3: Start state
fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
print("Start state:")
input_tens = engine.fen_to_tensor(fen)
print("\t White value =", chess_net(torch.tensor(input_tens).unsqueeze(0))[0].item())
fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1"
input_tens = engine.fen_to_tensor(fen)
print("\t Black value =", chess_net(torch.tensor(input_tens).unsqueeze(0))[0].item())

fen = "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1"
print("One move in:")
input_tens = engine.fen_to_tensor(fen)
print("\t White value =", chess_net(torch.tensor(input_tens).unsqueeze(0))[0].item())
fen = "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1"
input_tens = engine.fen_to_tensor(fen)
print("\t Black value =", chess_net(torch.tensor(input_tens).unsqueeze(0))[0].item())
