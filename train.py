import chess
import chess.svg
import chess_moves
from tree.tree import GameTree, Node
import time
import torch

from neural_nets.conv_net import ChessNet

if __name__ == '__main__':

    chess_net = ChessNet(input_size=[12, 8, 8], output_size=[70, 8, 8], num_repeats=16)

    # tree = GameTree(chess_moves.ChessEngine, num_threads=1, neural_net=chess_net, training=True)

    tree = GameTree(chess_moves.ChessEngine, num_threads=1, neural_net=chess_net, training=True, multiprocess=True, num_evalators=2)

    sims = 1000
    max_length = 400

    for game in range(1000):

        move_count = 0
        main_board = chess.Board()

        tree.reset()

        node = tree.root
        winner = None

        # # Load new random chess games in
        # tree.memory.load_data()

        while not main_board.is_game_over():

            start_time = time.time()
            tree.parallel_search(current_node=node,number_of_expansions=sims)
            end_time = time.time()

            print(f"Time with parallel search {end_time-start_time:.3f}")

            #TODO: sort out now visiting

            if move_count < 30:
                tau = 1.
            else:
                tau = 0.1

            node, move = tree.root.select_new_root_node(tau=tau)

            chess_move = chess.Move.from_uci(move)
            main_board.push(chess_move)

            print(main_board)

            tree.root = node

            # Clear references to the tree above
            tree.root.parent_edge = None

            if main_board.is_checkmate():
                print("Checkmate!")
                winner = main_board.fen().split()[1]
            elif main_board.is_stalemate():
                print("Stalemate!")
            elif main_board.is_insufficient_material():
                print("Insufficient material to checkmate!")
            elif move_count >= max_length:
                print("Maximum move length")
            elif main_board.is_check():
                print("King is in check!")

        if winner == 'w':
            white_win = True
        elif winner == 'b':
            white_win = False
        else:
            white_win = None

        if not tree.multiprocess:
            tree.end_game(white_win)

        # torch.save(chess_net.state_dict(), f"networks/model_{game}.pt")
        # print(f"New best model saved with rolling avg loss: {white_win:.4f}")

