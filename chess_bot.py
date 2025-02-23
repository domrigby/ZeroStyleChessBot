import multiprocessing as mp
if __name__ == "__main__":
    try:
       mp.set_start_method('spawn', force=True)
       print("spawned")
    except RuntimeError:
        print("Failed to spawn")
        pass

import chess
import chess.svg
import chess_moves
from util.queues import create_agents
from util.test_fen_strings import FenTests



from neural_nets.conv_net import ChessNet

if __name__ == '__main__':

    chess_net = ChessNet(input_size=[12, 8, 8], output_size=[70, 8, 8], num_repeats=32)
    # chess_net.load_network(r"/home/dom/Code/chess_bot/networks/RL_tuned_37000.pt")
    # chess_net.load_network(r"/home/dom/Code/chess_bot/networks/best_model2_23.pt")
    chess_net.eval()

    start_state = FenTests.MATE_IN_TW0

    tree, evaluator, _, _ = create_agents(1, 1, 0, chess_net, training=False,
                                          start_state=start_state)

    sims = 10000

    tree = tree[0]

    node = tree.root

    evaluator[0].start()

    main_board = chess.Board()
    main_board.set_fen(start_state)

    import time
    ctr = 0

    tree.reset()

    while not main_board.is_game_over():
        start_time = time.time()
        tree.parallel_search(current_node=node,number_of_expansions=sims)
        end_time = time.time()

        print(f"Time with parallel search {end_time-start_time:.3f}")

        #TODO: sort out now visiting

        if ctr < 30:
            tau = 0.01
        else:
            tau = 0.01

        node, move, move_idx = tree.root.greedy_select_new_root_node()

        chess_move = chess.Move.from_uci(move)
        main_board.push(chess_move)
        svg = chess.svg.board(board=main_board, arrows=[chess.svg.Arrow(chess_move.from_square, chess_move.to_square,
                                                                        color="#0000cccc")])

        print('\n')
        print(main_board)


        with open(f"save_game/move_{ctr}.svg", "w") as file:
            file.write(svg)

        if node:
            print(
                f"Move chosen: {move} Prob: {node.parent_node.Ps[move_idx]:.3f} Q: {node.parent_node.Qs[move_idx]:.3f} N: {node.parent_node.Ns[move_idx]}")
        print(f"Game over: {main_board.is_game_over()}")
        print('\n')

        tree.root = node
        tree.root.parent_move = None

        if main_board.is_checkmate():
            print("Checkmate!")
            break
        elif main_board.is_stalemate():
            print("Stalemate!")
            break
        elif main_board.is_insufficient_material():
            print("Insufficient material to checkmate!")
            break
        elif main_board.is_check():
            print("King is in check!")

        if int(node.state.split()[-2]) >= 25:
            print("Draw")
            break

        ctr += 1

