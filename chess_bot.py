import chess
import chess.svg
import chess_moves
from tree.tree import GameTree


from neural_nets.conv_net import ChessNet

if __name__ == '__main__':

    chess_net = ChessNet(input_size=[12, 8, 8], output_size=[70, 8, 8], num_repeats=16)
    chess_net.load_network(r"/home/dom/Code/chess_bot/networks/best_model_127.pt")
    chess_net.eval()

    tree = GameTree(chess_moves.ChessEngine, num_threads=1, neural_net=chess_net, multiprocess=True)

    sims = 5000

    node = tree.root

    main_board = chess.Board()

    import time
    ctr = 0

    while not main_board.is_game_over():
        start_time = time.time()
        tree.parallel_search(current_node=node,number_of_expansions=sims)
        end_time = time.time()

        print(f"Time with parallel search {end_time-start_time:.3f}")

        #TODO: sort out now visiting

        if ctr < 30:
            tau = 0.5
        else:
            tau = 0.1

        node, move = tree.root.select_new_root_node(tau=tau)

        chess_move = chess.Move.from_uci(move)
        main_board.push(chess_move)
        svg = chess.svg.board(board=main_board, arrows=[chess.svg.Arrow(chess_move.from_square, chess_move.to_square,
                                                                        color="#0000cccc")])

        print('\n')
        print(main_board)

        print([f"{edge.move} {edge.N} {edge.Q:.4f}, {edge.P:.3f} \n" for edge in tree.root.edges])

        print(sum([edge.P for edge in tree.root.edges]))

        with open(f"save_game/move_{ctr}.svg", "w") as file:
            file.write(svg)

        print(f"Move: {move}")
        print(f"Game over: {main_board.is_game_over()}")
        print('\n')
        tree.root = node

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

        ctr += 1

