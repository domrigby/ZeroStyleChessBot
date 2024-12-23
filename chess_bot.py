import chess
import chess_moves
from tree.tree import GameTree

if __name__ == '__main__':

    tree = GameTree(chess_moves.ChessEngine, num_threads=1)

    sims = 25000

    import time
    start_time = time.time()
    tree.parallel_search(number_of_expansions=sims)
    end_time = time.time()

    print(f"Time with parallel search {end_time-start_time:.3f}")
    print(sum([edge.N for edge in tree.root.edges]))

    print(len(tree.memory.states))

    # tree = Tree(chess.Board)
    # start_time = time.time()
    # tree.search_down_tree(sims)
    # end_time = time.time()
    # #
    # print(f"Time with no parallel search {end_time-start_time:.3f}")

    print([edge.Q for edge in tree.root.edges])

    best_moves = sorted(tree.root.edges, key= lambda x: x.Q, reverse=True)

    # Create the root of the tree
    initial_board = chess.Board()

    # for move in best_moves:
    #     initial_board.reset_board()
    #     initial_board.push(move.move)
    #     print(initial_board, "\n")