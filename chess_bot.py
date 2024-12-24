import chess
import chess_moves
from tree.tree import GameTree

if __name__ == '__main__':

    tree = GameTree(chess_moves.ChessEngine, num_threads=1)

    sims = 25000

    node = tree.root

    import time
    for _ in range(10):
        start_time = time.time()
        tree.parallel_search(current_node=node,number_of_expansions=sims)
        end_time = time.time()

        print(f"Time with parallel search {end_time-start_time:.3f}")

        #TODO: sort out now visiting
        print(sum([edge.N for edge in tree.root.edges]))

        node = tree.root.select_new_root_node()
        tree.root = node
