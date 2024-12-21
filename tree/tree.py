import chess
import numpy as np
from numpy.random import default_rng

from copy import copy

import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor

from typing import List
import os
from line_profiler import profile

import gc


class Tree:
    def __init__(self, env, env_kwargs: dict = None, num_threads: int = 6):

        if env_kwargs is None:
            env_kwargs = {}

        # Create a set of game environments for each thread
        self.envs = [env(**env_kwargs) for _ in range(num_threads)]

        # Create the root node of the tree
        self.root = Node(self.envs[0])
        self.nodes: List[Node] = []

        # Create the queues
        self.state_queue = Queue()
        self.lock = threading.Lock()
        self.num_workers = num_threads

    def search_down_tree(self, num_expands: int):

        for idx in range(num_expands):
            self.expand_the_tree(self.root)

    def expand_the_tree(self, current_node):

        # Create bool for while loop
        leaf = False

        # Create list of visited edges
        visited_edges: List[Edge] = []

        while not leaf:

            # Select the best child node according to PUCT
            edge = current_node.puct()

            # Add current node to the list of visited nodes
            visited_edges.append(edge)

            if edge.child_node is None:
                # If the node has no children, it's a leaf
                leaf = True
                # Set board to correct state with the nodes FEN
                self.envs[0].set_fen(current_node.state)
                current_node, reward, done = edge.expand(self.envs[0])
                self.nodes.append(current_node)
            else:
                # Else carry on searching
                current_node = edge.child_node

        self.backpropagation(visited_edges, reward)

    def parallel_search(self, current_node = None, number_of_expansions: int = 1000):

        if current_node is None:
            current_node = self.root

        # Preallocated the memory

        def search_down_tree(board, thread_num_expansions: int, current_node, thread_num: int = 0):

            # Sharaing a radndom number generator SEVERELY slows down the process... explainationn pending
            thread_rng = default_rng()

            for idx in range(thread_num_expansions):

                # Thread num is used a draw breaker for early iterations such that they dont search down the same branch
                edge = current_node.puct(thread_num, rng_generator=thread_rng)
                visited_edges: List[Edge] = [edge]
                leaf = False

                while edge.child_node and not current_node.branch_complete:

                    visited_edges.append(edge)

                    with edge.lock:
                        # Tunes PUCT such that other threads won't follow it
                        edge.virtual_loss = -10.

                    current_node = edge.child_node

                    # Get new edge
                    edge = current_node.puct(rng_generator=thread_rng)

                board.set_fen(current_node.state)
                current_node, reward, done = edge.expand(board)
                self.backpropagation(visited_edges, reward)

                # TODO: change this so it doesnt lock the whole of self.nodes
                # with self.lock:
                #     # Write to the nodes
                #     self.nodes.append(current_node)

                # Reset the top level node
                current_node = self.root

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:

            futures = [executor.submit(search_down_tree, board, number_of_expansions//self.num_workers,
                                       current_node, thread_num)
                       for thread_num, board in enumerate(self.envs)]

            for futures in futures:
                futures.result()

    def backpropagation(self, visited_edges, observed_reward: float):

        gamma_factor = 0.99

        for edge in reversed(visited_edges):

            with edge.lock:

                edge.N += 1
                edge.W += gamma_factor * observed_reward

                # Unlock the edge for further exploration
                edge.virtual_loss = 0

                gamma_factor *= 0.99

class Node:
    # Makingnew commit
    def __init__(self, board, parent_edge=None):

        self.state= board.fen()  # The chess fen string object representing this state
        self.parent_edge = parent_edge  # Reference to the parent edge

        legal_moves = board.legal_moves

        self.number_legal_moves = len(list(legal_moves))

        # Check if we have fully searched the branch
        if self.number_legal_moves == 0:
            self.branch_complete = True
        else:
            self.branch_complete = False

        self.edges= [Edge(self, move) for move in legal_moves]

        self.has_policy = False

    def puct(self, draw_num: int = None, rng_generator: default_rng = None):
        # Do all puct stuff in one go so that other threads cant edit it whilst
        # visits = np.sum([edge.N for edge in self.edges])
        # return max(self.edges, key=lambda edge: ((edge.Q + edge.virtual_loss) + 5.0 * np.sqrt(visits + 1) / (edge.N + 1)))

        Qs = np.array([edge.Q - edge.virtual_loss for edge in self.edges])
        Ns = np.array([edge.N for edge in self.edges])

        # PUCT calculation
        puct_vals = Qs + 5.0 * np.sqrt(np.sum(Ns)) / (1 + Ns)

        # Choose the child node with the highest PUCT value
        max_indices = np.where(puct_vals== np.max(puct_vals))[0]

        # Randomly choose one of the indices
        if draw_num is None:
            if rng_generator is None:
                winner_index = np.random.choice(max_indices)
            else:
                # Im aware this could be done using choice...but this extremely slow in parallel
                rnd_idx = rng_generator.integers(low=0, high=len(max_indices))
                winner_index = max_indices[rnd_idx]
        else:
            draw_num = draw_num % len(max_indices)
            winner_index = max_indices[draw_num]

        return self.edges[winner_index]


class Edge:

    def __init__(self, parent_node, move):
        # Corresponding move
        self.move = move

        # Tree connections
        self.parent_node = parent_node
        self.child_node = None

        # Game statistics
        self.W = 0.
        self.N = 0.

        # Multi-processing variables
        self.virtual_loss = 0
        self.lock = threading.Lock()

    @profile
    def expand(self, board: chess.Board):

        # Check if the move is a capture (not a pawn move)
        capture = board.is_capture(self.move)

        # Run the sim to update the board
        board.push(self.move)

        # Create a new child node
        self.child_node = Node(board, parent_edge=self)

        # Check if done
        done = board.is_game_over()

        # Calculate the reward
        if done:
            reward = 1. if board.result() == "1-0" else -1. if board.result() == "0-1" else 0.
        else:
            reward = 0.

        if capture:
            reward += 1.

        # Update statistics

        return self.child_node, reward, done

    @property
    def Q(self):
        Q =  self.W / self.N if self.N > 0 else 0.
        return Q



#
tree = Tree(chess.Board)


import time
start_time = time.time()
tree.parallel_search(number_of_expansions=50000)
end_time = time.time()

# print(f"Time with parallel search {end_time-start_time:.3f} Nodes: {len(tree.nodes)}")
# print([edge.N for edge in tree.root.edges])

# tree = Tree(chess.Board)
# start_time = time.time()
# tree.search_down_tree(50000)
# end_time = time.time()

print(f"Time with no parallel search {end_time-start_time:.3f} Nodes: {len(tree.nodes)}")

print([edge.Q for edge in tree.root.edges])

best_moves = sorted(tree.root.edges, key= lambda x: x.Q, reverse=True)

# # Create the root of the tree
# initial_board = chess.Board()
#
# for move in best_moves:
#     initial_board.reset_board()
#     initial_board.push(move.move)
#     print(initial_board, "\n")
