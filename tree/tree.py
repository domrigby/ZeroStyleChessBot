import chess
import numpy as np
from numpy.random import default_rng

# You need to build the C++ bit first
import os
import chess_moves

from copy import copy

import threading
from queue import Queue, SimpleQueue
from concurrent.futures import ThreadPoolExecutor

from typing import List
from line_profiler import profile

import gc
from numba import jit

from tree.memory import Memory


class GameTree:
    def __init__(self, env, env_kwargs: dict = None, num_threads: int = 6,
                 start_state: str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"):

        if env_kwargs is None:
            env_kwargs = {}

        # Create a set of game environments for each thread
        self.env = env

        # Create the root node of the tree
        self.root = Node(self.env(), state=start_state)
        self.nodes: List[Node] = []

        self.memory = Memory(100000)

        # Create the queues
        self.state_queue = SimpleQueue()
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
                self.envs[0].set_position(current_node.state)
                current_node, reward, done = edge.expand(self.envs[0])
                self.nodes.append(current_node)
            else:
                # Else carry on searching
                current_node = edge.child_node

        self.backpropagation(visited_edges, reward)

    def parallel_search(self, current_node = None, number_of_expansions: int = 1000):

        if current_node is None:
            current_node = self.root

        @profile
        def search_down_tree(env, thread_num_expansions: int, current_node, thread_num: int = 0):

            # Sharaing a radndom number generator SEVERELY slows down the process... explainationn pending
            thread_rng = default_rng()

            # Initialise the threads env
            thread_env = env()

            for idx in range(thread_num_expansions):

                # Thread num is used a draw breaker for early iterations such that they dont search down the same branch
                edge = current_node.puct(thread_num, rng_generator=thread_rng)
                visited_edges: List[Edge] = [edge]

                while edge.child_node and not current_node.branch_complete:

                    # Append edge to visited list
                    visited_edges.append(edge)

                    with edge.lock:
                        # Tunes PUCT such that other threads won't follow it
                        edge.virtual_loss = -10.

                    # Get new node and then a new edge
                    current_node = edge.child_node

                    if not current_node.branch_complete:
                        edge = current_node.puct(rng_generator=thread_rng)

                current_node, reward, done = edge.expand(thread_env, state=current_node.state, node_queue=self.nodes, thread_num=thread_num)
                self.backpropagation(visited_edges, reward)

                current_node = self.root

            self.search_for_sufficiently_visited_nodes(self.root)
            self.save_results_to_memory(self.root)
            print("here")

        self.nodes = ThreadLocalNodePool(self.num_workers, number_of_expansions//self.num_workers)
        for _ in range(number_of_expansions):
            self.nodes.put(Node())

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:

            # TODO: change the to multiprocess with shared memory tree... stupidly misread the thread GIL relationship
            futures = [executor.submit(search_down_tree, self.env, number_of_expansions//self.num_workers,
                                       current_node, thread_num)
                       for thread_num in range(self.num_workers)]

            for future in futures:
                future.result()


    def backpropagation(self, visited_edges, observed_reward: float):

        gamma_factor = 0.99

        for edge in reversed(visited_edges):

            with edge.lock:

                edge.N += 1
                edge.W += gamma_factor * observed_reward

                # Unlock the edge for further exploration
                edge.virtual_loss = 0

                gamma_factor *= 0.99

    def search_for_sufficiently_visited_nodes(self, root_node):

        def recursive_search(node):

            self.save_results_to_memory(node)

            if node.branch_complete:
                return

            for edge in node.edges:
                if edge.N >= 500:
                    recursive_search(edge.child_node)

    def save_results_to_memory(self, current_node):

        state = current_node.state
        moves = [edge.move for edge in current_node.edges]
        visit_counts = [edge.N for edge in current_node.edges]

        self.memory.save_state_to_moves(state, moves, visit_counts)

class ThreadLocalNodePool:
    def __init__(self, num_threads, pool_size):
        self.num_threads = num_threads
        self.ctr = 0
        self.local_pools = [Queue() for _ in range(num_threads)]
        for pool in self.local_pools:
            for _ in range(pool_size):
                pool.put(Node())

    def get(self, thread_id):
        return self.local_pools[thread_id].get()

    def put(self, node):
        thread = self.ctr % self.num_threads
        self.ctr += 1
        self.local_pools[thread].put(node)


class Node:

    DEFAULT_EDGE_NUM = 24
    def __init__(self, board=None, parent_edge=None, state=None):

        self.lock = threading.Lock()

        if board is not None:
            # TODO: sort out edge initialisation
            self.edges: List[Edge] = []
            self.re_init(state, board, parent_edge)
        else:
            # Initialise empty forms of the class
            self.state: board.fen = None  # The chess fen string object representing this state
            self.parent_edge: Edge = None

            self.number_legal_moves: int = 0
            self.branch_complete = False  # Flag to indicate if the branch has been fully searched

            self.edges: List[Edge] = [Edge() for _ in range(self.DEFAULT_EDGE_NUM)]

            self.has_policy = False

    def re_init(self, state:str, board, parent_edge=None):

        self.state = state  # The chess fen string object representing this state

        self.parent_edge = parent_edge  # Reference to the parent edge

        board.set_fen(self.state)
        legal_moves =  board.legal_moves()

        self.number_legal_moves = len(list(legal_moves))

        # Check if we have fully searched the branch
        if self.number_legal_moves == 0:
            self.branch_complete = True
        else:
            self.branch_complete = False

        created_edge_num = len(self.edges)
        for idx, move in enumerate(legal_moves):
            if idx < created_edge_num:
                self.edges[idx].re_init(self, move)
            else:
                self.edges.append(Edge(self, move))

        self.has_policy = False

    def puct(self, draw_num: int = None, rng_generator: default_rng = None):

        #TODO sort puct out with the locks
        # Do all puct stuff in one go so that other threads cant edit it whilst
        # visits = np.sum([edge.N for edge in self.edges])
        # return max(self.edges, key=lambda edge: ((edge.Q + edge.virtual_loss) + 5.0 * np.sqrt(visits + 1) / (edge.N + 1)))
        used_edges = [edge for edge in self.edges if edge.move is not None]

        Qs = np.array([edge.Q + edge.virtual_loss for edge in used_edges])
        Ns = np.array([edge.N for edge in used_edges])

        # PUCT calculation
        puct_vals = Qs + 5.0 * np.sqrt(np.sum(Ns)) / (1 + Ns)

        if len(puct_vals) == 0:
            print(self.number_legal_moves)

        # Choose the child node with the highest PUCT value
        max_indices = np.where(puct_vals == np.max(puct_vals))[0]

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

        return used_edges[winner_index]


class Edge:

    def __init__(self, parent_node: Node = None, move=None):

        if parent_node is not None and move is not None:
            self.re_init(parent_node, move)
        else:
            self.parent_node: Node = None
            self.move = None

        self.child_node = None

        # Game statistics
        self.W = 0.
        self.N = 0.

        # Multi-processing variables
        self.virtual_loss = 0
        self.lock = threading.Lock()

    def re_init(self, parent_node, move):
        self.parent_node = parent_node
        self.move = move

    @profile
    def expand(self, board: chess.Board, state: str = None, node_queue: Queue = None, thread_num: int = 0):

        if state is None:
            state = board.fen()  # Update the board state to the provided FEN string

        # Run the sim to update the board
        new_state = board.push(state, str(self.move))

        # Create a new child node
        if node_queue is not None:
            self.child_node = node_queue.get(thread_num)
            self.child_node.re_init(new_state, board, parent_edge=self)
        else:  # If no queue provided, create a new node directly
            self.child_node = Node(board, parent_edge=self)

        # Calculate the reward
        done = board.is_game_over()
        if done:
            reward = 1. if board.result() == "1-0" else -1. if board.result() == "0-1" else 0.
        else:
            reward = 0.

        # Update statistics

        return self.child_node, reward, done

    @property
    def Q(self):
        Q =  self.W / self.N if self.N > 0 else 0.
        return Q

#
if __name__ == '__main__':

    tree = GameTree(chess_moves.ChessEngine, num_threads=1)

    sims = 25000

    import time
    start_time = time.time()
    tree.parallel_search(number_of_expansions=sims)
    end_time = time.time()

    print(f"Time with parallel search {end_time-start_time:.3f}")
    print(sum([edge.N for edge in tree.root.edges]))

    best_moves = sorted(tree.root.edges, key= lambda x: x.Q, reverse=True)

    # Create the root of the tree
    initial_board = chess.Board()
