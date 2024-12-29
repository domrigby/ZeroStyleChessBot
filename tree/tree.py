import chess
import numpy as np
from numpy.random import default_rng

# You need to build the C++ bit first
import chess_moves
import torch

from multiprocessing import Process, Manager #, Lock
from threading import Lock
from queue import Queue, SimpleQueue

from typing import List
from line_profiler import profile

from tree.memory import Memory
from neural_nets.conv_net import ChessNet


class GameTree:
    def __init__(self, env, env_kwargs: dict = None, num_threads: int = 6,
                 start_state: str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                 neural_net = None, manager: Manager = None, training: bool =  False,
                 multiprocess: bool = False):

        if env_kwargs is None:
            env_kwargs = {}

        # Create a set of game environments for each thread
        self.env = env

        # Create the root node of the tree
        self.root = Node(self.env(), state=start_state)
        self.nodes: List[Node] = []

        # Create the queues
        self.num_workers = num_threads

        self.neural_net = neural_net

        # Give this the manager object
        self.manager = manager

        # Training switch
        self.training = training
        self.multiprocess = multiprocess

        if not self.multiprocess:
            self.memory = Memory(100000, preload_data="/home/dom/Code/chess_bot/neural_nets/data/games.pkl")

        # if self.training:
        #     self.evaluator = Evaluator(queue=self.state_queue, lock=self.lock, neural_network=self.neural_net)
        #     self.evaluator.start()

    def reset(self):
        self.root = Node(self.env(), state="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        self.nodes = []

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

                    if self.training and idx % 30 == 0:
                        self.train_neural_network_local()

                current_node, reward, done = edge.expand(thread_env, state=current_node.state, node_queue=self.nodes, thread_num=thread_num)

                current_node = self.root

                if self.neural_net:
                    value = self.neural_net.node_evaluation(current_node)
                    reward += value
                    reward /= 2

                self.backpropagation(visited_edges, reward)

            self.search_for_sufficiently_visited_nodes(self.root)

        self.nodes = ThreadLocalNodePool(self.num_workers, number_of_expansions//self.num_workers)
        for _ in range(number_of_expansions):
            self.nodes.put(Node())

        search_down_tree(self.env, number_of_expansions // self.num_workers, current_node, 0)



    def backpropagation(self, visited_edges, observed_reward: float):

        gamma_factor = 1

        player = visited_edges[-1].parent_node.player

        for edge in reversed(visited_edges):

            with edge.lock:

                if edge.parent_node.player == player:
                    # Flip for when the other player is observing reward
                    flip = 1
                else:
                    flip = -1

                edge.N += 1
                edge.W += gamma_factor * flip * observed_reward

                # Unlock the edge for further exploration
                edge.virtual_loss = 0

                gamma_factor *= 1

    def search_for_sufficiently_visited_nodes(self, root_node):
        def recursive_search(node):

            self.save_results_to_memory(node)

            if node.branch_complete:
                return

            for edge in node.edges:
                if edge.N >= 250:
                    recursive_search(edge.child_node)

            return

        recursive_search(root_node)

    def save_results_to_memory(self, current_node):

        state = current_node.state
        moves = [edge.move for edge in current_node.edges]
        visit_counts = [edge.N for edge in current_node.edges]

        self.memory.save_state_to_moves(state, moves, visit_counts)

    def train_neural_network_local(self):
        if len(self.memory) < 32 or self.memory.games_played < 1:
            return
        states, moves, probs, wins = self.memory.get_batch(32)
        state, moves, wins, legal_move_mask = self.neural_net.tensorise_batch(states, moves, probs, wins)
        self.neural_net.train_batch(state, target=(wins, moves), legal_move_mask=legal_move_mask)

    def end_game(self, white_win: bool):
        self.memory.end_game(white_win)


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

        self.lock = Lock()

        if board is not None:
            # TODO: sort out edge initialisation
            self.edges: List[Edge] = []
            self.re_init(state, board, parent_edge)
        else:
            # Initialise empty forms of the class
            self.state: board.fen = None  # The chess fen string object representing this state
            self.parent_edge: Edge = None
            self.player = None

            self.number_legal_moves: int = 0
            self.branch_complete = False  # Flag to indicate if the branch has been fully searched

            self.edges: List[Edge] = [Edge() for _ in range(self.DEFAULT_EDGE_NUM)]

            self.has_policy = False

    def re_init(self, state:str, board, parent_edge=None):

        self.state = state  # The chess fen string object representing this state
        self.player = self.state.split()[1]
        assert self.player in ['w', 'b'], ValueError('Player must be w or b')

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

        if self.number_legal_moves < len(self.edges):
            # Get rid of excess edges
            self.edges = self.edges[:self.number_legal_moves]

        self.has_policy = False

    def puct(self, draw_num: int = None, rng_generator: default_rng = None):
        #TODO sort puct out with the locks
        # Do all puct stuff in one go so that other threads cant edit it whilst
        visits = np.sum([edge.N for edge in self.edges])
        return max(self.edges, key=lambda edge: ((edge.Q + edge.virtual_loss) + 5.0 * edge.P * np.sqrt(visits + 1) / (edge.N + 1)))

    def select_new_root_node(self, tau: float = 1.0):
        Ns = np.array([edge.N for edge in self.edges])
        N_to_tau = np.power(Ns, 1./tau)
        probs = N_to_tau / np.sum(N_to_tau)
        chosen_edge = np.random.choice(self.edges, p=probs)
        return chosen_edge.child_node, chosen_edge.move

    def apply_policy(self, state_tensor: torch.tensor):
        pass

    @property
    def legal_move_strings(self):
        return [str(edge) for edge in self.edges]

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
        self.P = 1.

        # Multi-processing variables
        self.virtual_loss = 0
        self.lock = Lock()

    def re_init(self, parent_node, move):
        self.parent_node = parent_node
        self.move = move

    def __str__(self):
        return str(self.move)

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

    chess_net = ChessNet(input_size=[12, 8, 8], output_size=[66, 8, 8], num_repeats=16)
    chess_net.load_network(r"/home/dom/Code/chess_bot/neural_nets/session/best_model_120.pt")
    chess_net.eval()

    tree = GameTree(chess_moves.ChessEngine, num_threads=1, neural_net=chess_net)

    sims = 25000

    import time
    start_time = time.time()
    for i in range(10):
        tree.parallel_search(number_of_expansions=sims)
        tree.root_node
    end_time = time.time()

    print(f"Time with parallel search {end_time-start_time:.3f}")
    print(sum([edge.N for edge in tree.root.edges]))

    best_moves = sorted(tree.root.edges, key= lambda x: x.Q, reverse=True)

    # Create the root of the tree
    initial_board = chess.Board()
