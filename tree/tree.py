import multiprocessing as mp

from duplicity.config import current_time

try:
   mp.set_start_method('spawn', force=True)
   print("spawned")
except RuntimeError:
   pass


import chess
import numpy as np
from numpy.random import default_rng
import time

# You need to build the C++ bit first
import chess_moves
import torch

from multiprocessing import Process, Manager, Queue, Lock
from queue import Empty

from typing import List
from line_profiler import profile

from tree.memory import Memory
from neural_nets.conv_net import ChessNet

from tree.evaluator import NeuralNetHandling


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

        # Give this the manager object
        self.manager = manager

        # Training switch
        self.training = training
        self.multiprocess = multiprocess

        self.neural_net = None
        if not self.multiprocess:
            self.neural_net = neural_net
            self.memory = Memory(100000)
        else:
            self.process_queue = Queue()
            self.process_queue_node_lookup = {}
            self.awaiting_backpropagation_lookup = {}

            self.experience_queue = Queue()
            self.results_queue = Queue()

            self.evaluator = NeuralNetHandling(neural_net=neural_net, process_queue=self.process_queue,
                                               experience_queue=self.experience_queue,
                                               results_queue=self.results_queue, batch_size=128)
            self.evaluator.start()

    def reset(self):
        self.root = Node(self.env(), state="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        self.nodes = []

    def parallel_search(self, current_node = None, number_of_expansions: int = 1000, time_limit: float = 100.):

        if current_node is None:
            current_node = self.root

        @profile
        def search_down_tree(env, thread_num_expansions: int, current_node, thread_num: int = 0, time_limit: float = 100.):

            # Sharaing a random number generator SEVERELY slows down the process... explainationn pending
            thread_rng = default_rng()

            # Initialise the threads env
            thread_env = env()
            start_time = time.time()

            for idx in range(thread_num_expansions):

                # Thread num is used a draw breaker for early iterations such that they dont search down the same branch
                edge = current_node.puct(thread_num, rng_generator=thread_rng)
                visited_edges: List[Edge] = [edge]

                while edge.child_node and not current_node.branch_complete:

                    # Append edge to visited list
                    visited_edges.append(edge)
                    edge.N += 1

                    # Get new node and then a new edge
                    current_node = edge.child_node

                    if not current_node.branch_complete:
                        edge = current_node.puct(rng_generator=thread_rng)

                    if not self.multiprocess and self.training and idx % 30 == 0:
                        self.train_neural_network_local()

                    if self.multiprocess:
                        self.apply_neural_net_results()

                    # Currently have an issue with time exploding in the late game
                    time_now = time.time()
                    if time_now - start_time > time_limit:
                        break

                if not current_node.branch_complete:
                    # If the graph is complete its already been done

                    current_node, reward, done = edge.expand(thread_env, state=current_node.state, node_queue=self.nodes, thread_num=thread_num)

                    if not self.multiprocess:
                        value = self.neural_net.node_evaluation(current_node)
                        current_node.V = value
                        reward += value
                        reward /= 2
                        self.backpropagation(visited_edges, reward)
                    else:

                        # Save the backpropagation until node results have returned
                        visited_edges_tuple = tuple(visited_edges)
                        visited_edges_hash = hash(visited_edges_tuple)

                        if visited_edges_hash not in self.awaiting_backpropagation_lookup:
                            self.awaiting_backpropagation_lookup[visited_edges_hash] = visited_edges_tuple

                            # TODO: current_node is massive overkill for communication
                            node_hash = hash(current_node)
                            states = current_node.state
                            legal_moves = current_node.legal_move_strings

                            self.process_queue_node_lookup[node_hash] = current_node

                            self.process_queue.put((node_hash, states, legal_moves, visited_edges_hash))

                            # Set virtual loss to discourage search down here for a bit
                            self.virtual_loss = -10.

                # Now return to root node
                current_node = self.root

                # Final check on time limit
                time_now = time.time()
                if time_now - start_time > time_limit:
                    print('Time limit exceeded. Returning.')
                    break

            self.search_for_sufficiently_visited_nodes(self.root)

        self.nodes = ThreadLocalNodePool(self.num_workers, number_of_expansions//self.num_workers)
        for _ in range(number_of_expansions):
            self.nodes.put(Node())

        search_down_tree(self.env, number_of_expansions // self.num_workers, current_node, 0, time_limit=time_limit)



    def backpropagation(self, visited_edges, observed_reward: float):

        gamma_factor = 1

        player = visited_edges[-1].parent_node.player

        for edge in reversed(visited_edges):

            if edge.parent_node.player == player:
                # Flip for when the other player is observing reward
                flip = 1
            else:
                flip = -1

            # Increment the visit count and update the total reward
            edge.W += gamma_factor * flip * observed_reward

            # Unlock the edge for further exploration
            edge.virtual_loss = 0

            gamma_factor *= 1

    def search_for_sufficiently_visited_nodes(self, root_node):
        def recursive_search(node, count):

            self.save_results_to_memory(node, root_node)

            if node.branch_complete or count > 400:
                return

            for edge in node.edges:
                if edge.N >= 100:
                    recursive_search(edge.child_node, count + 1)

            return

        recursive_search(root_node, 0)

    def save_results_to_memory(self, current_node, root_node):

        state = current_node.state
        moves = [edge.move for edge in current_node.edges]
        visit_counts = [edge.N for edge in current_node.edges]
        predicted_value = current_node.V

        if not self.multiprocess:
            self.memory.save_state_to_moves(state, moves, visit_counts, predicted_value, current_node is root_node)
        else:
            self.experience_queue.put((state, moves, visit_counts, predicted_value, current_node is root_node))

    def apply_neural_net_results(self):
        """
        Process results from the neural network in batches, updating nodes and applying Dirichlet noise.
        """
        max_batch_size = 64  # Process up to 10 items at a time

        # Collect a batch of items from the queue
        results = []
        for _ in range(max_batch_size):
            try:
                results.append(self.results_queue.get_nowait())  # Non-blocking get
            except Empty:
                break

        if not results:
            return  # No results to process

        for the_hash, value, move_probs, index_map, visited_edges_hash in results:
            # Retrieve the node associated with this hash
            node = self.process_queue_node_lookup.pop(the_hash)

            # Update node value
            node.V = value

            # Update parent's edge total value if applicable
            if node.parent_edge is not None:
                node.parent_edge.W += value

            # Parameters for Dirichlet noise
            alpha = 0.3  # Dirichlet distribution parameter
            epsilon = 0.25  # Blending factor

            # Convert move probabilities to a numpy array
            Ps = np.array([prob[-1] for prob in move_probs])

            # Normalize probabilities
            sum_Ps = Ps.sum()
            if sum_Ps > 1e-6:
                Ps /= sum_Ps
            else:
                Ps = np.ones_like(Ps) / len(Ps)  # Assign uniform probability if sum is too small

            # Apply Dirichlet noise
            dirichlet_noise = np.random.dirichlet([alpha] * len(Ps))
            Ps = (1 - epsilon) * Ps + epsilon * dirichlet_noise

            # Update edge probabilities
            for edge, prob in zip(node.edges, Ps):
                edge.P = prob

            # Perform backpropagation
            visited_edges = self.awaiting_backpropagation_lookup.pop(visited_edges_hash)
            self.backpropagation(visited_edges, value)


    def train_neural_network_local(self):
        if len(self.memory) < 32:
            return
        states, moves, probs, wins = self.memory.get_batch(32)
        state, moves, wins, legal_move_mask = self.neural_net.tensorise_batch(states, moves, probs, wins)
        self.neural_net.loss_function(state, target=(wins, moves), legal_move_mask=legal_move_mask)

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
            self.team = None

            # State value
            self.V = 0

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
        self.team = "white" if self.state.split()[1] == 'w' else "black"

        # State value
        self.V = 0

    def puct(self, draw_num: int = None, rng_generator: default_rng = None):
        """
        Select the best edge using the PUCT formula,
        ignoring edges that lead to completed branches.
        """
        edge_N = np.array([edge.N for edge in self.edges])
        edge_P = np.array([edge.P for edge in self.edges])
        edge_Q = np.array([edge.Q for edge in self.edges])
        edge_virtual_loss = np.array([edge.virtual_loss for edge in self.edges])
        branch_complete = np.array([edge.child_node.branch_complete if edge.child_node is not None else False for edge in self.edges])

        # Ignore completed branches
        valid_edges = np.where(~branch_complete)[0]

        if len(valid_edges) == 0:
            raise RuntimeError("All branches are complete; no valid moves.")

        # Use only valid edges in PUCT calculation
        visits = np.sum(edge_N[valid_edges])
        puct_values = (edge_Q[valid_edges] + edge_virtual_loss[valid_edges]) + \
                      5.0 * edge_P[valid_edges] * np.sqrt(visits + 1) / (edge_N[valid_edges] + 1)

        # Select the edge with the maximum PUCT value
        best_index = valid_edges[np.argmax(puct_values)]

        return self.edges[best_index]

    def select_new_root_node(self, tau: float = 1.0):
        Ns = np.array([edge.N for edge in self.edges])
        N_to_tau = np.power(Ns, 1./tau)
        probs = N_to_tau / np.sum(N_to_tau)
        chosen_edge = np.random.choice(self.edges, p=probs)
        return chosen_edge.child_node, chosen_edge.move

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
        done, result_code = board.is_game_over()

        if done:
            print("End game found!")
            self.child_node.branch_complete = True
            if result_code == 1:
                reward = 1.
            else:
                reward = 0.
        else:
            reward = 0.

        # Update statistics

        return self.child_node, reward, done

    @property
    def Q(self):
        Q =  self.W / self.N if self.N > 0 else torch.tensor(0)
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
    end_time = time.time()

    print(f"Time with parallel search {end_time-start_time:.3f}")
    print(sum([edge.N for edge in tree.root.edges]))

    best_moves = sorted(tree.root.edges, key= lambda x: x.Q, reverse=True)

    # Create the root of the tree
    initial_board = chess.Board()
