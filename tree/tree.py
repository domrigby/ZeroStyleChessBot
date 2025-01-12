import multiprocessing as mp

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

from typing import List, Optional, Union, Dict
from line_profiler import profile

from tree.memory import Memory
from neural_nets.conv_net import ChessNet

from tree.evaluator import NeuralNetHandling
from tree.trainer import TrainingProcess


class GameTree:

    add_dirichlet_noise = False

    def __init__(self, env, env_kwargs: dict = None, num_threads: int = 6,
                 start_state: str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                 neural_net = None, manager: Manager = None, training: bool =  False,
                 multiprocess: bool = False, num_evalators: int = 1):

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

            # Create the queues
            self.process_queue = Queue()
            self.process_queue_node_lookup: Dict[int, tuple] = {}

            self.experience_queue = Queue()
            self.results_queue = Queue()

            self.evaluators: List[NeuralNetHandling] = []

            for _ in range(num_evalators):
                self.evaluators.append(NeuralNetHandling(neural_net=neural_net, process_queue=self.process_queue,
                                               results_queue=self.results_queue, batch_size=128))

            self.trainer = TrainingProcess(neural_net=neural_net, experience_queue=self.experience_queue, batch_size=128)

            [evaluator.start() for evaluator in self.evaluators]
            self.trainer.start()

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
                move = current_node.puct(thread_num, rng_generator=thread_rng)
                visited_moves: List[Move] = [move]

                while move.child_node and not current_node.branch_complete:

                    # Append move to visited list
                    visited_moves.append(move)
                    move.N += 1

                    # Get new node and then a new move
                    current_node = move.child_node

                    while all(move.child_node and move.child_node.awaiting_processing for move in current_node.moves):
                        print('Waiting')
                        time.sleep(0.0001)

                    if not current_node.branch_complete:
                        move = current_node.puct(rng_generator=thread_rng)

                    if not self.multiprocess and self.training and idx % 30 == 0:
                        self.train_neural_network_local()

                    if self.multiprocess:
                        self.apply_neural_net_results()

                    if move is None:
                        break

                    # Set virtual loss to discourage search down here for a bit
                    if self.multiprocess:
                        move.virtual_loss = -1.

                    # Currently have an issue with time exploding in the late game
                    time_now = time.time()
                    if time_now - start_time > time_limit:
                        break

                if not current_node.branch_complete and move is not None:
                    # If the graph is complete its already been done

                    current_node, reward, done = move.expand(thread_env, state=current_node.state, node_queue=self.nodes, thread_num=thread_num)

                    if not self.multiprocess:
                        value = self.neural_net.node_evaluation(current_node)
                        current_node.V = value
                        reward += value
                        reward /= 2
                        self.backpropagation(visited_moves, reward)
                    else:

                        # Save the backpropagation until node results have returned

                        # Create a hash of the node such that we can relocate this node later
                        node_hash = hash(current_node)

                        if node_hash not in self.process_queue_node_lookup:

                            states = current_node.state
                            legal_moves = current_node.legal_move_strings

                            self.process_queue_node_lookup[node_hash] = (current_node, visited_moves)

                            self.process_queue.put((node_hash, states, legal_moves))

                            # Set the node is current node is awaiting processing
                            current_node.awaiting_processing = False

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

    def backpropagation(self, visited_moves, observed_reward: float):

        gamma_factor = 1

        player = visited_moves[-1].parent_node.player

        for move in reversed(visited_moves):

            if move.parent_node.player == player:
                # Flip for when the other player is observing reward
                flip = 1
            else:
                flip = -1

            # Increment the visit count and update the total reward
            move.W += gamma_factor * flip * observed_reward

            # Unlock the move for further exploration
            move.virtual_loss = 0

            gamma_factor *= 1

    def search_for_sufficiently_visited_nodes(self, root_node):
        def recursive_search(node, count):

            self.save_results_to_memory(node, root_node)

            if node.branch_complete or count > 400:
                return

            for move in node.moves:
                if move.N >= 100:
                    recursive_search(move.child_node, count + 1)

            return

        recursive_search(root_node, 0)

    def save_results_to_memory(self, current_node, root_node):

        state = current_node.state
        moves = [move.move for move in current_node.moves]
        visit_counts = [move.N for move in current_node.moves]
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

        for the_hash, value, move_probs, index_map in results:
            # Retrieve the node associated with this hash
            node, visited_moves = self.process_queue_node_lookup.pop(the_hash)

            # Update node value
            node.V = value

            # Update parent's move total value if applicable
            if node.parent_move is not None:
                node.parent_move.W += value

            # Parameters for Dirichlet noise
            alpha = 0.3  # Dirichlet distribution parameter
            epsilon = 0.25  # Blending factor
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
            if self.add_dirichlet_noise:
                dirichlet_noise = np.random.dirichlet([alpha] * len(Ps))
                Ps = (1 - epsilon) * Ps + epsilon * dirichlet_noise

            # Update move probabilities
            for move, prob in zip(node.moves, Ps):
                move.P = prob

            node.awaiting_processing = False
            self.backpropagation(visited_moves, value)


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
    def __init__(self, board=None, parent_move=None, state=None):

        if board is not None:
            # TODO: sort out move initialisation
            self.moves: List[Move] = []
            self.re_init(state, board, parent_move)
        else:
            # Initialise empty forms of the class
            self.state: Optional[str] = None  # The chess fen string object representing this state
            self.parent_move: Optional[Move] = None
            self.player: Optional[str] = None

            self.number_legal_moves: int = 0
            self.branch_complete: bool = False  # Flag to indicate if the branch has been fully searched

            self.moves: List[Move] = [Move() for _ in range(self.DEFAULT_EDGE_NUM)]

            self.has_policy: bool = False
            self.team: Optional[str] = None

            # State value
            self.V: float = 0

            # Set a flag saying it has not been processed yet
            self.awaiting_processing: bool = True

    def re_init(self, state:str, board, parent_move=None):

        self.state = state  # The chess fen string object representing this state
        self.player = self.state.split()[1]
        assert self.player in ['w', 'b'], ValueError('Player must be w or b')

        self.parent_move = parent_move  # Reference to the parent move

        board.set_fen(self.state)
        legal_moves =  board.legal_moves()

        self.number_legal_moves = len(list(legal_moves))

        # Check if we have fully searched the branch
        if self.number_legal_moves == 0:
            self.branch_complete = True
        else:
            self.branch_complete = False

        created_move_num = len(self.moves)
        for idx, move in enumerate(legal_moves):
            if idx < created_move_num:
                self.moves[idx].re_init(self, move)
            else:
                self.moves.append(Move(self, move))

        if self.number_legal_moves < len(self.moves):
            # Get rid of excess moves
            self.moves = self.moves[:self.number_legal_moves]

        self.has_policy = False
        self.team = "white" if self.state.split()[1] == 'w' else "black"

        # State value
        self.V = 0

    def puct(self, draw_num: int = None, rng_generator: default_rng = None):
        """
        Select the best move using the PUCT formula,
        ignoring moves that lead to completed branches.
        """
        move_N = np.array([move.N for move in self.moves])
        move_P = np.array([move.P for move in self.moves])
        move_Q = np.array([move.Q for move in self.moves])
        move_virtual_loss = np.array([move.virtual_loss for move in self.moves])
        branch_complete = np.array([move.child_node.branch_complete if move.child_node is not None else False for move in self.moves])

        # Ignore completed branches
        valid_moves = np.where(~branch_complete)[0]

        if len(valid_moves) == 0:
            return None

        # Use only valid moves in PUCT calculation
        visits = np.sum(move_N[valid_moves])
        puct_values = (move_Q[valid_moves] + move_virtual_loss[valid_moves]) + \
                      5.0 * move_P[valid_moves] * np.sqrt(visits + 1) / (move_N[valid_moves] + 1)

        # Select the move with the maximum PUCT value
        best_value = np.max(puct_values)
        best_indexes = np.where(puct_values == best_value)[0]

        best_index = rng_generator.choice(best_indexes)

        return self.moves[best_index]

    def select_new_root_node(self, tau: float = 1.0):
        Ns = np.array([move.N for move in self.moves])
        N_to_tau = np.power(Ns, 1./tau)
        probs = N_to_tau / np.sum(N_to_tau)
        chosen_move = np.random.choice(self.moves, p=probs)
        return chosen_move.child_node, chosen_move.move

    def greedy_select_new_root_node(self):
        Qs = np.array([move.Q for move in self.moves])
        chosen_idx = np.argmax(Qs)
        chosen_move = self.moves[chosen_idx]
        print(Qs)
        return chosen_move.child_node, chosen_move.move

    @property
    def legal_move_strings(self):
        return [str(move) for move in self.moves]

class Move:

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
            self.child_node.re_init(new_state, board, parent_move=self)
        else:  # If no queue provided, create a new node directly
            self.child_node = Node(board, parent_move=self)

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
    print(sum([move.N for move in tree.root.moves]))

    best_moves = sorted(tree.root.moves, key= lambda x: x.Q, reverse=True)

    # Create the root of the tree
    initial_board = chess.Board()
