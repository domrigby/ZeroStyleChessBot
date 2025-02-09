import chess
import numpy as np
from numpy.random import default_rng
from copy import copy
import time
import torch
from multiprocessing import Queue, Process
from queue import Empty
from typing import List, Dict
from tree.memory import Memory
import chess_moves
import multiprocessing as mp
import cProfile

class GameTree(Process):

    add_dirichlet_noise = True

    agent_count = 0

    def __init__(self, neural_net = None, training: bool =  False,
                 multiprocess: bool = False, process_queue: Queue = None,
                 experience_queue: Queue = None, results_queue: Queue = None):

        super().__init__()

        self.agent_id = copy(GameTree.agent_count)
        GameTree.agent_count += 1

        # Training switch
        self.training = training
        self.multiprocess = multiprocess

        # Local count for the number of memories we've saved
        self.saved_memory_local = 0

        self.neural_net = None
        if not self.multiprocess:
            self.neural_net = neural_net
            self.memory = Memory(100000)
        else:
            # Link the queues
            self.process_queue = process_queue
            self.process_queue_node_lookup: Dict[int, tuple] = {}

            self.experience_queue = experience_queue
            self.results_queue = results_queue

        # Initiate root
        self.root = None

    def reset(self):
        self.root = Node(state="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", board=chess_moves.ChessEngine())

    def parallel_search(self, current_node = None, number_of_expansions: int = 1000, time_limit: float = 100.):

        profile = False
        if profile:
            profiler = cProfile.Profile()
            profiler.enable()

        if current_node is None:
            current_node = self.root

        # Sharing a random number generator SEVERELY slows down the process... explainationn pending
        thread_rng = default_rng()

        # Initialise the threads env
        thread_env = chess_moves.ChessEngine()

        for idx in range(number_of_expansions):

            # Thread num is used a draw breaker for early iterations such that they dont search down the same branch
            visited_moves: List[Move] = []
            move_has_child_node = True

            # Expand to leaf
            while move_has_child_node and not current_node.branch_complete:

                # Select
                move = current_node.puct(rng_generator=thread_rng)
                visited_moves.append(move)

                # Statistics
                move.N += 1

                # Set virtual loss to discourage search down here for a bit
                if self.multiprocess:
                    move.virtual_loss -= 1.

                if move.child_node:

                    # Get new node and then a new move
                    current_node = move.child_node

                else:
                    # We have found a leaf move
                    move_has_child_node = False

                # Check for bottleneck conditions
                while self.multiprocess and all([move.child_node and move.child_node.awaiting_processing
                                                and not move.child_node.branch_complete
                                                for move in current_node.moves]) and not current_node.branch_complete:
                    print(f"\rBOTTLENECK WARNING: Queue size: {self.process_queue.qsize()} {[move.child_node.awaiting_processing for move in current_node.moves]}", end='')
                    # Clear the queue
                    self.apply_neural_net_results()

                # If the queue has become too fully than wait for it to process... get it down to batch size for the
                # actual call to finish off
                while self.multiprocess and self.process_queue.qsize() > 128:
                    self.apply_neural_net_results()

                if not self.multiprocess and self.training and idx % 30 == 0:
                    self.train_neural_network_local()

                if self.multiprocess:
                    self.apply_neural_net_results()

                if move is None:
                    break

                if all([move.child_node and move.child_node.branch_complete for move in current_node.moves]):
                    current_node.branch_complete = True

            if not current_node.branch_complete and move is not None and visited_moves:
                # If the graph is complete its already been done

                current_node, reward, done = move.expand(thread_env, state=current_node.state)

                if not self.multiprocess:
                    value = self.neural_net.node_evaluation(current_node)
                    current_node.V = value
                    reward += value
                    reward /= 2
                    self.backpropagation(visited_moves, reward)

                else:

                    # Create a hash of the node such that we can relocate this node later
                    node_hash = hash(current_node)

                    if node_hash not in self.process_queue_node_lookup:

                        states = current_node.state
                        legal_moves = current_node.legal_move_strings

                        self.process_queue_node_lookup[node_hash] = (current_node, visited_moves, reward)

                        self.process_queue.put((self.agent_id, node_hash, states, legal_moves))

                        # Set the node is current node is awaiting processing
                        current_node.awaiting_processing = True

                    else:
                        # TODO: handle these better
                        self.undo_informationless_rollout(visited_moves)

            # Now return to root node
            current_node = self.root

        if self.training:
            self.search_for_sufficiently_visited_nodes(self.root)

        if profile:
            profiler.disable()
            profiler.print_stats(sort="cumtime")

    @staticmethod
    def undo_informationless_rollout(visited_moves):
        for move in visited_moves:
            # Undo stats
            move.N -= 1
            move.virtual_loss += 1

    def backpropagation(self, visited_moves, observed_reward: float):

        gamma_factor = 1

        # This is the other player as player who made a the move into the new state is the opposite player to whos turn
        # it is in the state
        player = visited_moves[-1].parent_node.player

        for idx, move in enumerate(reversed(visited_moves)):

            # After doing a need evaluation, we get the other players chance of winning...
            # We therefore flip the score when it us playing

            if move.parent_node.player == player:
                # Flip for when the other player is observing reward
                flip = -1
            else:
                flip = 1

            # Increment the visit count and update the total reward
            new_Q = gamma_factor**idx * flip * observed_reward

            move.W += new_Q

            if self.multiprocess:
                move.virtual_loss += 1

    def search_for_sufficiently_visited_nodes(self, root_node):
        def recursive_search(node, count):

            if node.branch_complete or count > 400:
                return

            self.save_results_to_memory(node, root_node)

            for move in node.moves:
                if move.N >= 100 and move.child_node:
                    recursive_search(move.child_node, count + 1)

            return

        recursive_search(root_node, 0)

    def save_results_to_memory(self, current_node, root_node):

        state = current_node.state
        moves = [move.move for move in current_node.moves]
        visit_counts = [move.N for move in current_node.moves]
        predicted_value = current_node.V

        self.saved_memory_local += 1

        if not self.multiprocess:
            self.memory.save_state_to_moves(state, moves, visit_counts, predicted_value, current_node is root_node)
        else:
            self.experience_queue.put((state, moves, visit_counts, predicted_value, current_node is root_node))

    def apply_neural_net_results(self):
        """
        Process results from the neural network in batches, updating nodes and applying Dirichlet noise.
        """
        max_batch_size = 128  # Process up to 10 items at a time

        results = []

        # Clear the queue
        for _ in range(self.results_queue.qsize()):
            try:
                results = self.results_queue.get_nowait()  # Non-blocking get
            except Empty:
                break

            for (the_hash, value, move_probs, index_map) in results:

                # Retrieve the node associated with this hash
                node, visited_moves, reward = self.process_queue_node_lookup.pop(the_hash)

                # Update node value
                node.V = value

                # Update parent's move total value if applicable
                if node.parent_move is not None:
                    node.parent_move.W += value

                # Parameters for Dirichlet noise
                alpha = 1  # Dirichlet distribution parameter
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
                self.backpropagation(visited_moves, (value + reward)/2)


    def train_neural_network_local(self):
        if len(self.memory) < 32:
            return
        states, moves, probs, wins = self.memory.get_batch(32)
        state, moves, wins, legal_move_mask = self.neural_net.tensorise_batch(states, moves, probs, wins)
        self.neural_net.loss_function(state, target=(wins, moves), legal_move_mask=legal_move_mask)

    def end_game(self, white_win: bool):
        self.memory.end_game(white_win)

    def run(self):
        self.train()

    def train(self, n_games: int = 1000000):

        game_count = 0
        max_length = 300
        sims = 500

        while True:

            # Start a new game

            move_count = 0
            main_board = chess.Board()

            self.reset()

            node = self.root
            winner = None

            while not main_board.is_game_over():

                start_time = time.time()
                self.parallel_search(current_node=node, number_of_expansions=sims)
                end_time = time.time()

                print(f"\n\nTime with parallel search {end_time - start_time:.3f}")

                # TODO: sort out now visiting

                if move_count < 30:
                    tau = 1.
                else:
                    tau = 0.1

                node, move = self.root.exploratory_select_new_root_node(tau=tau)
                chess_move = chess.Move.from_uci(move)
                main_board.set_fen(self.root.state)
                main_board.push(chess_move)

                print(f"Agent {self.agent_id} Game {game_count} Move: {move_count}")
                print("Board:")
                print(main_board)

                print(f"FEN String: {main_board.fen()}")

                if main_board.is_checkmate():
                    print("Checkmate!")
                    winner = main_board.fen().split()[1]
                elif main_board.is_stalemate():
                    print("Stalemate!")
                elif main_board.is_insufficient_material():
                    print("Insufficient material to checkmate!")
                elif move_count >= max_length:
                    print("Maximum move length")
                    break
                else:

                    print(
                        f"Move chosen: {move} Prob: {node.parent_move.P:.3f} Q: {node.parent_move.Q:.3f} N: {node.parent_move.N}")
                    print(f"Game over: {main_board.is_game_over()}")
                    print(f"Memory length: {self.saved_memory_local} Process queue: {self.process_queue.qsize()} Results queue: {self.results_queue.qsize()}")
                    print(f"Tree node complete: {node.branch_complete} Reason: {node.branch_complete_reason}")

                    if main_board.is_check():
                        print("King is in check!")

                    self.root = node

                    # Clear references to the tree above
                    self.root.parent_move = None

                    # Increment move count
                    move_count += 1

                if self.root.branch_complete:
                    break

            if winner == 'w':
                white_win = True
            elif winner == 'b':
                white_win = False
            else:
                white_win = None

            if not self.multiprocess:
                self.end_game(white_win)

            game_count += 1


class Node:

    def __init__(self, state: str , board: chess_moves.ChessEngine =None, parent_move =None):

        # The chess fen string object representing this state
        self.state: str = state
        self.player: str = self.state.split()[1]

        # Save a reference to the parent node
        self.parent_move = parent_move  # Reference to the parent move

        # Generate child nodes for legal moves
        board.set_fen(self.state)
        legal_moves =  board.legal_moves()
        self.number_legal_moves = len(list(legal_moves))

        # Check if we have fully searched the branch
        if self.number_legal_moves == 0:
            self.branch_complete_reason = 'No legal moves found'
            self.branch_complete = True
        else:
            self.branch_complete = False

        # Create edges
        self.moves: List[Move] = []
        created_move_num = len(self.moves)
        for idx, move in enumerate(legal_moves):
            self.moves.append(Move(self, move))

        if self.number_legal_moves < len(self.moves):
            # Get rid of excess moves
            self.moves = self.moves[:self.number_legal_moves]

        # Save whos go it is
        self.team = "white" if self.state.split()[1] == 'w' else "black"

        # State value
        self.V = 0

        # Set a flag saying it has not been processed yet
        self.branch_complete_reason: str = None
        self.awaiting_processing: bool = False

    def puct(self, draw_num: int = None, rng_generator: default_rng = None):
        """
        Select the best move using the PUCT formula,
        ignoring moves that lead to completed branches.
        """
        move_N = np.array([move.N for move in self.moves])
        move_P = np.array([move.P for move in self.moves])
        move_Q = np.array([move.Q for move in self.moves])
        branch_complete = np.array([move.child_node.branch_complete if move.child_node is not None else False for move in self.moves])

        # Ignore completed branches
        valid_moves = np.where(~branch_complete)[0]

        if len(valid_moves) == 0:
            return None

        # Use only valid moves in PUCT calculation
        visits = np.sum(move_N[valid_moves])
        puct_values = (move_Q[valid_moves]) + 3.0 * move_P[valid_moves] * np.sqrt(visits + 1) / (move_N[valid_moves] + 1)

        # Select the move with the maximum PUCT value
        best_value = np.max(puct_values)
        best_indexes = np.where(puct_values == best_value)[0]

        best_index = rng_generator.choice(best_indexes)

        return self.moves[best_index]

    def exploratory_select_new_root_node(self, tau: float = 1.0):
        """
        Select a random child node with a temperature adjusted probability distribution
        :param tau:
        :return:
        """
        Ns = np.array([move.N for move in self.moves])
        N_to_tau = np.power(Ns, 1./tau)
        probs = N_to_tau / np.sum(N_to_tau)
        chosen_move = np.random.choice(self.moves, p=probs)
        return chosen_move.child_node, chosen_move.move

    def greedy_select_new_root_node(self):
        """
        Select child node with the highest Q
        :return:
        """
        Qs = np.array([move.Q for move in self.moves]).astype(np.float32)
        legit_moves = np.array([True if move.N > 0 else False for move in self.moves])
        Qs[~legit_moves] = -np.inf
        q_max = np.max(Qs)
        max_idxs = np.where(Qs==q_max)[0]
        chosen_move = self.moves[np.random.choice(max_idxs)]

        return chosen_move.child_node, chosen_move.move

    @property
    def legal_move_strings(self):
        return [str(move) for move in self.moves]

class Move:

    def __init__(self, parent_node: Node, move: str):

        self.parent_node = parent_node
        self.move = move

        self.child_node = None

        # Move statistics
        self.W = 0.
        self.N = 0.
        self.P = 1.

        # When minimax is turned on we used Q_max rather than Q mean
        # This works under assumption than player will make their best move rather than a random one
        self.Q_max = -np.inf

        # Multi-processing variables
        self.virtual_loss = 0

    def __str__(self):
        return str(self.move)

    def expand(self, board: chess.Board, state: str):

        if state is None:
            state = board.fen()  # Update the board state to the provided FEN string

        # Run the sim to update the board
        new_state = board.push(state, str(self.move))

        # Create a new child node
        self.child_node = Node(new_state, board, parent_move=self)

        # Calculate the reward
        done, result_code = board.is_game_over()

        if done:
            print("\rEnd game found!", end="")
            self.child_node.branch_complete_reason = f'End game found in sim: {result_code}'
            self.child_node.branch_complete = True
            if result_code == 1:
                reward = 1.
            else:
                reward = 0.
        else:
            reward = 0.

        # Update statistic
        return self.child_node, reward, done

    @property
    def Q(self):
        # Virtual loss is the equivalent of
        Q =  (self.W + self.virtual_loss) / self.N if self.N > 0 else 0
        return Q
