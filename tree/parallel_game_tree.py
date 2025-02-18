import chess
import numpy as np
from numpy.random import default_rng
import time
from multiprocessing import Queue, Process
from queue import Empty
from typing import List, Dict
from tree.memory import Memory, Turn
import chess_moves
import os
from enum import Enum
import time

from util.parallel_error_log import error_logger
from util.parallel_profiler import parallel_profile
from util.chess_functions import save_chess_board_as_im

def all_custom(iterable):
    return len(iterable) > 0 and all(iterable)

class GameOverType(Enum):
    STALEMATE = 0
    CHECKMATE = 1
    NO_TAKES_DRAW = 2


class Node:

    def __init__(self, state: str , board: chess_moves.ChessEngine =None, parent_node =None):

        # The chess fen string object representing this state
        self.state: str = state
        self.player: str = self.state.split()[1]

        # Save a reference to the parent node
        self.parent_node = parent_node  # Reference to the parent move

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

        self.game_won = False
        self.game_over_type: GameOverType = None

        # Create edges
        self.moves = legal_moves
        self.Ws = np.zeros(len(self.moves))
        self.Ns = np.zeros(len(self.moves))
        self.Ps = np.zeros(len(self.moves))
        self.virtual_losses = np.zeros(len(self.moves))
        self.child_nodes: Dict[int, Node] = {}

        # Save whose go it is
        self.team = Turn.WHITE if self.state.split()[1] == 'w' else Turn.BLACK

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
        if self.number_legal_moves == 1:
            # Save some compute
            return 0, self.moves[0]

        branch_complete = np.zeros(len(self.moves), dtype=np.bool_)

        for move_idx, child_node in self.child_nodes.items():
            branch_complete[move_idx] = child_node.branch_complete

        # Ignore completed branches
        valid_moves = np.where(~branch_complete)[0]

        if sum(valid_moves) == 0:
            return None, None

        # Use only valid moves in PUCT calculation
        visits = np.sum(self.Ns[valid_moves])

        Qs = self.Ws/self.Ns
        Qs[self.Ns==0] = 0
        Qs[~valid_moves] = -np.inf

        puct_values = Qs + 3.0 * self.Ps * np.sqrt(visits + 1) / (self.Ns + 1)

        # Select the move with the maximum PUCT value
        best_value = np.max(puct_values)
        best_indexes = np.where(puct_values == best_value)[0]

        best_index = rng_generator.choice(best_indexes)

        return best_index, self.moves[best_index]

    def expand(self, move_idx: int, board: chess.Board, state: str):

        if state is None:
            state = board.fen()  # Update the board state to the provided FEN string

        # Run the sim to update the board
        new_state: str = board.push(state, self.moves[move_idx])

        # Create a new child node
        self.child_nodes[move_idx] = Node(new_state, board, parent_node=self)

        # Calculate the reward
        done, result_code = board.is_game_over()

        if done:

            print("\rEnd game found!", end="")
            self.child_nodes[move_idx].branch_complete_reason = f'End game found in sim: {result_code}'
            self.child_nodes[move_idx].branch_complete = True

            # This corresponds to the output codes from chess engine
            self.child_nodes[move_idx].game_over_type = GameOverType(result_code)

            if result_code == 1:
                reward = 1.
                # It's our turn and we won the game
                self.child_nodes[move_idx].game_won = True
            else:
                reward = 0.
        else:
            reward = 0.

        # Update statistic
        return self.child_nodes[move_idx], reward, done

    def exploratory_select_new_root_node(self, tau: float = 1.0):
        """
        Select a random child node with a temperature adjusted probability distribution
        :param tau:
        :return:
        """
        N_to_tau = np.power(self.Ns, 1./tau)
        probs = N_to_tau / np.sum(N_to_tau)
        probs[self.Ns == 0] = 0 # Assert this... was occassioanlly getting an error in which it chooses unvisited
        chosen_move = np.random.choice(len(self.moves), p=probs)

        if chosen_move not in self.child_nodes:
            print(self.Ns, chosen_move, self.moves, self.child_nodes, probs)

        return self.child_nodes[chosen_move], self.moves[chosen_move], chosen_move

    def greedy_select_new_root_node(self):
        """
        Select child node with the highest Q
        :return:
        """
        Qs = self.Ws/self.Ns
        legit_moves = self.Ns > 0
        Qs[~legit_moves] = -np.inf
        q_max = np.max(Qs)
        max_idxs = np.where(Qs==q_max)[0]
        chosen_move = np.random.choice(max_idxs)
        return self.child_nodes[chosen_move], self.moves[chosen_move], chosen_move

    @property
    def legal_move_strings(self):
        return [str(move) for move in self.moves]

    @property
    def Qs(self):
        return self.Ws/self.Ns


class GameTree(Process):

    add_dirichlet_noise = True
    save_non_root_states = False
    save_images_of_checkmates = True

    agent_count = 0

    def __init__(self, save_dir: str, neural_net = None, training: bool =  False,
                 multiprocess: bool = False, process_queue: Queue = None,
                 experience_queue: Queue = None, results_queue: Queue = None,
                 data_queue: Queue = None):

        super().__init__()

        self.agent_id = GameTree.agent_count
        GameTree.agent_count += 1

        self.save_dir = save_dir
        self.log_file_name = os.path.join(save_dir, f"{self.__class__.__name__}_{self.agent_id}.txt")

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
            self.data_queue = data_queue

        # Initiate root
        self.root = None
        self.time_waiting = 0.

    def reset(self):
        self.root = Node(state="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", board=chess_moves.ChessEngine())

    def parallel_search(self, current_node: Node = None, number_of_expansions: int = 1000):

        if current_node is None:
            current_node = self.root

        # Sharing a random number generator SEVERELY slows down the process... explaination pending
        thread_rng = default_rng()

        # Initialise the threads env
        thread_env = chess_moves.ChessEngine()

        for idx in range(number_of_expansions):

            # Thread num is used a draw breaker for early iterations such that they don't search down the same branch
            visited_moves = []
            move_has_child_node = True

            # Explore to leaf
            while move_has_child_node and not current_node.branch_complete:

                # Select
                move_idx, new_move = current_node.puct(rng_generator=thread_rng)
                visited_moves.append([current_node, move_idx])

                # Statistics
                current_node.Ns[move_idx] += 1

                # Set virtual loss to discourage search down here for a bit
                if self.multiprocess:
                    current_node.virtual_losses[move_idx] -= 1.

                if move_idx in current_node.child_nodes:
                    # Get new node and then a new move
                    current_node = current_node.child_nodes[move_idx]

                else:
                    # We have found a leaf move
                    move_has_child_node = False

                pre_check_time = time.perf_counter_ns()
                # Check for bottleneck conditions: despite virtual loss we have reached a state in which all moves are awaiting processing
                if self.multiprocess and all_custom([move_idx in current_node.child_nodes and current_node.child_nodes[move_idx].awaiting_processing
                                     and not current_node.branch_complete for move_idx in range(len(current_node.moves))]):

                    print(f"\rBOTTLENECK WARNING: Queue size: {self.process_queue.qsize()} "
                          f"{[current_node.child_nodes[move_idx].awaiting_processing for move_idx in range(len(current_node.moves))]}")

                    while all_custom([current_node.child_nodes[move_idx].awaiting_processing for move_idx in range(len(current_node.moves))]):
                        self.apply_neural_net_results()

                # If the queue has become too fully than wait for it to process... get it down to batch size for the
                # actual call to finish off
                while self.multiprocess and self.process_queue.qsize() > 128:
                    self.apply_neural_net_results()

                self.time_waiting += (time.perf_counter_ns() - pre_check_time) / 1e9

                if not self.multiprocess and self.training and idx % 30 == 0:
                    self.train_neural_network_local()

                if self.multiprocess:
                    self.apply_neural_net_results()

                if move_idx is None:
                    break

                if all([child_node_idx in current_node.child_nodes
                        and current_node.child_nodes[child_node_idx].branch_complete
                        for child_node_idx in range(len(current_node.moves))]):
                    current_node.branch_complete = True

            if not current_node.branch_complete and move_idx is not None and visited_moves:

                # If the graph is complete its already been done
                current_node, reward, done = current_node.expand(move_idx, thread_env, state=current_node.state)

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

                # Save checkmates and stalemate games to memory. Got rid of no takes draws as they just spam the memory
                # as quite often when you get close to 50 no takes there are tens of differnt possible games which lead
                # a draw
                if self.training and done and current_node.game_over_type != GameOverType.NO_TAKES_DRAW:
                    # We have found a full game... save result to memory
                    self.save_results_to_memory(current_node)

            # Now return to root node
            current_node = self.root

    @staticmethod
    def undo_informationless_rollout(visited_moves):
        for node, move_idx in visited_moves:
            # Undo stats
            node.Ns[move_idx] -= 1
            node.virtual_losses[move_idx] += 1

    def backpropagation(self, visited_moves, observed_reward: float):

        gamma_factor = 1

        # This is the other player as player who made a the move into the new state is the opposite player to whos turn
        # it is in the state
        player = visited_moves[-1][0].player

        for idx, (node, move_idx) in enumerate(reversed(visited_moves)):

            # After doing a need evaluation, we get the other players chance of winning...
            # We therefore flip the score when it us playing

            if node.player == player:
                # Flip for when the other player is observing reward
                flip = -1
            else:
                flip = 1

            # Increment the visit count and update the total reward
            new_Q = gamma_factor**idx * flip * observed_reward

            node.Ws[move_idx] += new_Q

            if self.multiprocess:
                node.virtual_losses[move_idx] += 1

    def search_for_sufficiently_visited_nodes(self, root_node):
        self.recursive_search(root_node, 0, root_node)

    def recursive_search(self, node, count, root_node):
        """
        Recursively searches for sufficiently visited nodes
        """

        if node.branch_complete or count > 400:
            return

        self.save_results_to_memory(node)

        for move_idx in range(len(node.moves)):
            if node.Ns[move_idx] >= 100 and move_idx in node.child_nodes:
                self.recursive_search(node.child_nodes[move_idx], count + 1)

        return

    def save_results_to_memory(self, current_node: Node):

        if current_node.game_won:
            winner = current_node.parent_node.player
        else:
            winner = None

        game_states = []

        # Traverse up the tree and save game states, visit counts, and values
        current_node = current_node.parent_node

        while current_node.parent_node is not None:

            state = current_node.state
            moves = current_node.moves
            visit_counts = current_node.Ns

            if winner is not None:
                if current_node.player == winner:
                    value = 1
                else:
                    value = -1
            else:
                value = 0

            game_states.append({'state': state, 'moves': moves, 'visit_counts': visit_counts, 'value': value})
            self.saved_memory_local += 1
            current_node = current_node.parent_node

        if not self.multiprocess:
            self.memory.save_game_to_memory((game_states, self.agent_id))
        else:
            self.experience_queue.put((game_states, self.agent_id))

    def apply_neural_net_results(self):
        """
        Process results from the neural network in batches, updating nodes and applying Dirichlet noise.
        """
        batch_processed = False
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
                for move_idx, prob in zip(range(len(node.moves)), Ps):
                    node.Ps[move_idx] = prob

                node.awaiting_processing = False
                self.backpropagation(visited_moves, (value + reward)/2)
                batch_processed = True

        return batch_processed

    def train_neural_network_local(self):
        if len(self.memory) < 32:
            return
        states, moves, probs, wins = self.memory.get_batch(32)
        state, moves, wins, legal_move_mask = self.neural_net.tensorise_batch(states, moves, probs, wins)
        self.neural_net.loss_function(state, target=(wins, moves), legal_move_mask=legal_move_mask)

    def update_node_in_training(self, new_node: Node, action: int):
        """
        In training, we remember old visited states to update when we finish games
        """
        # Clear references to other parts of tree
        self.root.child_nodes = {action: new_node}

        # Update root node
        self.root = new_node

    def end_game(self, white_win: bool):
        self.memory.end_game(white_win)

    def run(self):
        self.train()

    @error_logger
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

                print(f"\n\nTime with parallel search {end_time - start_time:.3f}s. {self.time_waiting:.3f} "
                      f"({100. *self.time_waiting / (end_time - start_time)}%) spent waiting)")
                self.time_waiting = 0.

                # TODO: sort out now visiting

                if move_count < 30:
                    tau = 1.
                else:
                    tau = 0.1

                node, move, move_idx = self.root.exploratory_select_new_root_node(tau=tau)
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

                    if self.save_images_of_checkmates:
                        save_chess_board_as_im(main_board, f"{self.save_dir}/agent_{self.agent_id}_game_{game_count}.svg",
                                               move=chess_move)

                elif main_board.is_stalemate():
                    print("Stalemate!")

                elif main_board.is_insufficient_material():
                    print("Insufficient material to checkmate!")

                elif move_count >= max_length:
                    print("Maximum move length")
                    break
                else:

                    print(
                        f"Move chosen: {move} Prob: {node.parent_node.Ps[move_idx]:.3f} Q: {node.parent_node.Qs[move_idx]:.3f} N: {node.parent_node.Ns[move_idx]}")
                    print(f"Game over: {main_board.is_game_over()}")
                    print(f"Memory length: {self.saved_memory_local} Process queue: {self.process_queue.qsize()} Results queue: {self.results_queue.qsize()}")
                    print(f"Tree node complete: {node.branch_complete} Reason: {node.branch_complete_reason}")

                    if main_board.is_check():
                        print("King is in check!")

                    self.update_node_in_training(node, move_idx)

                    # Increment move count
                    move_count += 1

                if self.root.branch_complete:
                    self.save_results_to_memory(self.root)
                    break

            if winner == 'w':
                white_win = True
            elif winner == 'b':
                white_win = False
            else:
                white_win = None

            if not self.multiprocess:
                self.end_game(white_win)

            if self.data_queue is not None:
                self.send_update_to_main_thread(white_win, move_count)

            game_count += 1

    def send_update_to_main_thread(self, white_win: bool, game_length: int):
        game_data_dict = {'white_win': white_win, 'game_length': game_length}
        self.data_queue.put_nowait(game_data_dict)


