import chess
import numpy as np
from numpy.random import default_rng
import time
from multiprocessing import Queue, Process
from queue import Empty
from typing import List, Dict, Optional

from neural_nets.data.fen_chess_puzzles.starting_states import LichessCuriculum
from tree.memory import Memory, Turn
import chess_moves
import os
from enum import Enum
import time
from numba import njit

from util.parallel_error_log import error_logger
from util.parallel_profiler import parallel_profile
from util.chess_functions import save_chess_board_as_im

def all_custom(iterable):
    return len(iterable) > 0 and all(iterable)

class GameOverType(Enum):
    STALEMATE = 0
    CHECKMATE = 1
    NO_TAKES_DRAW = 2

@njit(cache=True)
def fast_puct(Qs, Ns, Ps, virtual_losses) -> int:
    """ numba compatible puct function. Outside class to only compile once"""

    visits = np.sum(Ns)

    Qs = Qs + virtual_losses
    Qs[Ns == 0] = 0

    puct_values = Qs + 3.0 * Ps * np.sqrt(visits + 1) / (Ns + 1)

    best_value = np.max(puct_values)
    best_indexes = np.where(puct_values == best_value)[0]

    best_index = best_indexes[np.random.randint(len(best_indexes))]

    if best_index is None:
        print(Qs)

    return best_index


class Node:

    def __init__(self, state: str, board: chess_moves.ChessEngine, move_idx: int = None , parent_node: "Node" =None):

        # The chess fen string object representing this state
        self.state: str = state
        self.player: str = self.state.split()[1]

        # Save a reference to the parent node
        self.parent_node: Node = parent_node  # Reference to the parent move
        self.move_idx: int = move_idx # Save the index of the move which created us

        # Generate child nodes for legal moves
        board.set_fen(self.state)
        legal_moves: List[str] =  board.legal_moves()
        self.number_legal_moves: int = len(legal_moves)

        # Check if we have fully searched the branch
        if self.number_legal_moves == 0:
            self.branch_complete_reason: str = 'No legal moves found'
            self.branch_complete = True
        else:
            self.branch_complete = False

        self.game_won = False
        self.game_over_type: GameOverType = None

        # Create edges
        self.moves = legal_moves
        self.Qs = np.zeros(self.number_legal_moves)
        self.Ns = np.zeros(self.number_legal_moves)
        self.Ps = np.ones(self.number_legal_moves) / self.number_legal_moves
        self.virtual_losses = np.zeros(self.number_legal_moves)
        self.child_nodes: Dict[int, Node] = {}

        # Save whose go it is
        self.team = Turn.WHITE if self.state.split()[1] == 'w' else Turn.BLACK

        # State value
        self.V = 0
        self.R = 0

        # Set a flag saying it has not been processed yet
        self.branch_complete_reason: str = None
        self.awaiting_processing: bool = False

    def puct(self) -> int :
        if self.number_legal_moves == 1:
            return 0

        move_idx  = fast_puct(self.Qs, self.Ns, self.Ps, self.virtual_losses)
        return move_idx

    def expand(self, move_idx: int, board: chess.Board, state: str) -> ("Node", bool, int):

        if state is None:
            state = board.fen()  # Update the board state to the provided FEN string

        # Run the sim to update the board
        new_state: str = board.push(state, self.moves[move_idx])

        # Create a new child node
        self.child_nodes[move_idx] = Node(new_state, move_idx=move_idx, board=board, parent_node=self)

        # Calculate the reward
        done, result_code = board.is_game_over()

        if done:

            self.child_nodes[move_idx].branch_complete_reason = f'End game found in sim: {result_code}'
            self.child_nodes[move_idx].branch_complete = True

            # This corresponds to the output codes from chess engine
            self.child_nodes[move_idx].game_over_type = GameOverType(result_code)

            if result_code == 1:
                reward = -1.
                # It's our turn and we won the game
                self.child_nodes[move_idx].game_won = True
            else:
                reward = 0.
        else:
            reward = 0.

        # Save the reward
        self.child_nodes[move_idx].R = reward

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
        probs[self.Ns == 0] = 0 # Assert this... was occasionly getting an error in which it chooses unvisited
        chosen_move = np.random.choice(len(self.moves), p=probs)

        return self.child_nodes[chosen_move], self.moves[chosen_move], chosen_move

    def greedy_select_new_root_node(self):
        """
        Select child node with the highest Q
        :return:
        """
        local_Qs = np.copy(self.Qs)
        legit_moves = self.Ns > 0
        local_Qs[~legit_moves] = -np.inf
        q_max = np.max(local_Qs)
        max_idxs = np.where(local_Qs==q_max)[0]
        chosen_move = np.random.choice(max_idxs)
        return self.child_nodes[chosen_move], self.moves[chosen_move], chosen_move

    @property
    def legal_move_strings(self):
        return [str(move) for move in self.moves]


class GameTree(Process):

    add_dirichlet_noise = True
    save_non_root_states = False
    save_images_of_checkmates = True
    full_rollout_virtual_loss = False
    curiculum = True
    maximum_lag = 8

    agent_count = 0

    def __init__(self, save_dir: str, neural_net = None, training: bool =  False,
                 multiprocess: bool = False, process_queue: Queue = None,
                 experience_queue: Queue = None, results_queue: Queue = None,
                 data_queue: Queue = None, start_state: str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"):

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
        self.start_state = str(start_state)

        # Set up the processing stat
        self.number_sent = 0
        self.number_returned = 0

        self.curiculum_generator = None if not self.curiculum \
            else LichessCuriculum('/home/dom/Code/chess_bot/neural_nets/data/fen_chess_puzzles/lichess_db_puzzle_sorted.csv')
        print("Lichess Curiculum loaded")

    def reset(self, start_state: str = None):
        if start_state is None:
            start_state = self.start_state
        self.root = Node(state=start_state, board=chess_moves.ChessEngine())

    def parallel_search(self, current_node: Node = None, number_of_expansions: int = 1000):

        end_nodes: List[Node] = []

        if current_node is None:
            current_node: Node = self.root

        # Initialise the threads env
        thread_env = chess_moves.ChessEngine()

        for idx in range(number_of_expansions):

            # Thread num is used a draw breaker for early iterations such that they don't search down the same branch
            visited_moves: List[List[Node, int]] = []
            move_has_child_node = True

            # Apply last round in case they have returned
            self.apply_neural_net_results()

            # Explore to leaf
            while move_has_child_node and not current_node.branch_complete:

                # Select
                move_idx = current_node.puct()
                visited_moves.append([current_node, move_idx])

                # Statistics
                current_node.Ns[move_idx] += 1

                # Set virtual loss to discourage search down here for a bit
                if self.multiprocess and self.full_rollout_virtual_loss:
                    current_node.virtual_losses[move_idx] -= 1.

                if move_idx in current_node.child_nodes:
                    # Get new node and then a new move
                    current_node = current_node.child_nodes[move_idx]

                else:
                    # We have found a leaf move
                    move_has_child_node = False

                # Record some time waiting data
                pre_check_time = time.perf_counter_ns()
                self.time_waiting += (time.perf_counter_ns() - pre_check_time) / 1e9

                if not self.multiprocess and self.training and idx % 30 == 0:
                    self.train_neural_network_local()

                if self.multiprocess:
                    self.apply_neural_net_results()

                if move_idx is None:
                    break
                
            if not current_node.branch_complete and move_idx is not None and visited_moves:

                # Set virtual loss to discourage search down here for a bit
                if self.multiprocess and not self.full_rollout_virtual_loss:
                    current_node.virtual_losses[move_idx] -= 1.
                    #  If are currently exploring all down here then get
                    if current_node.parent_node is not None and np.all(current_node.virtual_losses < 0):
                        current_node.parent_node.virtual_losses[current_node.move_idx] = -1

                # If the graph is complete its already been done
                current_node, reward, done = current_node.expand(move_idx, thread_env, state=current_node.state)

                if not self.multiprocess:
                    value = self.neural_net.node_evaluation(current_node)
                    current_node.V = value
                    reward += value
                    reward /= 2
                    self.backpropagation(visited_moves, reward)

                else:

                    if reward != 0:
                        # Send reward signal up the tree
                        self.backpropagation(visited_moves, reward, undo_v_loss=False)

                    # Create a hash of the node such that we can relocate this node later
                    node_hash = hash(current_node)

                    if node_hash not in self.process_queue_node_lookup:

                        states = current_node.state
                        legal_moves = current_node.legal_move_strings

                        self.process_queue_node_lookup[node_hash] = (current_node, visited_moves, reward)

                        self.process_queue.put((self.agent_id, node_hash, states, legal_moves))

                        # Increment the number of nodes sent for processing
                        self.number_sent += 1

                        # Set the node is current node is awaiting processing
                        current_node.awaiting_processing = True

                    else:
                        # TODO: handle these better
                        self.undo_informationless_rollout(visited_moves)

                if current_node.branch_complete and current_node.game_over_type is not GameOverType.NO_TAKES_DRAW:
                    end_nodes.append(current_node)

            # Now return to root node
            current_node = self.root

        # Wait until the end of the rollout and then save any games which finished.
        if self.training:
            for node in end_nodes:
                self.save_path_whilst_optimal(node)

    @staticmethod
    def undo_informationless_rollout(visited_moves):
        for node, move_idx in visited_moves:
            # Undo stats
            node.Ns[move_idx] -= 1
            node.virtual_losses[move_idx] += 1

    def backpropagation(self, visited_moves, observed_reward: float, undo_v_loss: bool = True):

        # New incoming state. If win... reward = -1 for current player as they have lost
        gamma_factor = 0.99

        # This is the other player as player who made a the move into the new state is the opposite player to whos turn
        # it is in the state
        reward = -observed_reward

        for idx, (node, move_idx) in enumerate(reversed(visited_moves)):

            # Increment the visit count and update the total reward
            new_Q = node.R + gamma_factor**idx * reward

            node.Qs[move_idx] = new_Q

            # Player will choose their best move... maximum Q
            Qs = node.Qs[node.Ns>0]
            node.V = np.max(Qs)

            # Only the best rewards propagate up the tree... otherwise player would not choose them
            reward = -np.copy(node.V)

            if self.full_rollout_virtual_loss and self.multiprocess and undo_v_loss:
                # For full rollouts add one back on... some will have mutliple V losses at the same time
                node.virtual_losses[move_idx] += 1
            elif self.multiprocess and undo_v_loss:
                # Set all to zero if not using full rollouts
                node.virtual_losses[move_idx] = 0

    def search_for_sufficiently_visited_nodes(self, root_node):
        self.recursive_search(root_node, 0, root_node)


    def save_path_whilst_optimal(self, start_node: Node):
        """ Find games in which both players have played optimally at least for two moves"""
        game_states: List[dict] = []


        # Get game winner
        if start_node.game_won:
            winner = start_node.parent_node.player
        else:
            winner = None

        # First playable state is the parent node
        current_node = start_node.parent_node

        # Iterate up the tree and save the states for processing
        while current_node.parent_node is not None:

            best_parent_move = current_node.parent_node.Qs.argmax()

            if best_parent_move != current_node.move_idx:
                break

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

        # Make sure its even to not get bias to wins
        if len(game_states) % 2 != 0:
            game_states = game_states[:len(game_states) // 2]

        if len(game_states) == 0:
            return

        if not self.multiprocess:
            self.memory.save_game_to_memory((game_states, self.agent_id))
        else:
            self.experience_queue.put((game_states, self.agent_id))


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

        # Clear the queue
        while True:
            try:
                results = self.results_queue.get_nowait()  # Non-blocking get
            except Empty:
                results = []

            for (the_hash, value, move_probs, index_map) in results:

                # Increment number returned
                self.number_returned += 1

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
                self.backpropagation(visited_moves, value + reward)

            if self.number_sent - self.number_returned < self.maximum_lag:
                #  If we have a processing lag of less than the maximum allowed lag then we can continue
                return



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
        sims = 1000

        print(f"Starting training run on agent {self.agent_id}")

        while True:

            # Start a new game
            move_count = 0
            main_board = chess.Board()

            if self.curiculum:
                # This has some progressively more difficult chess puzzles
                start_state, max_length = self.curiculum_generator.get_start_state(game_count)
                move_count = int(start_state.split()[-1])
                max_length = 300
            else:
                start_state = self.start_state
                max_length  = 300

            # Reset main board to start state
            self.reset(start_state)
            main_board.set_fen(start_state)

            node = self.root
            player_check_mated = None

            while not main_board.is_game_over():

                start_time = time.time()
                self.parallel_search(current_node=node, number_of_expansions=sims)
                end_time = time.time()

                print(f"\n\nTime with parallel search {end_time - start_time:.3f}s. {self.time_waiting:.3f} "
                      f"({100. *self.time_waiting / (end_time - start_time):.3f}%) spent waiting)")
                self.time_waiting = 0.

                # TODO: sort out now visiting
                if move_count > 30 or (self.curiculum and game_count < 250):
                    tau = 0.1
                else:
                    tau = 1.

                node, move, move_idx = self.root.exploratory_select_new_root_node(tau=tau)
                chess_move = chess.Move.from_uci(move)
                main_board.set_fen(self.root.state)
                main_board.push(chess_move)

                print(f"Agent {self.agent_id} Game {game_count} Move: {move_count}")
                # print("Board:")
                # print(main_board)
                #
                # print(f"FEN String: {main_board.fen()}")

                if main_board.is_checkmate():

                    player_check_mated = main_board.fen().split()[1]

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

                    # print(
                    #     f"Move chosen: {move} Prob: {node.parent_node.Ps[move_idx]:.3f} Q: {node.parent_node.Qs[move_idx]:.3f} N: {node.parent_node.Ns[move_idx]}")
                    # print(f"Game over: {main_board.is_game_over()}")
                    # print(f"Memory length: {self.saved_memory_local} Process queue: {self.process_queue.qsize()} Results queue: {self.results_queue.qsize()}")
                    # print(f"Tree node complete: {node.branch_complete} Reason: {node.branch_complete_reason}")

                    if main_board.is_check():
                        print("King is in check!")

                    self.update_node_in_training(node, move_idx)

                    # Increment move count
                    move_count += 1

                # Here we search down the tree using the best moves and see if there is a checkmate
                if self.root.branch_complete:
                    self.save_results_to_memory(self.root)
                    break

            if player_check_mated == 'w':
                white_win = False
            elif player_check_mated == 'b':
                white_win = True
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


