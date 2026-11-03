import chess
import numpy as np
import time
from multiprocessing import Queue, Process, Event, synchronize
from queue import Empty
from typing import List, Dict, Optional, Tuple

from neural_nets.data.fen_chess_puzzles.curriculum import LichessCurriculum
from util.config import SearchConfig
from zero_style_chess_engine.messages import EvalRequest, EvalResult, GameRecord, GameStats, PositionRecord
import chess_moves
import os
from enum import Enum
from numba import njit

from util.parallel_error_log import error_logger
from util.chess_functions import save_chess_board_as_im


class GameOverType(Enum):
    STALEMATE = 0
    CHECKMATE = 1
    NO_TAKES_DRAW = 2


@njit(cache=True)
def fast_puct(Qs: np.ndarray, Ns: np.ndarray, Ps: np.ndarray, virtual_losses: np.ndarray, c_puct: float) -> int:
    """numba compatible puct function. Outside class to only compile once"""

    visits = np.sum(Ns)

    Qs = Qs + virtual_losses
    Qs[Ns == 0] = 0

    puct_values = Qs + c_puct * Ps * np.sqrt(visits + 1) / (Ns + 1)

    best_value = np.max(puct_values)
    best_indexes = np.where(puct_values == best_value)[0]

    best_index = best_indexes[np.random.randint(len(best_indexes))]

    return best_index


class Node:
    def __init__(
        self,
        state: str,
        board: chess_moves.ChessEngine,
        move_idx: Optional[int] = None,
        parent_node: Optional["Node"] = None,
    ) -> None:

        # The chess fen string object representing this state
        self.state: str = state
        self.player: str = self.state.split()[1]

        # Save a reference to the parent node
        self.parent_node: Optional[Node] = parent_node  # Reference to the parent move
        self.move_idx: Optional[int] = move_idx  # Save the index of the move which created us

        # Generate child nodes for legal moves
        board.set_fen(self.state)
        legal_moves: List[str] = board.legal_moves()
        self.number_legal_moves: int = len(legal_moves)

        # Check if we have fully searched the branch
        if self.number_legal_moves == 0:
            self.branch_complete_reason: Optional[str] = "No legal moves found"
            self.branch_complete = True
        else:
            self.branch_complete_reason = None
            self.branch_complete = False

        self.game_won = False
        self.game_over_type: Optional[GameOverType] = None

        # Create edges
        self.moves = legal_moves
        self.Qs = np.zeros(self.number_legal_moves)
        self.Ns = np.zeros(self.number_legal_moves)
        self.Ps = np.ones(self.number_legal_moves) / self.number_legal_moves
        self.virtual_losses = np.zeros(self.number_legal_moves)
        self.child_nodes: Dict[int, Node] = {}

        # State value
        self.V: float = 0
        self.R: float = 0

        # Set a flag saying it has not been processed yet
        self.awaiting_processing: bool = False

    def puct(self, c_puct: float) -> int:
        if self.number_legal_moves == 1:
            return 0

        move_idx = fast_puct(self.Qs, self.Ns, self.Ps, self.virtual_losses, c_puct)
        return move_idx

    def expand(self, move_idx: int, board: chess_moves.ChessEngine, state: str) -> Tuple["Node", float, bool]:

        if state is None:
            state = board.fen()  # Update the board state to the provided FEN string

        # Run the sim to update the board
        new_state: str = board.push(state, self.moves[move_idx])

        # Create a new child node
        self.child_nodes[move_idx] = Node(new_state, move_idx=move_idx, board=board, parent_node=self)

        # Calculate the reward
        done, result_code = board.is_game_over()

        if done:
            self.child_nodes[move_idx].branch_complete_reason = f"End game found in sim: {result_code}"
            self.child_nodes[move_idx].branch_complete = True

            # This corresponds to the output codes from chess engine
            self.child_nodes[move_idx].game_over_type = GameOverType(result_code)

            if result_code == 1:
                reward = -1.0
                # It's our turn and we won the game
                self.child_nodes[move_idx].game_won = True
            else:
                reward = 0.0
        else:
            reward = 0.0

        # Save the reward
        self.child_nodes[move_idx].R = reward

        # Update statistic
        return self.child_nodes[move_idx], reward, done

    def exploratory_select_new_root_node(self, tau: float = 1.0) -> Tuple["Node", str, int]:
        """
        Select a random child node with a temperature adjusted probability distribution
        :param tau:
        :return:
        """
        N_to_tau = np.power(self.Ns, 1.0 / tau)
        probs = N_to_tau / np.sum(N_to_tau)
        probs[self.Ns == 0] = 0  # Assert this... was occasionly getting an error in which it chooses unvisited
        chosen_move = np.random.choice(len(self.moves), p=probs)

        return self.child_nodes[chosen_move], self.moves[chosen_move], chosen_move

    def greedy_select_new_root_node(self) -> Tuple["Node", str, int]:
        """
        Select child node with the highest Q
        :return:
        """
        local_Qs = np.copy(self.Qs)
        legit_moves = self.Ns > 0
        local_Qs[~legit_moves] = -np.inf
        q_max = np.max(local_Qs)
        max_idxs = np.where(local_Qs == q_max)[0]
        chosen_move = np.random.choice(max_idxs)
        return self.child_nodes[chosen_move], self.moves[chosen_move], chosen_move

    @property
    def legal_move_strings(self) -> List[str]:
        return [str(move) for move in self.moves]


class GameTree(Process):
    save_images_of_checkmates = False
    full_rollout_virtual_loss = False
    lag_wait_timeout_seconds = 0.1

    def __init__(
        self,
        save_dir: str,
        agent_id: int = 0,
        training: bool = False,
        eval_request_queue: Optional[Queue] = None,
        game_record_queue: Optional[Queue] = None,
        eval_result_queue: Optional[Queue] = None,
        stats_queue: Optional[Queue] = None,
        start_state: str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        verbose: bool = False,
        inference_mode: bool = False,
        infering_mode_queue: Optional[Queue] = None,
        send_results_queue: Optional[Queue] = None,
        stop_event: Optional[synchronize.Event] = None,
        sims: int = 1000,
        maximum_lag: int = 8,
        curriculum: Optional[LichessCurriculum] = None,
        search_config: Optional[SearchConfig] = None,
    ) -> None:

        super().__init__()

        self.stop_event = stop_event if stop_event is not None else Event()

        # Key of this agent's result queue in the evaluator
        self.agent_id = agent_id

        self.save_dir = save_dir
        self.log_file_name = os.path.join(save_dir, f"{self.__class__.__name__}_{self.agent_id}.txt")

        # Training switch
        self.training = training

        # Search budget per move and how many evaluations may be outstanding before we wait for the evaluator
        self.sims = sims
        self.maximum_lag = maximum_lag

        # c_puct, gamma, Dirichlet noise, game length and temperature schedule
        self.search_config = search_config if search_config is not None else SearchConfig()

        # Local count for the number of memories we've saved
        self.saved_memory_local = 0

        # Link the queues
        self.environment: Optional[chess_moves.ChessEngine] = None
        self.eval_request_queue = eval_request_queue

        # Nodes awaiting evaluation, keyed by node id. Ids come from a counter that is never reset, so a result for a
        # node from a previous game can't be mistaken for one in the current game
        self.pending_evaluations: Dict[int, Tuple[Node, List[Tuple[Node, int]], float]] = {}
        self.next_node_id = 0

        self.game_record_queue = game_record_queue
        self.eval_result_queue = eval_result_queue
        self.stats_queue = stats_queue

        self.inference_mode: bool = inference_mode
        if self.inference_mode:
            self.inference_mode_queue: Optional[Queue] = infering_mode_queue
            self.send_results_queue: Optional[Queue] = send_results_queue

        # Initiate root (set by reset below)
        self.root: Node
        self.start_state = str(start_state)

        # Set up the processing stat
        self.number_sent = 0
        self.number_returned = 0

        # Set verbose
        self.verbose = verbose

        # Puzzle curriculum for the first games; None plays every game from start_state
        self.curriculum = curriculum

        # Reset the engine
        self.reset()

    def reset(self, start_state: Optional[str] = None) -> None:
        if start_state is None:
            start_state = self.start_state
        self.root = Node(state=start_state, board=chess_moves.ChessEngine())

    def parallel_search(self, current_node: Optional[Node] = None, number_of_expansions: int = 1000) -> None:

        end_nodes: List[Node] = []

        #  Initialise environment if not yet done so
        if self.environment is None:
            self.environment = chess_moves.ChessEngine()

        # Set the root node to the opening state
        if current_node is None:
            current_node = self.root

        # Perform N rollouts
        for idx in range(number_of_expansions):
            if self.stop_event.is_set():
                break

            if self.verbose:
                print(f"\rMove = {idx}", end="")

            # Thread num is used a draw breaker for early iterations such that they don't search down the same branch
            visited_moves: List[Tuple[Node, int]] = []
            move_has_child_node: bool = True

            # Explore to leaf
            while move_has_child_node and not current_node.branch_complete:
                # Select a move
                move_idx = current_node.puct(self.search_config.c_puct)
                visited_moves.append((current_node, move_idx))

                # Statistics
                current_node.Ns[move_idx] += 1

                # Set virtual loss to discourage search down here for a bit
                if self.full_rollout_virtual_loss:
                    current_node.virtual_losses[move_idx] -= 1.0

                if move_idx in current_node.child_nodes:
                    # Get new node and then a new move
                    current_node = current_node.child_nodes[move_idx]
                else:
                    # We have found a leaf move
                    move_has_child_node = False

                # Collect data from inference server
                self.apply_neural_net_results()

            # Completed rollout processing
            if not current_node.branch_complete and visited_moves:
                # Set virtual loss to discourage search down here for a bit
                if not self.full_rollout_virtual_loss:
                    last_move_idx: int = visited_moves[-1][1]
                    current_node.virtual_losses[last_move_idx] -= 1.0
                    #  If are currently exploring all the children then discourage next rollout from following us
                    #  NOTE: this indicates your evaluation is far too slow
                    if current_node.parent_node is not None and np.all(current_node.virtual_losses < 0):
                        current_node.parent_node.virtual_losses[current_node.move_idx] = -1

                # If the graph is complete its already been done
                current_node, reward, done = current_node.expand(move_idx, self.environment, state=current_node.state)

                if reward != 0:
                    # If this move had reward... we don't need to wait until the value network returns value to send
                    # the result back up the tree
                    self.backpropagation(visited_moves, reward, undo_v_loss=False)

                # Give the node an id so we can find it again when its evaluation comes back
                node_id = self.next_node_id
                self.next_node_id += 1

                self.pending_evaluations[node_id] = (current_node, visited_moves, reward)
                assert self.eval_request_queue is not None
                self.eval_request_queue.put(
                    EvalRequest(
                        agent_id=self.agent_id,
                        node_id=node_id,
                        fen=current_node.state,
                        legal_moves=current_node.legal_move_strings,
                    )
                )

                # Increment the number of nodes sent for processing
                self.number_sent += 1

                # Set the node is current node is awaiting processing
                current_node.awaiting_processing = True

                # If we reach a terminal node not in the main search we still save the game to increase data.
                if current_node.branch_complete and current_node.game_over_type is not GameOverType.NO_TAKES_DRAW:
                    end_nodes.append(current_node)

                # Below is purely for inference mode when the user is playing it
                if self.inference_mode:
                    assert self.send_results_queue is not None and self.inference_mode_queue is not None
                    try:
                        result = self.send_results_queue.get_nowait()
                        if result["send_result"]:
                            self.inference_mode_queue.put((self.root.Ns, self.root.Qs, self.root.moves))
                        if result["move_to_push"]:
                            print("MOVE RECEIVED")
                            move_idx = self.root.moves.index(result["move_to_push"])
                            if move_idx not in self.root.child_nodes.keys():
                                # This occassionally happens when engine takes too long to initialise and then user plays move very quickly
                                new_state: str = self.environment.push(self.root.state, self.root.moves[move_idx])
                                self.root.child_nodes[move_idx] = Node(
                                    new_state, move_idx=move_idx, board=self.environment, parent_node=self.root
                                )
                                print("NEW NODE CREATED")
                                print(move_idx, self.root.child_nodes)
                            #  Make this become the new root node
                            self.root = self.root.child_nodes[move_idx]

                    except Empty:
                        pass

            # Now return to root node
            current_node = self.root

        # Wait until the end of the rollout and then save any games which finished.
        if self.training:
            for node in end_nodes:
                # These are states which we didn't visit with the root node but were end games.
                # In order to maintain high quality data we only save the move whilst the game was being played optimally
                # I.e. if it was only found due to randomness then we don't count it as it a move a real player would never make
                self.save_path_whilst_optimal(node)

    def backpropagation(
        self, visited_moves: List[Tuple[Node, int]], observed_reward: float, undo_v_loss: bool = True
    ) -> None:

        # New incoming state. If win... reward = -1 for current player as they have lost
        gamma_factor = self.search_config.gamma

        # This is the other player as player who made a the move into the new state is the opposite player to whos turn
        # it is in the state
        reward = -observed_reward

        for idx, (node, move_idx) in enumerate(reversed(visited_moves)):
            # Increment the visit count and update the total reward
            new_Q = node.R + gamma_factor**idx * reward

            node.Qs[move_idx] = (node.Ns[move_idx] * node.Qs[move_idx] + new_Q) / (node.Ns[move_idx] + 1)

            # Player will choose their best move... maximum Q
            Qs = node.Qs[node.Ns > 0]
            node.V = np.max(Qs)

            # Only the best rewards propagate up the tree... otherwise player would not choose them
            reward = -float(node.V)

            if self.full_rollout_virtual_loss and undo_v_loss:
                # For full rollouts add one back on... some will have mutliple V losses at the same time
                node.virtual_losses[move_idx] += 1
            elif undo_v_loss:
                # Set all to zero if not using full rollouts
                node.virtual_losses[move_idx] = 0

    def save_path_whilst_optimal(self, start_node: Node) -> None:
        """Find games in which both players have played optimally at least for two moves"""
        game_states: List[PositionRecord] = []
        assert start_node.parent_node is not None and self.game_record_queue is not None

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

            game_states.append(PositionRecord(fen=state, moves=moves, visit_counts=visit_counts, value=value))
            self.saved_memory_local += 1
            current_node = current_node.parent_node

        # Make sure its even to not get bias to wins
        if len(game_states) % 2 != 0:
            game_states = game_states[:-1]

        if len(game_states) == 0:
            return

        self.game_record_queue.put(GameRecord(agent_id=self.agent_id, positions=game_states))

    def save_results_to_memory(self, current_node: Node) -> None:
        assert current_node.parent_node is not None and self.game_record_queue is not None

        if current_node.game_won:
            winner = current_node.parent_node.player
        else:
            winner = None

        game_states: List[PositionRecord] = []

        # Traverse up the tree to the game's starting position and save game states, visit counts, and values
        position: Optional[Node] = current_node.parent_node

        while position is not None:
            state = position.state
            moves = position.moves
            visit_counts = position.Ns

            if winner is not None:
                if position.player == winner:
                    value = 1
                else:
                    value = -1
            else:
                value = 0

            game_states.append(PositionRecord(fen=state, moves=moves, visit_counts=visit_counts, value=value))
            self.saved_memory_local += 1
            position = position.parent_node

        self.game_record_queue.put(GameRecord(agent_id=self.agent_id, positions=game_states))

    def apply_neural_net_results(self) -> None:
        """
        Process results from the neural network in batches, updating nodes and applying Dirichlet noise.
        """

        assert self.eval_result_queue is not None

        # Clear the queue
        while True:
            lagging = self.number_sent - self.number_returned >= self.maximum_lag
            results: List[EvalResult]
            try:
                # Block (briefly, so the stop event is still checked) only when we must wait for the evaluator
                results = (
                    self.eval_result_queue.get(timeout=self.lag_wait_timeout_seconds)
                    if lagging
                    else self.eval_result_queue.get_nowait()
                )
            except Empty:
                results = []

            for result in results:
                # Increment number returned
                self.number_returned += 1

                # Retrieve the node this result belongs to
                node, visited_moves, reward = self.pending_evaluations.pop(result.node_id)

                # Update node value
                node.V = result.value

                # Parameters for Dirichlet noise
                alpha = self.search_config.dirichlet_alpha  # Dirichlet distribution parameter
                epsilon = self.search_config.dirichlet_epsilon  # Blending factor

                # Copy: normalised in place below
                Ps = np.array(result.priors)

                # Normalize probabilities
                sum_Ps = Ps.sum()
                if sum_Ps > 1e-6:
                    Ps /= sum_Ps
                else:
                    Ps = np.ones_like(Ps) / len(Ps)  # Assign uniform probability if sum is too small

                # Apply Dirichlet noise
                if self.training:
                    dirichlet_noise = np.random.dirichlet([alpha] * len(Ps))
                    Ps = (1 - epsilon) * Ps + epsilon * dirichlet_noise

                # Update move probabilities
                for move_idx, prob in zip(range(len(node.moves)), Ps):
                    node.Ps[move_idx] = prob

                node.awaiting_processing = False
                self.backpropagation(visited_moves, result.value + reward)

            if self.number_sent - self.number_returned < self.maximum_lag or self.stop_event.is_set():
                #  If we have a processing lag of less than the maximum allowed lag then we can continue
                return

    def update_node_in_training(self, new_node: Node, action: int) -> None:
        """
        In training, we remember old visited states to update when we finish games
        """
        # Clear references to other parts of tree
        self.root.child_nodes = {action: new_node}

        # Update root node
        self.root = new_node

    def run(self) -> None:
        self.environment = chess_moves.ChessEngine()

        if not self.inference_mode:
            self.train()
        else:
            print("Beginning infinite rollouts")
            while not self.stop_event.is_set():
                self.parallel_search(self.root, int(10e25))

        # Don't block exit flushing queues whose readers have already stopped
        for queue in (self.eval_request_queue, self.game_record_queue, self.stats_queue):
            if queue is not None:
                queue.cancel_join_thread()

    @error_logger
    def train(self, n_games: int = 1000000) -> None:

        game_count = 0

        print(f"Starting training run on agent {self.agent_id}")

        while not self.stop_event.is_set():
            # Start a new game
            move_count = 0
            main_board = chess.Board()

            if self.curriculum is not None:
                # This has some progressively more difficult chess puzzles
                start_state, max_length = self.curriculum.get_start_state(game_count)
                move_count = int(start_state.split()[-1])
                max_length = self.search_config.max_game_length
            else:
                start_state = self.start_state
                max_length = self.search_config.max_game_length

            # Reset main board to start state
            self.reset(start_state)
            main_board.set_fen(start_state)

            node = self.root
            player_check_mated: Optional[str] = None

            while not main_board.is_game_over() and not self.stop_event.is_set():
                start_time = time.time()
                self.parallel_search(current_node=node, number_of_expansions=self.sims)
                end_time = time.time()

                print(f"\n\nTime with parallel search {end_time - start_time:.3f}s.")

                # TODO: sort out now visiting
                if move_count > self.search_config.tau_switch_move or (
                    self.curriculum is not None and game_count < self.search_config.curriculum_low_tau_games
                ):
                    tau = self.search_config.tau_late
                else:
                    tau = self.search_config.tau_early

                node, move, move_idx = self.root.exploratory_select_new_root_node(tau=tau)
                chess_move = chess.Move.from_uci(move)
                main_board.set_fen(self.root.state)
                main_board.push(chess_move)

                print(f"Agent {self.agent_id} Game {game_count} Move: {move_count}")

                if main_board.is_checkmate():
                    player_check_mated = main_board.fen().split()[1]

                    if self.save_images_of_checkmates:
                        save_chess_board_as_im(
                            main_board, f"{self.save_dir}/agent_{self.agent_id}_game_{game_count}.svg", move=chess_move
                        )

                elif main_board.is_stalemate():
                    print("Stalemate!")

                elif main_board.is_insufficient_material():
                    print("Insufficient material to checkmate!")

                elif move_count >= max_length:
                    print("Maximum move length")
                    break
                else:
                    if main_board.is_check():
                        print("King is in check!")

                    self.update_node_in_training(node, move_idx)

                    # Increment move count
                    move_count += 1

                # Save the played game once it's over. A game-ending move doesn't advance the root, so check the board
                # and the node just played, not self.root. The engine also ends games python-chess doesn't (50 moves)
                if main_board.is_game_over() or node.branch_complete:
                    self.save_results_to_memory(node)
                    break

            if self.stop_event.is_set():
                break

            white_win: Optional[bool]
            if player_check_mated == "w":
                white_win = False
            elif player_check_mated == "b":
                white_win = True
            else:
                white_win = None

            if self.stats_queue is not None:
                self.send_update_to_main_thread(white_win, move_count)

            game_count += 1

    def send_update_to_main_thread(self, white_win: Optional[bool], game_length: int) -> None:
        assert self.stats_queue is not None
        self.stats_queue.put_nowait(GameStats(white_win=white_win, game_length=game_length))
