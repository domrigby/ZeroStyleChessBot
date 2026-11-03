"""Pure Monte Carlo tree search: the tree, PUCT selection and the evaluator round trip. No process, no game loop."""

import numpy as np
from multiprocessing import Queue, synchronize
from queue import Empty
from typing import List, Dict, Optional, Tuple

from util.config import SearchConfig
from zero_style_chess_engine.messages import EvalRequest, EvalResult
import chess_moves
from enum import Enum
from numba import njit

START_STATE = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


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

    @property
    def legal_move_strings(self) -> List[str]:
        return [str(move) for move in self.moves]


class MCTS:
    """Searches from root, sending leaves to an evaluator over a queue pair and applying its results as they arrive.

    Several evaluations can be in flight at once (up to maximum_lag); virtual losses steer later rollouts away from
    them meanwhile. Cheap to construct and picklable, so it can be built in the parent and sent to a spawned child.
    """

    full_rollout_virtual_loss = False
    lag_wait_timeout_seconds = 0.1

    def __init__(
        self,
        eval_request_queue: Optional[Queue],
        eval_result_queue: Optional[Queue],
        agent_id: int,
        stop_event: synchronize.Event,
        search_config: Optional[SearchConfig] = None,
        add_noise: bool = False,
        maximum_lag: int = 8,
        verbose: bool = False,
    ) -> None:
        """
        :param agent_id: key of this search's result queue in the evaluator
        :param add_noise: mix Dirichlet noise into the priors (self-play exploration)
        :param maximum_lag: how many evaluations may be outstanding before we wait for the evaluator
        """
        self.eval_request_queue = eval_request_queue
        self.eval_result_queue = eval_result_queue
        self.agent_id = agent_id
        self.stop_event = stop_event

        # c_puct, gamma and Dirichlet noise
        self.search_config = search_config if search_config is not None else SearchConfig()
        self.add_noise = add_noise
        self.maximum_lag = maximum_lag
        self.verbose = verbose

        # Created on first search, in the process that searches
        self.environment: Optional[chess_moves.ChessEngine] = None

        # Nodes awaiting evaluation, keyed by node id. Ids come from a counter that is never reset, so a result for a
        # node from a previous game can't be mistaken for one in the current game
        self.pending_evaluations: Dict[int, Tuple[Node, List[Tuple[Node, int]], float]] = {}
        self.next_node_id = 0

        # Set up the processing stat
        self.number_sent = 0
        self.number_returned = 0

        # Initiate root
        self.root: Node
        self.reset()

    def reset(self, start_state: str = START_STATE) -> None:
        self.root = Node(state=start_state, board=chess_moves.ChessEngine())

    def search(self, number_of_expansions: int = 1000) -> List[Node]:
        """
        Run rollouts from the root. One evaluation is usually still pending on return; apply_eval_results picks it up.
        :return: terminal nodes (other than fifty-move draws) reached off the played line during this search
        """

        end_nodes: List[Node] = []

        #  Initialise environment if not yet done so
        if self.environment is None:
            self.environment = chess_moves.ChessEngine()

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
                self.apply_eval_results()

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

            # Now return to root node
            current_node = self.root

        return end_nodes

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

    def apply_eval_results(self) -> None:
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
                if self.add_noise:
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

    def select_child(self, tau: float = 1.0) -> Tuple[Node, str, int]:
        """
        Select a random child of the root with a temperature adjusted probability distribution
        :param tau:
        :return:
        """
        N_to_tau = np.power(self.root.Ns, 1.0 / tau)
        probs = N_to_tau / np.sum(N_to_tau)
        probs[self.root.Ns == 0] = 0  # Assert this... was occasionly getting an error in which it chooses unvisited
        chosen_move = np.random.choice(len(self.root.moves), p=probs)

        return self.root.child_nodes[chosen_move], self.root.moves[chosen_move], chosen_move

    def select_greedy(self) -> Tuple[Node, str, int]:
        """
        Select the child of the root with the highest Q
        :return:
        """
        local_Qs = np.copy(self.root.Qs)
        legit_moves = self.root.Ns > 0
        local_Qs[~legit_moves] = -np.inf
        q_max = np.max(local_Qs)
        max_idxs = np.where(local_Qs == q_max)[0]
        chosen_move = np.random.choice(max_idxs)
        return self.root.child_nodes[chosen_move], self.root.moves[chosen_move], chosen_move

    def advance_root(self, new_node: Node, action: int) -> None:
        """
        Make new_node (the root's child for action) the root, dropping the root's other subtrees. The old root stays
        reachable as new_node.parent_node, so a finished game can be walked back to its start
        """
        # Clear references to other parts of tree
        self.root.child_nodes = {action: new_node}

        # Update root node
        self.root = new_node
