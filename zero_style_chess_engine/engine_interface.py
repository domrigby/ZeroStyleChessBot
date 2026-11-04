# engine_interface.py
import chess
import chess.svg
from neural_nets.conv_net import ChessNet
from zero_style_chess_engine.config import HeuristicValue, MoveSelection, TrainingConfig
from zero_style_chess_engine.evaluator import Evaluator
from zero_style_chess_engine.heuristic_evaluator import HeuristicEvaluator
from zero_style_chess_engine.mcts import MCTS
from zero_style_chess_engine.search_insights import SearchSummary, summarise_search
from multiprocessing import Event, Queue
from typing import Any, Callable, Optional, Tuple, cast
import queue
import threading
import time


class EngineInterface:
    # Rollouts between progress reports during a search
    progress_interval = 100
    # Rollouts between checks for a request to stop pondering: the user's move waits for up to one chunk
    ponder_chunk = 128
    # Rollouts between clock checks in a timed search: a move overshoots its time by up to one chunk
    timed_chunk = 16
    _ponder_thread: Optional[threading.Thread] = None
    # Seconds per move; when set, the search runs for this long instead of a fixed number of rollouts
    move_time: Optional[float] = None

    def __init__(
        self,
        network_path: Optional[str] = None,
        num_rollouts: int = 2000,
        move_selection: MoveSelection = "visits",
        value_function: Optional[HeuristicValue] = None,
        c_puct: Optional[float] = None,
        move_time: Optional[float] = None,
    ) -> None:
        """
        :param network_path: weights to search with: positions are evaluated by the network, in an evaluator process
        :param value_function: instead of a network, search with uniform priors and this hand-written value function,
            answered in this process (a baseline player). Give exactly one of network_path and value_function
        :param c_puct: PUCT exploration constant; None keeps training's (search.c_puct)
        :param move_time: seconds to search each move, in place of num_rollouts; None searches num_rollouts
        """
        if (network_path is None) == (value_function is None):
            raise ValueError("Give exactly one of network_path and value_function")

        # Search constants, lag and evaluator batch size as in training
        config = TrainingConfig()
        search_config = config.search if c_puct is None else config.search.model_copy(update={"c_puct": c_puct})
        self.stop_event = Event()
        self.evaluator: Optional[Evaluator] = None
        eval_request_queue: Queue
        eval_result_queue: Queue
        if value_function is not None:
            # Answered in this thread, so thread queues: same put/get, no feeder-thread delay
            results_queue: "queue.Queue[Any]" = queue.Queue()
            eval_request_queue = cast(Queue, HeuristicEvaluator(results_queue, value_function))
            eval_result_queue = cast(Queue, results_queue)
        else:
            eval_request_queue, eval_result_queue = Queue(), Queue()

        # The search runs in this process; only the evaluator is a separate process
        self.mcts = MCTS(
            eval_request_queue,
            eval_result_queue,
            agent_id=0,
            stop_event=self.stop_event,
            search_config=search_config,
            maximum_lag=config.maximum_lag,
            verbose=True,
        )
        if network_path is not None:
            chess_net = ChessNet(input_size=[12, 8, 8], output_size=[70, 8, 8], num_repeats=32)
            chess_net.load_network(network_path)
            chess_net.eval()
            self.evaluator = Evaluator(
                neural_net=chess_net,
                eval_request_queues=[eval_request_queue],
                weights_queue=Queue(maxsize=1),  # Nothing sends new weights while playing
                eval_result_queues={self.mcts.agent_id: eval_result_queue},
                batch_size=config.evaluator_batch_size,
                stop_event=self.stop_event,
            )

        # Initialize the board (standard starting position)
        self.current_board = chess.Board()
        self.mcts.reset(start_state=self.current_board.fen())
        self.rollouts = num_rollouts
        self.move_selection: MoveSelection = move_selection
        self.move_time = move_time

        if self.evaluator is not None:
            self.evaluator.start()

    def close(self, timeout: float = 10.0) -> None:
        """Stop the evaluator process; otherwise it keeps the program alive."""
        self.stop_pondering()
        self.stop_event.set()
        if self.evaluator is None:
            return
        self.evaluator.join(timeout)
        if self.evaluator.is_alive():
            self.evaluator.terminate()
            self.evaluator.join()

    def reset(self, board: chess.Board) -> None:
        """
        Start a fresh search tree at the board's position. The board's move history is kept, so a caller can tell
        later whether a game continues from it
        """
        self.current_board = board.copy()
        self.mcts.reset(start_state=board.fen())

    def set_rollouts(self, rollout_count: int) -> None:
        self.rollouts = rollout_count

    def start_pondering(self, max_rollouts: int, on_progress: Optional[Callable[[SearchSummary], None]] = None) -> None:
        """
        Search the current position in a background thread while the user decides, until stop_pondering or
        max_rollouts. The user's move then keeps the search below it. Call stop_pondering before touching the tree
        :param on_progress: called from the pondering thread after every chunk; never after stop_pondering returns
        """
        self._ponder_stop = threading.Event()
        self._ponder_thread = threading.Thread(target=self._ponder, args=(max_rollouts, on_progress), daemon=True)
        self._ponder_thread.start()

    def _ponder(self, max_rollouts: int, on_progress: Optional[Callable[[SearchSummary], None]]) -> None:
        start_time = time.perf_counter()
        rollouts_done = 0
        while rollouts_done < max_rollouts and not self._ponder_stop.is_set() and not self.mcts.stop_event.is_set():
            chunk = min(self.ponder_chunk, max_rollouts - rollouts_done)
            self.mcts.search(chunk)
            rollouts_done += chunk
            if on_progress is not None:
                on_progress(summarise_search(self.mcts.root, rollouts_done, time.perf_counter() - start_time))

    def stop_pondering(self) -> None:
        """Stop pondering and wait for the current chunk of rollouts to finish. Safe to call when not pondering"""
        if self._ponder_thread is None:
            return
        self._ponder_stop.set()
        self._ponder_thread.join()
        self._ponder_thread = None

    def apply_user_move(self, move_uci: str) -> bool:
        """Apply a user move (in UCI format) to the board and reset the tree to the new state."""
        move = chess.Move.from_uci(move_uci)
        if move in self.current_board.legal_moves:
            self.current_board.push(move)
            # Reset the search tree with the updated FEN.

            castling_moves = {
                "e1g1": "O-O",  # White kingside
                "e1c1": "O-O-O",  # White queenside
                "e8g8": "O-O",  # Black kingside
                "e8c8": "O-O-O",
            }  # Black queenside

            if move_uci in castling_moves.keys():
                move_uci = castling_moves[move_uci]

            move_idx = self.mcts.root.moves.index(move_uci) if move_uci in self.mcts.root.moves else None
            if move_idx is not None and move_idx in self.mcts.root.child_nodes:
                # Keep the search already done below the user's move
                self.mcts.advance_root(self.mcts.root.child_nodes[move_idx], move_idx)
            else:
                self.mcts.reset(start_state=self.current_board.fen())

            return True
        return False

    def get_engine_move(self, on_progress: Optional[Callable[[SearchSummary], None]] = None) -> Tuple[str, str]:
        """
        Search for move_time seconds if set, otherwise for the set number of rollouts, and return the engine move and
        new FEN.
        :param on_progress: called from this thread every progress_interval rollouts, and once at the end of the search
        """
        start_time = time.perf_counter()
        rollouts_done = 0
        last_reported = 0
        while not self.mcts.stop_event.is_set():
            # Chunked searches run the same rollouts as one search: the tree carries everything between them
            if self.move_time is not None:
                chunk = self.timed_chunk
            else:
                chunk = min(self.progress_interval, self.rollouts - rollouts_done)
            self.mcts.search(chunk)
            rollouts_done += chunk
            elapsed = time.perf_counter() - start_time
            finished = elapsed >= self.move_time if self.move_time is not None else rollouts_done >= self.rollouts
            if on_progress is not None and (finished or rollouts_done - last_reported >= self.progress_interval):
                on_progress(summarise_search(self.mcts.root, rollouts_done, elapsed))
                last_reported = rollouts_done
            if finished:
                break

        node, move, move_idx = self.mcts.select_greedy(self.move_selection)
        print(
            f"Move chosen by {self.move_selection}: Q {self.mcts.root.Qs[move_idx]:.3f}, "
            f"{int(self.mcts.root.Ns[move_idx])} visits"
        )

        # Convert to Python chess format
        self.current_board.push(chess.Move.from_uci(move))

        # Root the tree at the user's turn, so their reply is looked up among the replies just searched
        self.mcts.advance_root(node, move_idx)

        return move, self.current_board.fen()

    def get_svg_board(self, size: int = 400) -> str:
        """Return an SVG image (as string) of the current board position."""
        return chess.svg.board(self.current_board, size=size)
