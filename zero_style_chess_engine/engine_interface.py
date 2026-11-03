# engine_interface.py
import chess
import chess.svg
from neural_nets.conv_net import ChessNet
from util.config import TrainingConfig
from zero_style_chess_engine.evaluator import Evaluator
from zero_style_chess_engine.mcts import MCTS
from multiprocessing import Event, Queue
from typing import Tuple


class EngineInterface:
    def __init__(self, network_path: str, num_rollouts: int = 2000) -> None:
        # Initialize your neural network and engine components.
        self.chess_net = ChessNet(input_size=[12, 8, 8], output_size=[70, 8, 8], num_repeats=32)
        self.chess_net.load_network(network_path)
        self.chess_net.eval()

        # Search constants, lag and evaluator batch size as in training
        config = TrainingConfig()
        self.stop_event = Event()
        eval_request_queue: Queue = Queue()
        eval_result_queue: Queue = Queue()

        # The search runs in this process; only the evaluator is a separate process
        self.mcts = MCTS(
            eval_request_queue,
            eval_result_queue,
            agent_id=0,
            stop_event=self.stop_event,
            search_config=config.search,
            maximum_lag=config.maximum_lag,
            verbose=True,
        )
        self.evaluator = Evaluator(
            neural_net=self.chess_net,
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

        self.evaluator.start()

    def close(self, timeout: float = 10.0) -> None:
        """Stop the evaluator process; otherwise it keeps the program alive."""
        self.stop_event.set()
        self.evaluator.join(timeout)
        if self.evaluator.is_alive():
            self.evaluator.terminate()
            self.evaluator.join()

    def set_rollouts(self, rollout_count: int) -> None:
        self.rollouts = rollout_count

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

    def get_engine_move(self) -> Tuple[str, str]:
        """Perform a search using the specified number of rollouts and return the engine move and new FEN."""
        # Do tree search and then tree
        self.mcts.search(self.rollouts)
        node, move, move_idx = self.mcts.select_greedy()
        print(f"Move chosen with Q of: {self.mcts.root.Qs[move_idx]}")

        # Convert to Python chess format
        self.current_board.push(chess.Move.from_uci(move))

        # Root the tree at the user's turn, so their reply is looked up among the replies just searched
        self.mcts.advance_root(node, move_idx)

        return move, self.current_board.fen()

    def get_svg_board(self, size: int = 400) -> str:
        """Return an SVG image (as string) of the current board position."""
        return chess.svg.board(self.current_board, size=size)
