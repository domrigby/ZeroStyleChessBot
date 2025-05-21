# engine_interface.py
import chess
import chess.svg
from neural_nets.conv_net import ChessNet
from util.test_fen_strings import FenTests
from util.queues import create_agents


class EngineInterface:
    def __init__(self, network_path, rollout_count=1000):
        # Initialize your neural network and engine components.
        self.chess_net = ChessNet(input_size=[12, 8, 8], output_size=[70, 8, 8], num_repeats=32)
        self.chess_net.load_network(network_path)
        self.chess_net.eval()

        # Create the tree and evaluator – note that create_agents returns lists.
        tree, evaluator, _, _ = create_agents("", 1, 1, 0, self.chess_net,
                                              training=False, verbose=True)
        self.tree = tree[0]

        # Initialize the board (standard starting position)
        self.current_board = chess.Board()
        self.tree.reset(start_state=self.current_board.fen())
        self.rollouts = rollout_count
        [ev.start() for ev in evaluator]

    def set_rollouts(self, rollout_count):
        self.rollouts = rollout_count

    def apply_user_move(self, move_uci):
        """Apply a user move (in UCI format) to the board and reset the tree to the new state."""
        move = chess.Move.from_uci(move_uci)
        if move in self.current_board.legal_moves:
            self.current_board.push(move)
            # Reset the search tree with the updated FEN.
            self.tree.reset(start_state=self.current_board.fen())
            return True
        return False

    def get_engine_move(self):
        """Perform a search using the specified number of rollouts and return the engine move and new FEN."""
        self.tree.parallel_search(current_node=self.tree.root, number_of_expansions=self.rollouts)
        # Use a greedy selection from the search tree.
        node, move, move_idx = self.tree.root.greedy_select_new_root_node()
        print(f"Move chosen with Q of: {self.tree.root.Qs[move_idx]}")
        engine_move = move  # UCI format
        self.current_board.push(chess.Move.from_uci(move))
        # Update the tree for subsequent searches.
        self.tree.root = node
        return engine_move, self.current_board.fen()

    def get_svg_board(self, size=400):
        """Return an SVG image (as string) of the current board position."""
        return chess.svg.board(self.current_board, size=size)
