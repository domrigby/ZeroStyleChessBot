# engine_interface.py
import chess
import chess.svg
from neural_nets.conv_net import ChessNet
from util.test_fen_strings import FenTests
from util.queues import create_agents
from multiprocessing import Queue, cpu_count
import time
import numpy as np

class EngineInterface:
    def __init__(self, network_path, parallel: bool = False, time_per_move: float = 15., num_rollouts: int = 2000):
        # Initialize your neural network and engine components.
        self.chess_net = ChessNet(input_size=[12, 8, 8], output_size=[70, 8, 8], num_repeats=32)
        self.chess_net.load_network(network_path)
        self.chess_net.eval()

        self.time_per_move: float = time_per_move

        #  Set these up for parallel processing
        self.parallel = parallel
        if parallel:
            self.num_processes: int = 1
            self.inference_queue = Queue()
            self.send_results_queues = [Queue() for _ in range(self.num_processes)]

            # Create the tree and evaluator – note that create_agents returns lists.
            self.tree, evaluator, _, _ = create_agents("", self.num_processes , 1, 0, self.chess_net,
                                                  training=False, verbose=False, inference_mode_queue=self.inference_queue,
                                                  send_results_queue=self.send_results_queues)
            [tree.start() for tree in self.tree]
        else:
            # Create the tree and evaluator – note that create_agents returns lists.
            tree, evaluator, _, _ = create_agents("", 1, 1, 0, self.chess_net,
                                                  training=False, verbose=True)
            self.tree = tree[0]

            # Initialize the board (standard starting position)
            self.current_board = chess.Board()
            self.tree.reset(start_state=self.current_board.fen())
            self.rollouts = num_rollouts

        # Initialize the board (standard starting position)
        self.current_board = chess.Board()
        [ev.start() for ev in evaluator]


    def set_rollouts(self, rollout_count):
        self.rollouts = rollout_count

    def apply_user_move(self, move_uci):
        """Apply a user move (in UCI format) to the board and reset the tree to the new state."""
        move = chess.Move.from_uci(move_uci)
        if move in self.current_board.legal_moves:
            self.current_board.push(move)
            # Reset the search tree with the updated FEN.

            castling_moves = {'e1g1': 'O-O',  # White kingside
                'e1c1': 'O-O-O',  # White queenside
                'e8g8': 'O-O',  # Black kingside
                'e8c8': 'O-O-O'}  # Black queenside

            if move_uci in castling_moves.keys():
                move_uci = castling_moves[move_uci]

            #  Send move to processes
            if self.parallel:
                [queue.put({"send_result": False, "move_to_push": move_uci}) for queue in self.send_results_queues]
            else:
                move_idx = self.tree.root.moves.index(move_uci) if move_uci in self.tree.root.moves else None
                if move_idx is not None and move_idx in self.tree.root.child_nodes:
                    self.tree.root = self.tree.root.child_nodes[move_idx]
                else:
                    self.tree.reset(start_state=self.current_board.fen())

            return True
        return False

    def get_engine_move(self):
        """Perform a search using the specified number of rollouts and return the engine move and new FEN."""
        if self.parallel:
            # In parallel we wait for a certain amount of time for rollouts to occur
            time_now = time.time()
            while time.time() - time_now <= self.time_per_move:
                print(f"\rThinking... {self.time_per_move - time.time() + time_now:.2f}s left", end="")

            # Request results from parallel processes
            [queue.put({"send_result": True, "move_to_push": None}) for queue in self.send_results_queues]

            for idx in range (len(self.send_results_queues)):
                if idx == 0:
                    Ns, Qs, moves = self.inference_queue.get()
                else:
                    temp_Ns, temp_Qs, _ = self.inference_queue.get()
                    Qs += temp_Qs

            # Calculate average Q
            Qs = Qs / len(self.send_results_queues)

            # Get the best move
            move_idx = np.argmax(Qs)
            move = moves[move_idx]
            [queue.put({"send_result": False, "move_to_push": move}) for queue in self.send_results_queues]

        else:
            # Do tree search and then tree
            self.tree.parallel_search(current_node=self.tree.root, number_of_expansions=self.rollouts)
            node, move, move_idx = self.tree.root.greedy_select_new_root_node()
            print(f"Move chosen with Q of: {self.tree.root.Qs[move_idx]}")

        # Convert to Python chess format
        self.current_board.push(chess.Move.from_uci(move))

        return move, self.current_board.fen()

    def get_svg_board(self, size=400):
        """Return an SVG image (as string) of the current board position."""
        return chess.svg.board(self.current_board, size=size)
