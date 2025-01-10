from multiprocessing import Process, Queue, Lock
from queue import Empty

from transformers.models.musicgen_melody.modeling_musicgen_melody import MusicgenMelodyOutputWithPast

from tree.memory import Memory
import numpy as np
import chess_moves

from neural_nets.conv_net import ChessNet

class NeuralNetHandling(Process):
    """ This is meant constantly run the neural network evaluation and training in parallelwith MCTS"""

    test_mode = True

    def __init__(self, neural_net: ChessNet, process_queue, results_queue,
                 batch_size: int = 64, nn_load_path: str = None):
        """
        :param queue: queue from the tree search
        :param lock:
        """
        super().__init__()
        self.process_queue = process_queue
        self.results_queue = results_queue

        self.running = True

        self.batch_size = batch_size

        self.neural_net = neural_net
        self.processed_count = 0


    def run(self):
        # If there are states in the qy

        while self.running:

            if not self.process_queue.empty():
                hashes = []
                states = []
                legal_moves = []

                for _ in range(min(self.process_queue.qsize(), self.batch_size)):
                    try:
                        the_hash, state, legal_moves_strings= self.process_queue.get_nowait()
                        hashes.append(the_hash)
                        states.append(state)
                        legal_moves.append(legal_moves_strings)

                        self.processed_count += 1
                    except Empty:
                        break

                if True:
                    self.test_mode_func(hashes, states, legal_moves)
                    continue

                # Check that items have actually been received
                if len(hashes) > 0:

                    # Create the input tensors
                    state_tens, legal_move_mask, legal_move_key = self.neural_net.tensorise_inputs(states, legal_moves)

                    # Perform forward pass
                    self.neural_net.eval()
                    values, policies = self.neural_net(state_tens, legal_move_mask)
                    self.neural_net.train()

                    # Results will come in tuples of (node, value, [[edge, prob]])
                    for idx, the_hash in enumerate(hashes):

                        move_probs = []
                        for edge, move_idx in legal_move_key[idx]:
                            move_probs.append([edge, policies[idx][move_idx].item()])

                        self.results_queue.put((the_hash, values[idx].item(), move_probs, legal_move_key[idx]))


    def test_mode_func(self, hashes, states, legal_moves):

        _, _, legal_move_key = self.neural_net.tensorise_inputs(states, legal_moves)

        for idx, (the_hash, state, legal_move_lookup) in enumerate(zip(hashes, states, legal_move_key)):

            value = material_advantage_normalized(state)

            move_probs = []

            if len(legal_move_lookup) == 0:
                # This shouldnt happen...
                self.results_queue.put((the_hash, -1., [], legal_move_lookup))
                continue

            uniform_prob = 1. / len(legal_move_lookup)

            for edge, move_idx in legal_move_lookup:
                move_probs.append([edge, uniform_prob])

            self.results_queue.put((the_hash, value, move_probs, legal_move_lookup))


    def update_node_and_edges(self, state, evaluation):
        pass

    def stop(self):
        self.running = False

def material_advantage_normalized(fen):
    """
    Calculates the material advantage of the current player based on a FEN string.
    Normalizes the material advantage between -1 and 1 using standard piece values.

    Parameters:
        fen (str): The FEN string representing the chess position.

    Returns:
        float: The normalized material advantage between -1 and 1.
    """
    # Define standard piece values
    piece_values = {
        'p': 1,  # Black pawn
        'n': 3,  # Black knight
        'b': 3,  # Black bishop
        'r': 5,  # Black rook
        'q': 9,  # Black queen
        'P': 1,  # White pawn
        'N': 3,  # White knight
        'B': 3,  # White bishop
        'R': 5,  # White rook
        'Q': 9   # White queen
    }

    # Extract the board part of the FEN string
    board_fen = fen.split(' ')[0]

    # Initialize material counts
    white_material = 0
    black_material = 0

    # Iterate through the FEN board string
    for char in board_fen:
        if char in piece_values:
            if char.isupper():  # White pieces
                white_material += piece_values[char]
            elif char.islower():  # Black pieces
                black_material += piece_values[char]

    # Determine whose turn it is
    current_player = fen.split(' ')[1]  # 'w' for White, 'b' for Black

    # Calculate the material advantage
    material_advantage = white_material - black_material
    if current_player == 'b':
        material_advantage = -material_advantage

    # Normalize the material advantage to the range [-1, 1]
    max_advantage = 39  # Maximum material difference (9+9+5+5+3+3+1+1)
    normalized_advantage = material_advantage / max_advantage

    return normalized_advantage


