from multiprocessing import Process, Queue, Lock
from queue import Empty
from typing import List, Dict
from neural_nets.conv_net import ChessNet
import numpy as np
import torch
from util.parallel_profiler import parallel_profile

class NeuralNetHandling(Process):
    """ This is meant constantly run the neural network evaluation and training in parallel with MCTS"""

    test_mode = False

    def __init__(self, neural_net: ChessNet, process_queues: List[Queue] = None,
                 results_queue_dict: Dict[int, Queue] = None, batch_size: int = 64):
        """
        :param queue: queue from the tree search
        :param lock:
        """
        super().__init__()
        self.process_queue = process_queues

        if results_queue_dict:
            self.results_queue = results_queue_dict
        else:
            self.results_queue = {}

        self.running = True

        self.batch_size = batch_size

        self.neural_net = neural_net
        self.processed_count = 0

    def run(self):
        """ Optimized function to process queued game states efficiently. """

        if self.results_queue is None:
            raise ValueError("NO RESULTS QUEUE: User must provide results queue from their game tree.")

        while self.running:
            self.collect_and_process()

    def collect_and_process(self):
        agent_ids, hashes, states, legal_moves = [], [], [], []

        queue_occupancy = np.array([queue.qsize() for queue in self.process_queue], dtype=int)
        total_occupancy = queue_occupancy.sum()

        if total_occupancy == 0:
            return
            # continue  # Skip iteration if queues are empty

        # Compute batch allocation **only for non-empty queues**
        batch_size_allowance = np.round(self.batch_size * queue_occupancy / total_occupancy).astype(int)

        # Process each queue in **parallel** (avoids nested loops)
        for idx, (queue, allowance) in enumerate(zip(self.process_queue, batch_size_allowance)):
            if queue.empty():
                continue

            for _ in range(min(queue_occupancy[idx], allowance)):  # Limit max processed items
                try:
                    agent_id, the_hash, state, legal_moves_strings = queue.get_nowait()
                    agent_ids.append(agent_id)
                    hashes.append(the_hash)
                    states.append(state)
                    legal_moves.append(legal_moves_strings)
                    self.processed_count += 1
                except Empty:
                    break  # Queue was unexpectedly empty

        if not hashes:
            return
            # continue  # Skip inference if nothing was processed

        # Tensorization & Inference
        state_tens, legal_move_mask, legal_move_key = self.neural_net.tensorise_inputs(states, legal_moves)

        self.neural_net.eval()
        with torch.no_grad():  # Prevents unnecessary gradient tracking
            values, policies = self.neural_net(state_tens, legal_move_mask, infering=True)
        self.neural_net.train()

        # Move from GPU to CPU. Non-blocking increases speed by allowing us to use a different CUDA core.
        # Another improvement was to convert whole bacth to numpy rather than using item as apparently that is slow
        values, policies = values.to("cpu", non_blocking=True).numpy(), policies.to("cpu", non_blocking=True).numpy()

        batched_results = {agent_id: [] for agent_id in set(agent_ids)}

        # Prepare results for the main thread
        for agent_id, the_hash, policy, value, key in zip(agent_ids, hashes, policies, values, legal_move_key):
            move_probs = [[edge, policy[move_idx]] for edge, move_idx in key]
            batched_results[agent_id].append((the_hash, value.item(), move_probs, key))

        # Send the batches back
        for agent_id, results in batched_results.items():
            self.results_queue[agent_id].put(results)

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


