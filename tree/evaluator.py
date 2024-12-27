from multiprocessing import Process, Queue, Lock
import numpy as np
import chess_moves

class NeuralNetHandling(Process):
    """ This is meant constantly run the neural network evaluation and training in parallelwith MCTS"""

    def __init__(self, queue, lock):
        """
        :param queue: queue from the tree search
        :param lock:
        """
        super().__init__()
        self.queue = queue
        self.lock = lock
        self.running = True

    def run(self):
        while self.running:
            if not self.queue.empty():
                states_to_evaluate = []
                while not self.queue.empty():
                    states_to_evaluate.append(self.queue.get())

                # Batch process states with the neural network
                evaluations = self.evaluate_batch([state['fen'] for state in states_to_evaluate])

                # Write results back to the nodes/edges
                for state, eval in zip(states_to_evaluate, evaluations):
                    with self.lock:
                        edge = state['edge']
                        edge.W += eval['value']
                        edge.N += 1
                        for move, prob in eval['policy'].items():
                            matching_edge = next(e for e in state['node'].edges if e.move == move)
                            matching_edge.W += prob

    def evaluate_batch(self, states):
        """
        Simulate neural network evaluation. Replace this with actual model inference.
        Returns a list of dictionaries with 'value' and 'policy'.
        """
        evaluations = []
        for state in states:
            evaluations.append({
                'value': np.random.uniform(-1, 1),  # Simulated value
                'policy': {move: np.random.uniform(0, 1) for move in range(24)}  # Simulated policy
            })
        return evaluations

    def stop(self):
        self.running = False