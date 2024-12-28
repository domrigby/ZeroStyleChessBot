from multiprocessing import Process, Queue, Lock
import numpy as np
import chess_moves

class NeuralNetHandling(Process):
    """ This is meant constantly run the neural network evaluation and training in parallelwith MCTS"""

    def __init__(self, neural_network, queue, lock):
        """
        :param queue: queue from the tree search
        :param lock:
        """
        super().__init__()
        self.queue = queue
        self.lock = lock
        self.running = True

        self.neural_network = neural_network

    def run(self):
        # If there are states in the qy
        while self.running:
            if not self.queue.empty():
                # Process inference
                states_to_evaluate = [self.queue.get() for _ in range(min(self.queue.qsize(), self.batch_size))]
                evaluations = self.evaluate_batch([state['fen'] for state in states_to_evaluate])

                # Update nodes/edges with evaluations
                for state, eval in zip(states_to_evaluate, evaluations):
                    self.update_node_and_edges(state, eval)

            self.train_neural_network()
                
    def train_neural_network(self):
        pass

    def update_node_and_edges(self, state, evaluation):
        pass
                
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