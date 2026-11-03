from multiprocessing import Process, Queue, Event, synchronize
from queue import Empty
from typing import Any, List, Dict, Optional, Tuple
from neural_nets.conv_net import ChessNet
import contextlib
import time
import torch


class NeuralNetHandling(Process):
    """This is meant constantly run the neural network evaluation and training in parallel with MCTS"""

    idle_sleep_seconds = 0.001

    def __init__(
        self,
        neural_net: ChessNet,
        process_queues: Optional[List[Queue]] = None,
        weights_queue: Optional[Queue] = None,
        results_queue_dict: Optional[Dict[int, Queue]] = None,
        batch_size: int = 64,
        stop_event: Optional[synchronize.Event] = None,
    ) -> None:
        """
        :param queue: queue from the tree search
        :param lock:
        """
        super().__init__()
        self.process_queue: Optional[List[Queue]] = process_queues
        self.weights_queue = weights_queue

        if results_queue_dict:
            self.results_queue = results_queue_dict
        else:
            self.results_queue = {}

        self.stop_event = stop_event if stop_event is not None else Event()

        self.batch_size = batch_size

        self.neural_net = neural_net
        self.neural_net.compile()

        self.processed_count = 0

    def run(self) -> None:
        """Optimized function to process queued game states efficiently."""

        if self.results_queue is None:
            raise ValueError("NO RESULTS QUEUE: User must provide results queue from their game tree.")

        stream_context = (
            torch.cuda.stream(torch.cuda.Stream()) if torch.cuda.is_available() else contextlib.nullcontext()
        )

        while not self.stop_event.is_set():
            with stream_context:
                self.collect_and_process()
            self.check_for_network_update()

        # Don't block exit flushing results to agents which have already stopped
        for queue in self.results_queue.values():
            queue.cancel_join_thread()

    def collect_and_process(self) -> None:
        agent_ids: List[int] = []
        hashes: List[int] = []
        states: List[str] = []
        legal_moves: List[List[str]] = []
        assert self.process_queue is not None

        # Equal share of the batch per agent. Queue.qsize() isn't used as it raises NotImplementedError on macOS.
        per_queue_allowance = max(1, self.batch_size // len(self.process_queue))

        for queue in self.process_queue:
            for _ in range(per_queue_allowance):
                try:
                    agent_id, the_hash, state, legal_moves_strings = queue.get_nowait()
                    agent_ids.append(agent_id)
                    hashes.append(the_hash)
                    states.append(state)
                    legal_moves.append(legal_moves_strings)
                    self.processed_count += 1

                except Empty:
                    break

        if not hashes:
            time.sleep(self.idle_sleep_seconds)
            return

        # Tensorization & Inference
        state_tens, legal_move_mask, legal_move_key = self.neural_net.tensorise_inputs(states, legal_moves)

        self.neural_net.eval()
        with torch.no_grad():  # Prevents unnecessary gradient tracking
            values, policies = self.neural_net(state_tens, legal_move_mask, infering=True)
        self.neural_net.train()

        # Move from GPU to CPU. Non-blocking increases speed by allowing us to use a different CUDA core.
        # Another improvement was to convert whole bacth to numpy rather than using item as apparently that is slow
        values, policies = values.to("cpu", non_blocking=True).numpy(), policies.to("cpu", non_blocking=True).numpy()

        batched_results: Dict[int, List[Tuple[Any, ...]]] = {agent_id: [] for agent_id in set(agent_ids)}

        # Prepare results for the main thread
        for agent_id, the_hash, policy, value, key in zip(agent_ids, hashes, policies, values, legal_move_key):
            move_probs = [[edge, policy[move_idx]] for edge, move_idx in key]
            batched_results[agent_id].append((the_hash, value.item(), move_probs, key))

        # Send the batches back
        for agent_id, results in batched_results.items():
            self.results_queue[agent_id].put(results)

    def check_for_network_update(self) -> None:
        assert self.weights_queue is not None
        try:
            new_weights = self.weights_queue.get_nowait()
            self.neural_net.load_state_dict(new_weights)
            self.neural_net.to(self.neural_net.device)
        except Empty:
            return
