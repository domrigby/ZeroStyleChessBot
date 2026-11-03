from multiprocessing import Process, Queue, Event, synchronize
from queue import Empty
from typing import List, Dict, Optional
from neural_nets.conv_net import ChessNet
from zero_style_chess_engine.messages import EvalRequest, EvalResult
import contextlib
import time
import numpy as np
import torch


class Evaluator(Process):
    """This is meant constantly run the neural network evaluation and training in parallel with MCTS"""

    idle_sleep_seconds = 0.001

    def __init__(
        self,
        neural_net: ChessNet,
        eval_request_queues: Optional[List[Queue]] = None,
        weights_queue: Optional[Queue] = None,
        eval_result_queues: Optional[Dict[int, Queue]] = None,
        batch_size: int = 64,
        stop_event: Optional[synchronize.Event] = None,
    ) -> None:
        """
        :param queue: queue from the tree search
        :param lock:
        """
        super().__init__()
        self.eval_request_queues: Optional[List[Queue]] = eval_request_queues
        self.weights_queue = weights_queue

        # Keyed by agent id
        if eval_result_queues:
            self.eval_result_queues = eval_result_queues
        else:
            self.eval_result_queues = {}

        self.stop_event = stop_event if stop_event is not None else Event()

        self.batch_size = batch_size

        self.neural_net = neural_net

        self.processed_count = 0

    def setup(self) -> None:
        """Heavy initialisation, run in the child process rather than pickled into it"""
        self.neural_net.compile()

    def run(self) -> None:
        """Optimized function to process queued game states efficiently."""

        self.setup()

        if self.eval_result_queues is None:
            raise ValueError("NO RESULTS QUEUE: User must provide results queue from their game tree.")

        stream_context = (
            torch.cuda.stream(torch.cuda.Stream()) if torch.cuda.is_available() else contextlib.nullcontext()
        )

        while not self.stop_event.is_set():
            with stream_context:
                self.collect_and_process()
            self.check_for_network_update()

        # Don't block exit flushing results to agents which have already stopped
        for queue in self.eval_result_queues.values():
            queue.cancel_join_thread()

    def collect_and_process(self) -> None:
        requests: List[EvalRequest] = []
        assert self.eval_request_queues is not None

        # Equal share of the batch per agent. Queue.qsize() isn't used as it raises NotImplementedError on macOS.
        per_queue_allowance = max(1, self.batch_size // len(self.eval_request_queues))

        for queue in self.eval_request_queues:
            for _ in range(per_queue_allowance):
                try:
                    requests.append(queue.get_nowait())
                    self.processed_count += 1

                except Empty:
                    break

        if not requests:
            time.sleep(self.idle_sleep_seconds)
            return

        # Tensorization & Inference
        state_tens, legal_move_mask, legal_move_key = self.neural_net.tensorise_inputs(
            [request.fen for request in requests], [request.legal_moves for request in requests]
        )

        self.neural_net.eval()
        with torch.no_grad():  # Prevents unnecessary gradient tracking
            values, policies = self.neural_net(state_tens, legal_move_mask, infering=True)
        self.neural_net.train()

        # Move from GPU to CPU. Non-blocking increases speed by allowing us to use a different CUDA core.
        # Another improvement was to convert whole bacth to numpy rather than using item as apparently that is slow
        values, policies = values.to("cpu", non_blocking=True).numpy(), policies.to("cpu", non_blocking=True).numpy()

        batched_results: Dict[int, List[EvalResult]] = {request.agent_id: [] for request in requests}

        # Prepare results for the main thread
        for request, policy, value, key in zip(requests, policies, values, legal_move_key):
            if len(request.legal_moves) == 1:
                # The mask has no key for a single legal move: it gets all the prior
                priors = np.ones(1)
            else:
                # The key is in legal_moves order, so the priors are aligned with them
                priors = np.array([policy[move_idx] for _, move_idx in key])
            batched_results[request.agent_id].append(EvalResult(request.node_id, value.item(), priors))

        # Send the batches back
        for agent_id, results in batched_results.items():
            self.eval_result_queues[agent_id].put(results)

    def check_for_network_update(self) -> None:
        assert self.weights_queue is not None
        try:
            new_weights = self.weights_queue.get_nowait()
            self.neural_net.load_state_dict(new_weights)
            self.neural_net.to(self.neural_net.device)
        except Empty:
            return
