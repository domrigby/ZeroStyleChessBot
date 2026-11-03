"""Launch a tiny CPU training pipeline (2 agents, 1 evaluator, 1 trainer) for the lifecycle tests.

Run as a script with the run directory as the working directory. The search budget and paths travel to the
children inside the pickled process objects, via the TrainingConfig passed to create_agents.
"""

import argparse
import multiprocessing as mp
import os
import sys
from typing import Any, ContextManager

from neural_nets.conv_net import ChessNet
from parallel_train import run_pipeline
from util import queues
from util.config import TrainingConfig
from zero_style_chess_engine.trainer import TrainingProcess

NUM_AGENTS = 2
NUM_EVALUATORS = 1
NUM_TRAINERS = 1
FAULT_AFTER_ITERATIONS = 50
TRAINING_BATCH_SIZE = 4
WEIGHTS_UPDATE_FREQUENCY = 2
SHUTDOWN_TIMEOUT_SECONDS = 30.0
TINY_SEARCH_SIMULATIONS = 16


class TinyTrainer(TrainingProcess):
    """Trains on tiny batches so a short CPU run exercises training and weight pushes."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.update(
            batch_size=TRAINING_BATCH_SIZE,
            minibatch_size=TRAINING_BATCH_SIZE,
            weights_update_freq=WEIGHTS_UPDATE_FREQUENCY,
        )
        super().__init__(*args, **kwargs)


class FaultyTrainer(TinyTrainer):
    """Raises after a fixed number of run-loop iterations to simulate a crashing trainer."""

    def collect_experience(self, stream_context: ContextManager[object]) -> bool:
        self.iterations = getattr(self, "iterations", 0) + 1
        if self.iterations > FAULT_AFTER_ITERATIONS:
            raise RuntimeError("Injected trainer fault")
        return super().collect_experience(stream_context)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--inject_fault", action="store_true")
    args = parser.parse_args()

    mp.set_start_method("spawn", force=True)

    setattr(queues, "TrainingProcess", FaultyTrainer if args.inject_fault else TinyTrainer)

    chess_net = ChessNet(input_size=[12, 8, 8], output_size=[70, 8, 8], num_repeats=1, save_dir=args.run_dir)
    chess_net.share_memory()

    # A tiny search budget keeps moves fast on CPU; no curriculum or expert data, generated data kept in the run dir
    config = TrainingConfig(
        num_agents=NUM_AGENTS,
        num_evaluators=NUM_EVALUATORS,
        num_trainers=NUM_TRAINERS,
        sims=TINY_SEARCH_SIMULATIONS,
        run_dir=args.run_dir,
        generated_data_dir=os.path.join(args.run_dir, "generated_data"),
    )
    sys.exit(run_pipeline(args.run_dir, chess_net, config, shutdown_timeout=SHUTDOWN_TIMEOUT_SECONDS))


if __name__ == "__main__":
    main()
