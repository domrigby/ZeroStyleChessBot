"""Launch a tiny CPU training pipeline (2 agents, 1 evaluator, 1 trainer) for the lifecycle tests.

Run as a script with the run directory as the working directory. Spawned children re-import this file as
``__mp_main__``, so the module-level class-attribute overrides below apply in every child too.
"""
import argparse
import multiprocessing as mp
import sys

from neural_nets.conv_net import ChessNet
from parallel_train import run_pipeline
from util import queues
from zero_style_chess_engine.parallel_game_tree import GameTree
from zero_style_chess_engine.trainer import TrainingProcess

NUM_AGENTS = 2
NUM_EVALUATORS = 1
NUM_TRAINERS = 1
FAULT_AFTER_ITERATIONS = 50
TRAINING_BATCH_SIZE = 4
WEIGHTS_UPDATE_FREQUENCY = 2
SHUTDOWN_TIMEOUT_SECONDS = 30.

# Curriculum puzzles need a hardcoded data path until Phase 3; a tiny search budget keeps moves fast on CPU
GameTree.curiculum = False
GameTree.sims = 16


class TinyTrainer(TrainingProcess):
    """Trains on tiny batches so a short CPU run exercises training and weight pushes."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs.update(batch_size=TRAINING_BATCH_SIZE, minibatch_size=TRAINING_BATCH_SIZE,
                      weights_update_freq=WEIGHTS_UPDATE_FREQUENCY)
        super().__init__(*args, **kwargs)


class FaultyTrainer(TinyTrainer):
    """Raises after a fixed number of run-loop iterations to simulate a crashing trainer."""

    def collect_experience(self, stream_context) -> bool:
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

    queues.TrainingProcess = FaultyTrainer if args.inject_fault else TinyTrainer

    chess_net = ChessNet(input_size=[12, 8, 8], output_size=[70, 8, 8], num_repeats=1, save_dir=args.run_dir)
    chess_net.share_memory()

    sys.exit(run_pipeline(args.run_dir, chess_net, NUM_AGENTS, NUM_EVALUATORS, NUM_TRAINERS,
                          shutdown_timeout=SHUTDOWN_TIMEOUT_SECONDS))


if __name__ == "__main__":
    main()
