import multiprocessing as mp
from neural_nets.conv_net import ChessNet
from zero_style_chess_engine.config import TrainingConfig, check_paths_exist, from_hydra, register_defaults
from util.queues import create_agents
from util.training_monitor import ChessEngineMonitor
from zero_style_chess_engine.messages import GameStats, TrainerStats
from queue import Empty
from multiprocessing import synchronize
from types import FrameType
from typing import Dict, List, Optional
import os
import signal
import sys
import time
import wandb
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
import torch
import numpy as np
import random


def run_pipeline(folder_name: str, chess_net: ChessNet, config: TrainingConfig, shutdown_timeout: float = 60.0) -> int:
    """
    Start the agents, evaluators and trainers, monitor them until SIGINT/SIGTERM or until a child dies, then shut
    them all down.
    :return: exit code: 0 if every child shut down cleanly, 1 if any child died, exited non-zero or had to be
        terminated
    """
    agents, evaluators, trainers, stats_queues, stop_event = create_agents(
        folder_name, config.num_agents, config.num_evaluators, config.num_trainers, chess_net, config=config
    )
    children: List[mp.Process] = [*trainers, *evaluators, *agents]

    # Children inherit SIG_IGN for SIGINT, so a Ctrl+C sent to the whole process group doesn't kill them mid-step.
    # They shut down through the stop event instead.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    for child in children:
        child.start()
    print("Agent training has commenced.")

    def request_stop(signum: int, frame: Optional[FrameType]) -> None:
        print(f"Received {signal.Signals(signum).name}: shutting down.")
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    child_died = False
    try:
        child_died = monitor_until_stopped(children, stats_queues, stop_event, ChessEngineMonitor(folder_name))
    finally:
        stop_event.set()
        shut_down(children, shutdown_timeout)

    failed_children = [child for child in children if child.exitcode != 0]
    for child in failed_children:
        print(f"{child.name} (pid {child.pid}) exited with code {child.exitcode}")

    return 1 if child_died or failed_children else 0


def monitor_until_stopped(
    children: List[mp.Process],
    stats_queues: Dict[str, List[mp.Queue]],
    stop_event: synchronize.Event,
    monitor: ChessEngineMonitor,
) -> bool:
    """
    Log progress from the children until the stop event is set.
    :return: True if a child died before the stop event was set
    """
    start_time = time.time()

    while not stop_event.is_set():
        dead_children = [child for child in children if not child.is_alive()]
        if dead_children:
            for child in dead_children:
                print(f"{child.name} (pid {child.pid}) died unexpectedly with exit code {child.exitcode}")
            return True

        received = False

        for queue in [*stats_queues["agents"], *stats_queues["training"]]:
            try:
                stats = queue.get_nowait()
            except Empty:
                continue

            received = True
            if isinstance(stats, GameStats):
                monitor.update_wins(stats.white_win)
                monitor.update_game_lengths(stats.game_length)
            elif isinstance(stats, TrainerStats):
                if stats.total_loss != 0:
                    monitor.update_losses(stats.total_loss, stats.value_loss, stats.policy_loss)
                monitor.update_experience(stats.experience_length)
            else:
                raise TypeError(f"Unexpected message on a stats queue: {stats!r}")

        time_now = time.time()

        if time_now - start_time > 600:
            monitor.save_progress()
            start_time = time_now

        if not received:
            time.sleep(0.1)

    return False


def shut_down(children: List[mp.Process], timeout: float) -> None:
    """Wait up to timeout in total for the children to exit (the trainer saves first), then terminate stragglers"""
    deadline = time.time() + timeout
    for child in children:
        child.join(max(0.0, deadline - time.time()))

    for child in children:
        if child.is_alive():
            print(f"{child.name} (pid {child.pid}) did not stop within {timeout}s: terminating")
            child.terminate()
            child.join()


register_defaults("training_defaults", TrainingConfig)


@hydra.main(version_base="1.3", config_path="../conf", config_name="train")
def main(hydra_config: DictConfig) -> None:

    config = from_hydra(TrainingConfig, hydra_config)
    check_paths_exist(config)

    # Set random seedd
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    try:
        mp.set_start_method("spawn", force=True)
        print("Multi-processing spawned.")
    except RuntimeError:
        print("Failed to spawn")
        pass

    chess_net = ChessNet(input_size=[12, 8, 8], output_size=[70, 8, 8], num_repeats=config.num_cnn_repeats)
    if config.network_path is not None:
        chess_net.load_network(config.network_path)
    chess_net.share_memory()

    # Hydra creates the run folder; relative to the working directory, as before (sessions/run_at_<timestamp>)
    folder_name = os.path.relpath(HydraConfig.get().runtime.output_dir)
    print(f"Run folder: {folder_name}")

    # Save useful daa with weights and biases (offline: no account or API key needed)
    wandb.init(
        project="ZeroStyleChessBot",
        name=folder_name,
        config=config.model_dump(),
        save_code=True,
        dir=folder_name,
        mode="offline",
    )

    sys.exit(run_pipeline(folder_name, chess_net, config))


if __name__ == "__main__":
    main()
