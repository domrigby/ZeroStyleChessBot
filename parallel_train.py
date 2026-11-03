import multiprocessing as mp
from neural_nets.conv_net import ChessNet
from util.queues import create_agents
from util.training_monitor import ChessEngineMonitor
from queue import Empty
from datetime import datetime
from typing import Dict, List
import os
import signal
import sys
import time
import wandb
import torch
import numpy as np
import random
import argparse


def run_pipeline(folder_name: str, chess_net: ChessNet, num_agents: int, num_evaluators: int, num_trainers: int,
                 expert_data_path: str = None, shutdown_timeout: float = 60.) -> int:
    """
    Start the agents, evaluators and trainers, monitor them until SIGINT/SIGTERM or until a child dies, then shut
    them all down.
    :return: exit code: 0 if every child shut down cleanly, 1 if any child died, exited non-zero or had to be
        terminated
    """
    agents, evaluators, trainers, data_queues_dicts, stop_event = create_agents(
        folder_name, num_agents, num_evaluators, num_trainers, chess_net, expert_data_path=expert_data_path)
    children = [*trainers, *evaluators, *agents]

    # Children inherit SIG_IGN for SIGINT, so a Ctrl+C sent to the whole process group doesn't kill them mid-step.
    # They shut down through the stop event instead.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    [child.start() for child in children]
    print("Agent training has commenced.")

    def request_stop(signum, frame):
        print(f"Received {signal.Signals(signum).name}: shutting down.")
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    child_died = False
    try:
        child_died = monitor_until_stopped(children, data_queues_dicts, stop_event, ChessEngineMonitor(folder_name))
    finally:
        stop_event.set()
        shut_down(children, shutdown_timeout)

    failed_children = [child for child in children if child.exitcode != 0]
    for child in failed_children:
        print(f"{child.name} (pid {child.pid}) exited with code {child.exitcode}")

    return 1 if child_died or failed_children else 0


def monitor_until_stopped(children: List[mp.Process], data_queues_dicts: Dict[str, List[mp.Queue]],
                          stop_event, monitor: ChessEngineMonitor) -> bool:
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

        for queue in data_queues_dicts['agents']:

            try:
                new_data_point = queue.get_nowait()
            except Empty:
                continue

            received = True
            monitor.update_wins(new_data_point['white_win'])
            monitor.update_game_lengths(new_data_point['game_length'])

        for queue in data_queues_dicts['training']:

            try:
                new_data_point = queue.get_nowait()
            except Empty:
                continue

            received = True
            if 'total_loss' in new_data_point:
                if new_data_point['total_loss'] != 0:
                    monitor.update_losses(new_data_point['total_loss'], new_data_point['value_loss'],
                                        new_data_point['policy_loss'])

            if 'experience_length' in new_data_point:
                monitor.update_experience(new_data_point['experience_length'])

        time_now = time.time()

        if time_now - start_time > 600:
            monitor.save_progress()
            start_time = time_now

        if not received:
            time.sleep(0.1)

    return False


def shut_down(children: List[mp.Process], timeout: float):
    """ Wait up to timeout in total for the children to exit (the trainer saves first), then terminate stragglers """
    deadline = time.time() + timeout
    for child in children:
        child.join(max(0., deadline - time.time()))

    for child in children:
        if child.is_alive():
            print(f"{child.name} (pid {child.pid}) did not stop within {timeout}s: terminating")
            child.terminate()
            child.join()


def main():

    # Set weights and biases to offline
    wandb.init(mode="offline")

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int,default=42, help="Random seed")
    parser.add_argument("--init_lr", type=float, default=2e-6, help="Initial learning rate")
    parser.add_argument("--num_agents", type=int, default=4, help="Number of self-play agents")
    parser.add_argument("--num_trainers", type=int, default=1, help="Number of trainers")
    parser.add_argument("--num_evaluators", type=int, default=1, help="Number of evaluators")
    parser.add_argument("--run_to_continue", type=str, default=None, help="Override default run name")
    parser.add_argument("--num_cnn_repeats", type=int, default=32, help="Number of repeated ResNet blocks in policy-value function")
    args = parser.parse_args()

    # Set random seedd
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    try:
       mp.set_start_method('spawn', force=True)
       print("Multi-processing spawned.")
    except RuntimeError:
        print("Failed to spawn")
        pass

    RUN_TO_CONTINUE: str = args.run_to_continue

    if not RUN_TO_CONTINUE:
        now = datetime.now()  # Get the current date and time
        datetime_string = now.strftime("run_at_%Y%m%d_%H%M%S")  # Format as string
        folder_name = f"sessions/{datetime_string}"
        print(f"Created run folder: {folder_name}")
        os.mkdir(folder_name)
    else:
        folder_name = RUN_TO_CONTINUE

    # Init with a very low learning rate as we are tuning
    chess_net = ChessNet(input_size=[12, 8, 8], output_size=[70, 8, 8], num_repeats=args.num_cnn_repeats, init_lr=args.init_lr)
    chess_net.load_network("neural_nets/example_network.pt")
    chess_net.share_memory()

    # Save useful daa with weights and biases
    wandb.init(project="ZeroStyleChessBot", name=folder_name, config=vars(args), save_code=True, dir=folder_name)
    cfg = wandb.config  # easy alias

    sys.exit(run_pipeline(folder_name, chess_net, args.num_agents, args.num_evaluators, args.num_trainers,
                          expert_data_path="/home/dom/1TB_drive/chess_data"))


if __name__ == "__main__":
    main()
