import multiprocessing as mp
from neural_nets.conv_net import ChessNet
from util.queues import create_agents
from util.training_monitor import ChessEngineMonitor
from queue import Empty
from datetime import datetime
import os
import time

class ParallelMonitor:

    def __init__(self):
        pass

if __name__ == "__main__":
    try:
       mp.set_start_method('spawn', force=True)
       print("Multi-processing spawned.")
    except RuntimeError:
        print("Failed to spawn")
        pass

    NUM_EVALUATORS: int = 1
    NUM_TRAINERS: int = 1
    NUM_AGENTS: int = 4

    RUN_TO_CONTINUE: str = ""

    if not RUN_TO_CONTINUE:
        now = datetime.now()  # Get the current date and time
        datetime_string = now.strftime("run_at_%Y%m%d_%H%M%S")  # Format as string
        folder_name = f"sessions/{datetime_string}"
        print(f"Created run folder: {folder_name}")
        os.mkdir(folder_name)
    else:
        folder_name = RUN_TO_CONTINUE


    # Init with a very low learning rate as we are tuning
    chess_net = ChessNet(input_size=[12, 8, 8], output_size=[70, 8, 8], num_repeats=32, init_lr=2e-6)
    chess_net.load_network("neural_nets/example_network.pt")
    chess_net.share_memory()

    agents, evaluators, trainers, data_queues_dicts = create_agents(folder_name, NUM_AGENTS, NUM_EVALUATORS, NUM_TRAINERS, chess_net,
                                                                    expert_data_path="/home/dom/1TB_drive/chess_data")

    [trainer.start() for trainer in trainers]
    [eval.start() for eval in evaluators]
    [agent.start() for agent in agents]
    print("Agent training has commenced.")

    monitor = ChessEngineMonitor(folder_name)

    start_time = time.time()

    while True:

        for queue in data_queues_dicts['agents']:

            try:
                new_data_point = queue.get_nowait()
                monitor.update_wins(new_data_point['white_win'])
                monitor.update_game_lengths(new_data_point['game_length'])
            except Empty:
                continue

        for queue in data_queues_dicts['training']:

            try:
                new_data_point = queue.get_nowait()

                if 'total_loss' in new_data_point:
                    if new_data_point['total_loss'] != 0:
                        monitor.update_losses(new_data_point['total_loss'], new_data_point['value_loss'],
                                            new_data_point['policy_loss'])

                if 'experience_length' in new_data_point:
                    monitor.update_experience(new_data_point['experience_length'])

            except Empty:
                continue

            time_now = time.time()

            if time_now - start_time > 600:
                monitor.save_progress()
                start_time = time_now
