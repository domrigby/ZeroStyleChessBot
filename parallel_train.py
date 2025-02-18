import multiprocessing as mp
from neural_nets.conv_net import ChessNet
from util.queues import create_agents
from util.training_monitor import ChessEngineMonitor
from queue import Empty

class ParallelMonitor:

    def __init__(self):
        pass

if __name__ == "__main__":
    try:
       mp.set_start_method('spawn', force=True)
       print("spawned")
    except RuntimeError:
        print("Failed to spawn")
        pass

    NUM_EVALUATORS = 1
    NUM_TRAINERS = 1
    NUM_AGENTS = 3

    chess_net = ChessNet(input_size=[12, 8, 8], output_size=[70, 8, 8], num_repeats=32, init_lr=0.001)
    chess_net.share_memory()

    agents, evaluators, trainers, data_queues_dicts = create_agents(NUM_AGENTS, NUM_EVALUATORS, NUM_TRAINERS, chess_net)

    [trainer.start() for trainer in trainers]
    [eval.start() for eval in evaluators]
    [agent.start() for agent in agents]

    monitor = ChessEngineMonitor()
    monitor.refresh()

    while True:

        for queue in data_queues_dicts['agents']:

            try:
                new_data_point = queue.get_nowait()
                monitor.update_wins(new_data_point['white_win'])
                monitor.update_game_lengths(new_data_point['game_length'])
            except Empty:
                continue
            monitor.refresh()

        for queue in data_queues_dicts['training']:

            try:
                new_data_point = queue.get_nowait()

                if 'total_loss' in new_data_point:
                    monitor.update_losses(new_data_point['total_loss'], new_data_point['value_loss'],
                                        new_data_point['policy_loss'])

                if 'experience_length' in new_data_point:
                    monitor.update_experience(new_data_point['experience_length'])

            except Empty:
                continue

            monitor.refresh()
