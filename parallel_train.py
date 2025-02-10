import multiprocessing as mp

from neural_nets.conv_net import ChessNet

from util.queues import create_agents

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

    NUM_EVALUATORS = 3
    NUM_TRAINERS = 1
    NUM_AGENTS = 3

    chess_net = ChessNet(input_size=[12, 8, 8], output_size=[70, 8, 8], num_repeats=32)
    chess_net.share_memory()

    agents, evaluators, trainers = create_agents(NUM_AGENTS, NUM_EVALUATORS, NUM_TRAINERS, chess_net)

    [trainer.start() for trainer in trainers]
    [eval.start() for eval in evaluators]
    [agent.start() for agent in agents]

    while True:
        pass
