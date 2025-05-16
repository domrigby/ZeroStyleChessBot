from multiprocessing import Queue
from torch import multiprocessing
from datetime import datetime
import os
from typing import List, Dict
from zero_style_chess_engine.parallel_game_tree import GameTree
from zero_style_chess_engine.evaluator import NeuralNetHandling
from zero_style_chess_engine.trainer import TrainingProcess

def create_agents(save_dir: str, num_agents: int, num_evaluators: int, num_trainers: int, network, training: bool = True,
                  start_state: str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"):
    """
    This function initialises all the agents and strings and distributes the queues accordingly
    :param num_agents: number of agents
    :param num_evaluators: number of evaluators
    :param num_trainers: number of trainers
    :param network: the neural network being used
    :param start_state
    :return:
    """

    agents_per_evaluator = num_agents / num_evaluators

    if num_trainers:
        agents_per_trainer = num_agents / num_trainers

    # Start initializing stuff
    agents: List[GameTree] = []
    results_queue_dicts: Dict[int, Queue] = {}
    process_queues: List[List[Queue]] = [[] for _ in range(num_evaluators)]
    training_queues: List[List[Queue]] = [[] for _ in range(num_trainers)]
    data_queues: Dict[str, List[Queue]] = {'agents': [], 'training': []}
    weights_queue: Queue = multiprocessing.Queue()

    for idx in range(num_agents):

        process_queue = Queue()
        results_queue = Queue()

        # Create data queue
        agent_data_queue = Queue()
        data_queues['agents'].append(agent_data_queue)

        process_queues[int(idx//agents_per_evaluator)].append(process_queue)

        if num_trainers:
            training_queue = Queue()
            training_queues[int(idx // agents_per_trainer)].append(training_queue)
        else:
            training_queue = None

        agent = GameTree(save_dir=save_dir, training=training, multiprocess=True, process_queue=process_queue,
                         experience_queue=training_queue, results_queue=results_queue, data_queue=agent_data_queue,
                         start_state=start_state)

        agents.append(agent)
        results_queue_dicts[agent.agent_id] = results_queue

    # Evaluator
    evaluators: List[NeuralNetHandling] = []
    for idx in range(num_evaluators):
        evaluator = NeuralNetHandling(neural_net=network, process_queues=process_queues[idx], weights_queue=weights_queue,
                                     results_queue_dict=results_queue_dicts, batch_size=256)
        evaluators.append(evaluator)

    trainers: List[TrainingProcess] = []
    for idx in range(num_trainers):
        training_data_queue = Queue()
        data_queues['training'].append(training_data_queue)
        trainers.append(TrainingProcess(save_dir=save_dir ,neural_net=network, experience_queues=training_queues[idx],
                                        batch_size=128, num_agents=num_agents, data_queue=training_data_queue,
                                        weights_queue=weights_queue, min_num_batches_to_train=1))

    return agents, evaluators, trainers, data_queues
