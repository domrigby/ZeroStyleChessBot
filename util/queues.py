from multiprocessing import Queue
from typing import List, Dict
from tree.parallel_game_tree import GameTree
from tree.evaluator import NeuralNetHandling
from tree.trainer import TrainingProcess

def create_agents(num_agents: int, num_evaluators: int, num_trainers: int, network,
                  training: bool = True):
    """
    This function initialises all the agents and strings and distributes the queues accordingly
    :param num_agents: number of agents
    :param num_evaluators: number of evaluators
    :param num_trainers: number of trainers
    :param network: the neural network being used
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

    for idx in range(num_agents):

        process_queue = Queue()
        results_queue = Queue()

        process_queues[int(idx//agents_per_evaluator)].append(process_queue)

        if num_trainers:
            training_queue = Queue()
            training_queues[int(idx//agents_per_trainer)].append(training_queue)
        else:
            training_queue = None

        agent = GameTree(training=training, multiprocess=True, process_queue=process_queue, experience_queue=training_queue,
                         results_queue=results_queue)

        agents.append(agent)
        results_queue_dicts[agent.agent_id] = results_queue

    # Evaluator
    evaluators: List[NeuralNetHandling] = []
    for idx in range(num_evaluators):
        evaluator = NeuralNetHandling(neural_net=network, process_queues=process_queues[idx],
                                     results_queue_dict=results_queue_dicts, batch_size=128)
        evaluators.append(evaluator)

    trainers: List[TrainingProcess] = []
    for idx in range(num_trainers):
        trainers.append(TrainingProcess(neural_net=network, experience_queues=training_queues[idx], batch_size=128))

    return agents, evaluators, trainers










