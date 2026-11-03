from multiprocessing import Queue, Event
from torch import multiprocessing
import copy
from typing import List, Dict
from neural_nets.data.fen_chess_puzzles.curiculum_maker import load_curricula
from util.config import TrainingConfig
from zero_style_chess_engine.parallel_game_tree import GameTree
from zero_style_chess_engine.evaluator import NeuralNetHandling
from zero_style_chess_engine.trainer import TrainingProcess


def create_agents(
    save_dir: str,
    num_agents: int,
    num_evaluators: int,
    num_trainers: int,
    network,
    training: bool = True,
    start_state: str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    verbose: bool = False,
    inference_mode_queue: Queue = None,
    send_results_queue: List[Queue] = None,
    config: TrainingConfig = None,
):
    """
    This function initialises all the agents and strings and distributes the queues accordingly
    TOOD: get rid of this
    :param num_agents: number of agents
    :param num_evaluators: number of evaluators
    :param num_trainers: number of trainers
    :param network: the neural network being used
    :param start_state
    :param config: paths and search/training knobs; defaults to TrainingConfig()
    :return: agents, evaluators, trainers, data queues for the supervisor, and the stop event shared by every
        process (set it to make them all shut down)
    """

    if config is None:
        config = TrainingConfig()

    agents_per_evaluator = num_agents / num_evaluators

    if num_trainers:
        agents_per_trainer = num_agents / num_trainers

    # Start initializing stuff
    agents: List[GameTree] = []
    results_queue_dicts: Dict[int, Queue] = {}
    process_queues: List[List[Queue]] = [[] for _ in range(num_evaluators)]
    training_queues: List[List[Queue]] = [[] for _ in range(num_trainers)]
    data_queues: Dict[str, List[Queue]] = {"agents": [], "training": []}
    stop_event = Event()

    # One weights queue per evaluator so every evaluator receives every update. maxsize=1: the trainer replaces an
    # update the evaluator hasn't picked up yet rather than queueing a backlog of state dicts.
    weights_queues: List[Queue] = [multiprocessing.Queue(maxsize=1) for _ in range(num_evaluators)]

    # Read the puzzle CSV once here; each agent gets only its own sample of positions
    curricula = load_curricula(config.curriculum_csv_path if training else None, num_agents)

    for idx in range(num_agents):
        process_queue = Queue()
        results_queue = Queue()

        # Create data queue
        agent_data_queue = Queue()
        data_queues["agents"].append(agent_data_queue)

        process_queues[int(idx // agents_per_evaluator)].append(process_queue)

        if num_trainers:
            training_queue = Queue()
            training_queues[int(idx // agents_per_trainer)].append(training_queue)
        else:
            training_queue = None

        if inference_mode_queue and send_results_queue:
            inference_mode = True
            send_queue = send_results_queue[idx]
        else:
            inference_mode = False
            send_queue = None

        agent = GameTree(
            save_dir=save_dir,
            training=training,
            process_queue=process_queue,
            experience_queue=training_queue,
            results_queue=results_queue,
            data_queue=agent_data_queue,
            start_state=start_state,
            verbose=verbose,
            inference_mode=inference_mode,
            infering_mode_queue=inference_mode_queue,
            send_results_queue=send_queue,
            stop_event=stop_event,
            sims=config.sims,
            maximum_lag=config.maximum_lag,
            curriculum=curricula[idx],
        )

        agents.append(agent)
        results_queue_dicts[agent.agent_id] = results_queue

    # Evaluator
    evaluators: List[NeuralNetHandling] = []
    for idx in range(num_evaluators):
        # Each evaluator gets its own copy. Spawn pickling would otherwise put the trainer's and evaluators' parameters
        # in the same shared storage, so an evaluator's load_state_dict would roll back the trainer's weights.
        # TODO: costs one weights copy per evaluator. For large nets, replace with a single inference copy shared by all
        #  evaluators, which the trainer copies its weights into every weights_update_freq steps under an mp.Lock.
        evaluator = NeuralNetHandling(
            neural_net=copy.deepcopy(network),
            process_queues=process_queues[idx],
            weights_queue=weights_queues[idx],
            results_queue_dict=results_queue_dicts,
            batch_size=config.evaluator_batch_size,
            stop_event=stop_event,
        )
        evaluators.append(evaluator)

    trainers: List[TrainingProcess] = []
    for idx in range(num_trainers):
        training_data_queue = Queue()
        data_queues["training"].append(training_data_queue)
        trainers.append(
            TrainingProcess(
                save_dir=save_dir,
                neural_net=network,
                experience_queues=training_queues[idx],
                batch_size=config.trainer_batch_size,
                num_agents=num_agents,
                data_queue=training_data_queue,
                weights_queues=weights_queues,
                min_num_batches_to_train=1,
                load_path=config.expert_data_path,
                stop_event=stop_event,
                weights_update_freq=config.weights_update_freq,
                generated_data_dir=config.generated_data_dir,
            )
        )

    return agents, evaluators, trainers, data_queues, stop_event
