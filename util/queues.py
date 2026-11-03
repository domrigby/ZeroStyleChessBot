from multiprocessing import Queue, Event, synchronize
from torch import multiprocessing
import copy
from typing import List, Dict, Optional, Tuple
from neural_nets.conv_net import ChessNet
from neural_nets.data.fen_chess_puzzles.curriculum import load_curricula
from util.config import TrainingConfig
from zero_style_chess_engine.parallel_game_tree import GameTree
from zero_style_chess_engine.evaluator import Evaluator
from zero_style_chess_engine.trainer import Trainer


def create_agents(
    save_dir: str,
    num_agents: int,
    num_evaluators: int,
    num_trainers: int,
    network: ChessNet,
    training: bool = True,
    start_state: str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    verbose: bool = False,
    inference_mode_queue: Optional[Queue] = None,
    send_results_queue: Optional[List[Queue]] = None,
    config: Optional[TrainingConfig] = None,
) -> Tuple[List[GameTree], List[Evaluator], List[Trainer], Dict[str, List[Queue]], synchronize.Event]:
    """
    This function initialises all the agents and strings and distributes the queues accordingly
    TOOD: get rid of this
    :param num_agents: number of agents
    :param num_evaluators: number of evaluators
    :param num_trainers: number of trainers
    :param network: the neural network being used
    :param start_state
    :param config: paths and search/training knobs; defaults to TrainingConfig()
    :return: agents, evaluators, trainers, stats queues for the supervisor, and the stop event shared by every
        process (set it to make them all shut down)
    """

    if config is None:
        config = TrainingConfig()

    agents_per_evaluator = num_agents / num_evaluators

    if num_trainers:
        agents_per_trainer = num_agents / num_trainers

    # Start initializing stuff
    agents: List[GameTree] = []
    eval_result_queues: Dict[int, Queue] = {}
    eval_request_queues: List[List[Queue]] = [[] for _ in range(num_evaluators)]
    game_record_queues: List[List[Queue]] = [[] for _ in range(num_trainers)]
    stats_queues: Dict[str, List[Queue]] = {"agents": [], "training": []}
    stop_event = Event()

    # One weights queue per evaluator so every evaluator receives every update. maxsize=1: the trainer replaces an
    # update the evaluator hasn't picked up yet rather than queueing a backlog of state dicts.
    weights_queues: List[Queue] = [multiprocessing.Queue(maxsize=1) for _ in range(num_evaluators)]

    # Read the puzzle CSV once here; each agent gets only its own sample of positions
    curricula = load_curricula(config.curriculum_csv_path if training else None, num_agents)

    for idx in range(num_agents):
        eval_request_queue: Queue = Queue()
        eval_result_queue: Queue = Queue()

        # Create stats queue
        agent_stats_queue: Queue = Queue()
        stats_queues["agents"].append(agent_stats_queue)

        eval_request_queues[int(idx // agents_per_evaluator)].append(eval_request_queue)

        game_record_queue: Optional[Queue]
        if num_trainers:
            game_record_queue = Queue()
            game_record_queues[int(idx // agents_per_trainer)].append(game_record_queue)
        else:
            game_record_queue = None

        send_queue: Optional[Queue]
        if inference_mode_queue and send_results_queue:
            inference_mode = True
            send_queue = send_results_queue[idx]
        else:
            inference_mode = False
            send_queue = None

        agent = GameTree(
            save_dir=save_dir,
            agent_id=idx,
            training=training,
            eval_request_queue=eval_request_queue,
            game_record_queue=game_record_queue,
            eval_result_queue=eval_result_queue,
            stats_queue=agent_stats_queue,
            start_state=start_state,
            verbose=verbose,
            inference_mode=inference_mode,
            infering_mode_queue=inference_mode_queue,
            send_results_queue=send_queue,
            stop_event=stop_event,
            sims=config.sims,
            maximum_lag=config.maximum_lag,
            curriculum=curricula[idx],
            search_config=config.search,
        )

        agents.append(agent)
        eval_result_queues[agent.agent_id] = eval_result_queue

    # Evaluator
    evaluators: List[Evaluator] = []
    for idx in range(num_evaluators):
        # Each evaluator gets its own copy. Spawn pickling would otherwise put the trainer's and evaluators' parameters
        # in the same shared storage, so an evaluator's load_state_dict would roll back the trainer's weights.
        # TODO: costs one weights copy per evaluator. For large nets, replace with a single inference copy shared by all
        #  evaluators, which the trainer copies its weights into every weights_update_freq steps under an mp.Lock.
        evaluator = Evaluator(
            neural_net=copy.deepcopy(network),
            eval_request_queues=eval_request_queues[idx],
            weights_queue=weights_queues[idx],
            eval_result_queues=eval_result_queues,
            batch_size=config.evaluator_batch_size,
            stop_event=stop_event,
        )
        evaluators.append(evaluator)

    trainers: List[Trainer] = []
    for idx in range(num_trainers):
        trainer_stats_queue: Queue = Queue()
        stats_queues["training"].append(trainer_stats_queue)
        trainers.append(
            Trainer(
                save_dir=save_dir,
                neural_net=network,
                game_record_queues=game_record_queues[idx],
                batch_size=config.trainer_batch_size,
                num_agents=num_agents,
                stats_queue=trainer_stats_queue,
                weights_queues=weights_queues,
                min_num_batches_to_train=1,
                load_path=config.expert_data_path,
                stop_event=stop_event,
                weights_update_freq=config.weights_update_freq,
                generated_data_dir=config.generated_data_dir,
                init_lr=config.init_lr,
            )
        )

    return agents, evaluators, trainers, stats_queues, stop_event
