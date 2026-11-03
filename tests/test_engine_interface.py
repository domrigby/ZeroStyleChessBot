import multiprocessing
import threading
from multiprocessing import Queue

import chess

from neural_nets.conv_net import ChessNet
from zero_style_chess_engine.evaluator import Evaluator
from zero_style_chess_engine.mcts import MCTS

AGENT_ID = 0
NUM_ROLLOUTS = 32
EVALUATOR_BATCH_SIZE = 8
JOIN_TIMEOUT_SECONDS = 30.0


def test_search_against_a_real_evaluator_returns_a_legal_move() -> None:
    """The human_vs_machine.py path: MCTS in this process, a real Evaluator (here a thread) answering it."""
    stop_event = multiprocessing.Event()
    eval_request_queue: Queue = Queue()
    eval_result_queue: Queue = Queue()
    mcts = MCTS(eval_request_queue, eval_result_queue, agent_id=AGENT_ID, stop_event=stop_event)
    evaluator = Evaluator(
        neural_net=ChessNet(input_size=[12, 8, 8], output_size=[70, 8, 8], num_repeats=1),
        eval_request_queues=[eval_request_queue],
        weights_queue=Queue(),
        eval_result_queues={AGENT_ID: eval_result_queue},
        batch_size=EVALUATOR_BATCH_SIZE,
        stop_event=stop_event,
    )
    evaluator_thread = threading.Thread(target=evaluator.run)
    evaluator_thread.start()
    try:
        mcts.search(NUM_ROLLOUTS)
        _, move, _ = mcts.select_greedy()
    finally:
        stop_event.set()
        evaluator_thread.join(JOIN_TIMEOUT_SECONDS)

    assert not evaluator_thread.is_alive()
    assert mcts.number_returned > 0
    assert chess.Move.from_uci(move) in chess.Board().legal_moves
