import multiprocessing
import multiprocessing.queues
import queue
import threading
from multiprocessing import Queue
from typing import Any, cast

import chess

from neural_nets.conv_net import ChessNet
from tests.test_search import InstantEvaluator
from zero_style_chess_engine.engine_interface import EngineInterface
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


# Kings only: three moves each, so with uniform priors every engine move is searched several plies deep
KINGS_ONLY_FEN = "7k/8/8/8/8/8/8/K7 w - - 0 1"


def _engine_with_instant_evaluator(rollouts: int) -> EngineInterface:
    """An EngineInterface searching against the instant test evaluator: no network or evaluator process"""
    results_queue: "queue.Queue[Any]" = queue.Queue()
    engine = EngineInterface.__new__(EngineInterface)
    engine.mcts = MCTS(
        cast(multiprocessing.queues.Queue, InstantEvaluator(results_queue)),
        cast(multiprocessing.queues.Queue, results_queue),
        agent_id=AGENT_ID,
        stop_event=multiprocessing.Event(),
    )
    engine.mcts.reset(KINGS_ONLY_FEN)
    engine.current_board = chess.Board(KINGS_ONLY_FEN)
    engine.rollouts = rollouts
    return engine


def test_searched_reply_keeps_its_subtree() -> None:
    engine = _engine_with_instant_evaluator(rollouts=200)
    start_root = engine.mcts.root

    engine_move, _ = engine.get_engine_move()
    engine_child = start_root.child_nodes[start_root.moves.index(engine_move)]
    reply_idx = max(engine_child.child_nodes, key=lambda idx: engine_child.Ns[idx])
    reply = engine_child.child_nodes[reply_idx]

    assert engine.apply_user_move(engine_child.moves[reply_idx])

    assert engine.mcts.root is reply
    assert engine.mcts.root.Ns.sum() > 0


def test_unsearched_reply_starts_a_fresh_tree_from_the_board() -> None:
    # One rollout: the engine's move has no searched replies under it
    engine = _engine_with_instant_evaluator(rollouts=1)
    engine.get_engine_move()

    assert engine.apply_user_move("h8g8")

    assert engine.mcts.root.state == engine.current_board.fen()
    assert engine.mcts.root.Ns.sum() == 0
