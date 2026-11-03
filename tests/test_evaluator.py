import multiprocessing.queues
import queue
from pathlib import Path
from typing import Any, List, cast

import chess_moves
import numpy as np

from neural_nets.conv_net import ChessNet
from zero_style_chess_engine.evaluator import Evaluator
from zero_style_chess_engine.messages import EvalRequest, EvalResult

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
# White's king is in check from the queen and its only legal move is to take it
ONE_LEGAL_MOVE_FEN = "7k/8/8/8/8/8/6q1/7K w - - 0 1"
AGENT_ID = 3


def _legal_moves(fen: str) -> List[str]:
    engine = chess_moves.ChessEngine()
    engine.set_fen(fen)
    return engine.legal_moves()


def test_priors_are_aligned_with_legal_moves_and_a_single_move_gets_all_the_prior(tmp_path: Path) -> None:
    # Thread queues stand in for the multiprocessing ones; setup() (compile) isn't needed to evaluate
    request_queue: "queue.Queue[Any]" = queue.Queue()
    result_queue: "queue.Queue[Any]" = queue.Queue()
    network = ChessNet(input_size=[12, 8, 8], output_size=[70, 8, 8], num_repeats=1, save_dir=str(tmp_path))
    evaluator = Evaluator(
        neural_net=network,
        eval_request_queues=[cast(multiprocessing.queues.Queue, request_queue)],
        eval_result_queues={AGENT_ID: cast(multiprocessing.queues.Queue, result_queue)},
    )
    requests = [
        EvalRequest(AGENT_ID, node_id=7, fen=START_FEN, legal_moves=_legal_moves(START_FEN)),
        EvalRequest(AGENT_ID, node_id=8, fen=ONE_LEGAL_MOVE_FEN, legal_moves=_legal_moves(ONE_LEGAL_MOVE_FEN)),
    ]
    for request in requests:
        request_queue.put(request)

    evaluator.collect_and_process()

    results: List[EvalResult] = result_queue.get_nowait()
    assert [result.node_id for result in results] == [7, 8]
    many_moves, one_move = results
    assert len(many_moves.priors) == len(requests[0].legal_moves)
    assert (many_moves.priors > 0).all()
    assert np.isclose(many_moves.priors.sum(), 1.0, atol=1e-5)
    assert one_move.priors.tolist() == [1.0]
