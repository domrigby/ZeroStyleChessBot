from pathlib import Path
from typing import Dict

import chess_moves
import numpy as np
import pytest
import torch

from neural_nets.encoding import tensorise_batch, tensorise_inputs

FIXTURES = Path(__file__).parent / "fixtures"
INPUT_SIZE = [12, 8, 8]
OUTPUT_SIZE = [70, 8, 8]
DEVICE = "cpu"
TARGET_VALUE = 0.5
POSITIONS = {
    "white_to_move": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "black_to_move": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
    "promotion": "8/4P3/8/8/8/8/8/k6K w - - 0 1",
}


def encode(fen: str) -> Dict[str, np.ndarray]:
    """Every tensor tensorise_inputs and tensorise_batch produce for one position, with distinct move targets."""
    engine = chess_moves.ChessEngine()
    engine.set_fen(fen)
    moves = engine.legal_moves()
    visits = np.arange(1, len(moves) + 1, dtype=np.float64)
    probabilities = list(visits / visits.sum())

    input_state, input_mask, _ = tensorise_inputs([fen], [moves], INPUT_SIZE, OUTPUT_SIZE, DEVICE)
    batch_state, batch_policy, batch_value, batch_mask = tensorise_batch(
        [fen], [moves], [probabilities], [TARGET_VALUE], INPUT_SIZE, OUTPUT_SIZE, DEVICE
    )
    tensors: Dict[str, torch.Tensor] = {
        "input_state": input_state,
        "input_mask": input_mask,
        "batch_state": batch_state,
        "batch_policy": batch_policy,
        "batch_value": batch_value,
        "batch_mask": batch_mask,
    }
    return {name: tensor.cpu().numpy() for name, tensor in tensors.items()}


@pytest.mark.parametrize("name", POSITIONS)
def test_encoding_matches_snapshot(name: str) -> None:
    """Snapshots were generated from the encoding before the architecture-cleanup refactor. Don't regenerate them."""
    encoded = encode(POSITIONS[name])

    with np.load(FIXTURES / f"encoding_{name}.npz") as snapshot:
        assert sorted(snapshot.files) == sorted(encoded)
        for key, expected in snapshot.items():
            np.testing.assert_array_equal(encoded[key], expected, err_msg=key)
