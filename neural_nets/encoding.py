"""Chess position and move encoding: FEN/move <-> tensor, legal-move masks and batch tensorisation."""

import torch

from typing import Any, List, Optional, Sequence, Tuple

import chess_moves
import numpy as np


chess_engine = chess_moves.ChessEngine()


def create_legal_move_mask(
    moves: List[str], output_size: Sequence[int], team: Optional[str] = None
) -> Tuple[torch.Tensor, List[List[Any]]]:

    # Initiliase mvoe to indices look up
    move_to_indices_lookup: List[List[Any]] = []

    # Collect indices for batch updates
    all_indices: List[Tuple[int, int, int]] = []

    # Create mask efficiently with batched updates
    legal_move_mask = torch.zeros(output_size, dtype=torch.float32)

    if len(moves) > 1:
        for legal_move in moves:
            if team == "black":
                legal_move_str = chess_engine.unflip_move(str(legal_move))
                indices = chess_engine.move_to_target_indices(legal_move_str)
            else:
                indices = chess_engine.move_to_target_indices(str(legal_move))

            all_indices.append(indices)
            move_to_indices_lookup.append([legal_move, indices])

        if all_indices:  # only if we have legal moves
            idx_tensor = torch.tensor(all_indices, dtype=torch.long)  # (N,3)
            c, r, k = idx_tensor[:, 0], idx_tensor[:, 1], idx_tensor[:, 2]
            legal_move_mask[c, r, k] = 1.0  # set exactly those cells to 1
    else:
        legal_move_mask = legal_move_mask + 1.0

    return legal_move_mask, move_to_indices_lookup


def get_move_from_tensor(tensor: np.ndarray) -> str:
    # Returns the argmax move from the tensor
    channel = np.argmax(tensor) // (8 * 8)
    flat_index = np.argmax(tensor) % (8 * 8)
    from_row = flat_index // 8
    from_col = flat_index % 8

    return chess_engine.indices_to_move(int(channel), int(from_row), int(from_col))


def board_to_tensor(board: Any) -> np.ndarray:
    fen = board.fen()
    return chess_engine.fen_to_tensor(fen)


def tensorise_batch(
    states: List[str],
    moves: List[List[str]],
    probabilities: List[List[float]],
    wins: List[float],
    input_size: Sequence[int],
    output_size: Sequence[int],
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    state_tens, legal_move_mask, _ = tensorise_inputs(states, moves, input_size, output_size, device)

    batch_size = len(moves)
    moves_tens = torch.zeros((batch_size, *output_size), device=device)
    value_tens = torch.tensor(wins, dtype=torch.float32, device=device).unsqueeze(1)  # Vectorized

    # Precompute max possible size (assuming moves are relatively uniform in length)
    num_moves = sum(len(move_set) for move_set in moves)

    all_indices = np.empty((num_moves, 3), dtype=np.int64)
    all_probs = np.empty(num_moves, dtype=np.float32)
    batch_indices = np.empty(num_moves, dtype=np.int64)

    index = 0
    for idx, (state, move_set, prob_set) in enumerate(zip(states, moves, probabilities)):
        # The board is seen from the player to move, so black's targets are flipped like its legal move mask
        black_to_move = state.split()[1] == "b"
        for move, prob in zip(move_set, prob_set):
            move_str = chess_engine.unflip_move(str(move)) if black_to_move else str(move)
            all_indices[index] = chess_engine.move_to_target_indices(move_str)  # Vectorized storage
            all_probs[index] = prob
            batch_indices[index] = idx
            index += 1  # Move to next position

    if index > 0:  # Avoid empty scatter calls
        indices_tens = torch.from_numpy(all_indices[:index]).to(device, non_blocking=True)
        probs_tens = torch.from_numpy(all_probs[:index]).to(device, non_blocking=True)
        batch_indices_tens = torch.from_numpy(batch_indices[:index]).to(device, non_blocking=True)

        # Efficiently scatter values into moves_tens
        moves_tens[batch_indices_tens, indices_tens[:, 0], indices_tens[:, 1], indices_tens[:, 2]] = probs_tens

    return state_tens, moves_tens, value_tens, legal_move_mask


def tensorise_inputs(
    states: List[str],
    legal_moves: List[List[str]],
    input_size: Sequence[int],
    output_size: Sequence[int],
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, List[List[List[Any]]]]:
    """
    Take a set of states and create the neural netwokr input
    :param states:
    :param legal_moves:
    :return:
    """

    batch_size = len(states)

    # Preallocate tensors
    state_tens = torch.empty((batch_size, *input_size), device=device)
    legal_move_mask = torch.empty((batch_size, *output_size), device=device)

    # Preallocate NumPy arrays
    fen_tensors = np.empty((batch_size, *input_size), dtype=np.float32)
    move_masks = np.empty((batch_size, *output_size), dtype=np.float32)
    legal_move_keys: List[List[List[Any]]] = []

    for i, (state, move_set) in enumerate(zip(states, legal_moves)):
        # Convert FEN state to tensor
        fen_tensors[i] = chess_engine.fen_to_tensor(state)

        # Determine player turn
        player_turn = "white" if state.split()[1] == "w" else "black"

        # Generate legal move mask
        this_move_mask, key = create_legal_move_mask(move_set, output_size, player_turn)
        move_masks[i] = this_move_mask
        legal_move_keys.append(key)

    # Move all data to GPU in one batch transfer
    state_tens.copy_(torch.from_numpy(fen_tensors).to(device, non_blocking=True))
    legal_move_mask.copy_(torch.from_numpy(move_masks).to(device, non_blocking=True))

    return state_tens, legal_move_mask, legal_move_keys
