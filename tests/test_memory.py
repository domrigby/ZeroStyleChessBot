import os
import pickle
import time
from pathlib import Path
from typing import List

import chess_moves
import numpy as np
import pytest

from zero_style_chess_engine.memory import DataPoint, Memory

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
GENERATED_DATA_DIRECTORY = "neural_nets/generated_data"


def _start_position_moves() -> List[str]:
    engine = chess_moves.ChessEngine()
    engine.set_fen(START_FEN)
    return engine.legal_moves()


def _game(length: int) -> List[dict]:
    """A fake game of identical start positions with spread-out visit counts (non-zero entropy)."""
    moves = _start_position_moves()
    visit_counts = np.arange(1, len(moves) + 1)
    return [{"state": START_FEN, "moves": moves, "visit_counts": visit_counts, "value": 0} for _ in range(length)]


@pytest.fixture
def empty_memory_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Memory loads and saves relative to the working directory."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_pickled_data_point_has_no_back_reference_to_its_game(empty_memory_directory: Path) -> None:
    game_length = 40
    standalone = DataPoint(START_FEN, _start_position_moves(), [0.05] * 20)
    memory = Memory(length_buffer=100)

    memory.save_game_to_memory(_game(game_length), agent_id=0)

    for data_point in memory.data:
        assert not hasattr(data_point, "parent_datapoint")
        assert not hasattr(data_point, "parent_datapont")
        # A back-reference would pickle the rest of the game too: ~game_length times bigger
        assert len(pickle.dumps(data_point)) < 2 * len(pickle.dumps(standalone))


@pytest.mark.parametrize("max_len, num_games", [
    pytest.param(5, 10, id="overfilled"),
    pytest.param(1, 3, id="single_slot"),
])
def test_buffer_never_exceeds_max_len(empty_memory_directory: Path, max_len: int, num_games: int) -> None:
    game_length = 6
    memory = Memory(length_buffer=max_len)

    for _ in range(num_games):
        memory.save_game_to_memory(_game(game_length), agent_id=0)
        assert len(memory) <= max_len

    assert len(memory) == max_len


def test_buffer_overwrites_oldest_data_point(empty_memory_directory: Path) -> None:
    max_len = 3
    memory = Memory(length_buffer=max_len)
    data_points = [DataPoint(START_FEN, [], [], win_val=float(idx)) for idx in range(max_len + 1)]

    for data_point in data_points:
        memory.add_data_point(data_point)

    assert data_points[0] not in memory.data
    assert all(data_point in memory.data for data_point in data_points[1:])


def test_get_batch_returns_requested_size_without_expert_data(empty_memory_directory: Path) -> None:
    batch_size = 8
    expert_ratio = 0.9
    memory = Memory(length_buffer=100)
    memory.save_game_to_memory(_game(10), agent_id=0)

    states, moves, probs, wins = memory.get_batch(batch_size, expert_ratio)

    assert len(states) == len(moves) == len(probs) == len(wins) == batch_size


def test_load_directory_keeps_only_the_most_recent_max_len_data_points(empty_memory_directory: Path) -> None:
    max_len = 4
    points_per_file = 3
    num_files = 3
    os.makedirs(GENERATED_DATA_DIRECTORY)
    for file_idx in range(num_files):
        file_points = [DataPoint(START_FEN, [], [], win_val=float(file_idx)) for _ in range(points_per_file)]
        file_path = os.path.join(GENERATED_DATA_DIRECTORY, f"data_{file_idx}.pkl")
        with open(file_path, "wb") as file:
            pickle.dump(file_points, file)
        modified_time = time.time() + file_idx
        os.utime(file_path, (modified_time, modified_time))

    memory = Memory(length_buffer=max_len)

    newest_file_idx = num_files - 1
    loaded_values = sorted(data_point.win_val for data_point in memory.data)
    assert len(memory) == max_len
    assert loaded_values == [newest_file_idx - 1] + [newest_file_idx] * points_per_file


def test_save_data_writes_only_unsaved_data_points(empty_memory_directory: Path) -> None:
    game_length = 6
    points_per_game = 2
    memory = Memory(length_buffer=100)
    first_file = "first.pkl"
    second_file = "second.pkl"

    memory.save_game_to_memory(_game(game_length), agent_id=0)
    memory.save_data(first_file)
    memory.save_game_to_memory(_game(game_length), agent_id=0)
    memory.save_data(second_file)

    for file_name in (first_file, second_file):
        with open(file_name, "rb") as file:
            assert len(pickle.load(file)) == points_per_game
