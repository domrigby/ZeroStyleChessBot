import random
import shutil
from typing import Any, Dict, List, Sequence

import chess
import chess.engine
import pytest

from tests.test_engine_interface import KINGS_ONLY_FEN, _engine_with_instant_evaluator
from zero_style_chess_engine.config import RandomPlayerConfig, UciPlayerConfig
from zero_style_chess_engine.players import (
    GameResult,
    IllegalMoveError,
    PlayerResignedError,
    RandomPlayer,
    UciPlayer,
    ZeroStylePlayer,
    make_player,
    play_game,
)

# White mates on move 4 (scholar's mate)
SCHOLARS_MATE_WHITE = ["e2e4", "f1c4", "d1h5", "h5f7"]
SCHOLARS_MATE_BLACK = ["e7e5", "b8c6", "g8f6"]
# Knights out and back twice: the starting position appears for the third time after the last move
REPETITION_WHITE = ["g1f3", "f3g1", "g1f3", "f3g1"]
REPETITION_BLACK = ["g8f6", "f6g8", "g8f6", "f6g8"]
# Kings and a white rook: white has mating material, black only king moves
ROOK_ENDGAME_FEN = "7k/8/8/8/8/8/8/K6R w - - 0 1"


class ScriptedPlayer:
    """Plays its moves in order and records every board it was shown; resigns when it runs out"""

    def __init__(self, name: str, moves: Sequence[str]) -> None:
        self.name = name
        self.moves = list(moves)
        self.boards_seen: List[chess.Board] = []

    def choose_move(self, board: chess.Board) -> chess.Move:
        self.boards_seen.append(board)
        if not self.moves:
            raise PlayerResignedError(self.name)
        return chess.Move.from_uci(self.moves.pop(0))

    def close(self) -> None:
        pass


def test_game_is_played_to_checkmate_with_sides_alternating() -> None:
    white = ScriptedPlayer("white", SCHOLARS_MATE_WHITE)
    black = ScriptedPlayer("black", SCHOLARS_MATE_BLACK)

    result = play_game(white, black)

    assert result == GameResult("1-0", "checkmate")
    assert [board.turn for board in white.boards_seen] == [chess.WHITE] * len(SCHOLARS_MATE_WHITE)
    assert [board.turn for board in black.boards_seen] == [chess.BLACK] * len(SCHOLARS_MATE_BLACK)
    # Each player sees the whole game so far
    assert [move.uci() for move in white.boards_seen[-1].move_stack] == [
        m for pair in zip(SCHOLARS_MATE_WHITE, SCHOLARS_MATE_BLACK) for m in pair
    ]


def test_illegal_move_raises_naming_the_player_the_move_and_the_position() -> None:
    white = ScriptedPlayer("white-net", ["e2e5"])

    with pytest.raises(IllegalMoveError) as error:
        play_game(white, ScriptedPlayer("black", []))

    assert "white-net" in str(error.value)
    assert "e2e5" in str(error.value)
    assert chess.STARTING_FEN in str(error.value)


@pytest.mark.parametrize(
    "white_moves, black_moves, expected",
    [
        pytest.param([], [], GameResult("0-1", "resignation"), id="white_resigns"),
        pytest.param(["e2e4"], [], GameResult("1-0", "resignation"), id="black_resigns"),
    ],
)
def test_resignation_loses_the_game(white_moves: List[str], black_moves: List[str], expected: GameResult) -> None:
    assert play_game(ScriptedPlayer("white", white_moves), ScriptedPlayer("black", black_moves)) == expected


def test_threefold_repetition_ends_the_game_as_a_draw() -> None:
    white = ScriptedPlayer("white", REPETITION_WHITE + ["e2e4"])

    result = play_game(white, ScriptedPlayer("black", REPETITION_BLACK))

    assert result == GameResult("1/2-1/2", "threefold_repetition")
    assert white.moves == ["e2e4"]  # The game ended before white's next move


def test_game_starts_from_the_board_given_without_changing_it() -> None:
    start = chess.Board(ROOK_ENDGAME_FEN)
    white = ScriptedPlayer("white", ["h1h2"])

    play_game(white, ScriptedPlayer("black", []), board=start)

    assert white.boards_seen[0].fen() == ROOK_ENDGAME_FEN
    assert start.fen() == ROOK_ENDGAME_FEN and not start.move_stack


def test_on_move_gets_a_copy_after_every_move() -> None:
    boards: List[chess.Board] = []

    play_game(
        ScriptedPlayer("white", SCHOLARS_MATE_WHITE),
        ScriptedPlayer("black", SCHOLARS_MATE_BLACK),
        on_move=boards.append,
    )

    assert [len(board.move_stack) for board in boards] == list(range(1, 8))
    assert len({id(board) for board in boards}) == len(boards)
    assert boards[-1].is_checkmate()


# ---- ZeroStylePlayer, searching against the instant test evaluator ----


def _zero_style_player(rollouts: int = 200) -> ZeroStylePlayer:
    return ZeroStylePlayer(_engine_with_instant_evaluator(rollouts), name="net")


def _searched_reply(player: ZeroStylePlayer) -> chess.Move:
    """The opponent's reply the engine searched most below its own last move"""
    root = player.engine.mcts.root
    reply_idx = max(root.child_nodes, key=lambda idx: root.Ns[idx])
    return chess.Move.from_uci(root.moves[reply_idx])


def test_zero_style_player_returns_a_legal_move() -> None:
    board = chess.Board(KINGS_ONLY_FEN)

    assert _zero_style_player().choose_move(board) in board.legal_moves


def test_zero_style_player_keeps_its_tree_after_the_opponents_move() -> None:
    player = _zero_style_player()
    board = chess.Board(KINGS_ONLY_FEN)  # The position the engine was set up in
    board.push(player.choose_move(board.copy()))
    reply = _searched_reply(player)
    reply_node = player.engine.mcts.root.child_nodes[player.engine.mcts.root.moves.index(reply.uci())]
    visits_before = reply_node.Ns.sum()
    board.push(reply)

    player.choose_move(board.copy())

    # The second search ran on from the reply's subtree, so the first search's visits carried over
    assert visits_before > 0
    assert reply_node.Ns.sum() == visits_before + player.engine.rollouts


def test_zero_style_player_keeps_its_tree_on_the_second_move_after_a_custom_start() -> None:
    player = _zero_style_player()
    board = chess.Board(ROOK_ENDGAME_FEN)  # Not the position the engine was set up in: it resets
    board.push(player.choose_move(board.copy()))
    reply = _searched_reply(player)
    reply_node = player.engine.mcts.root.child_nodes[player.engine.mcts.root.moves.index(reply.uci())]
    visits_before = reply_node.Ns.sum()
    board.push(reply)

    player.choose_move(board.copy())

    assert visits_before > 0
    assert reply_node.Ns.sum() == visits_before + player.engine.rollouts


@pytest.mark.parametrize(
    "board",
    [
        pytest.param(chess.Board(ROOK_ENDGAME_FEN), id="different_start_position"),
        pytest.param(chess.Board(), id="new_game"),
    ],
)
def test_zero_style_player_resets_when_the_game_is_not_the_one_it_saw(board: chess.Board) -> None:
    player = _zero_style_player()
    seen = chess.Board(KINGS_ONLY_FEN)
    seen.push(player.choose_move(seen.copy()))
    old_root = player.engine.mcts.root

    move = player.choose_move(board.copy())

    assert move in board.legal_moves
    assert player.engine.mcts.root is not old_root
    assert player.engine.current_board.root() == board.root()


def test_zero_style_player_resets_for_a_custom_start_before_any_move() -> None:
    # Both move stacks are empty: only the starting positions differ
    player = _zero_style_player()

    player.choose_move(chess.Board(ROOK_ENDGAME_FEN))

    assert player.engine.current_board.root().fen() == ROOK_ENDGAME_FEN


def test_zero_style_player_resets_when_the_history_differs() -> None:
    player = _zero_style_player()
    board = chess.Board(ROOK_ENDGAME_FEN)
    played = player.choose_move(board.copy())
    board.push(next(move for move in map(chess.Move.from_uci, ["h1h2", "h1h3"]) if move != played))
    board.push(chess.Move.from_uci("h8g8"))

    move = player.choose_move(board.copy())

    assert move in board.legal_moves
    assert player.engine.current_board.move_stack[:2] == board.move_stack


# ---- UciPlayer ----


class _FakeUciEngine:
    def __init__(self, options: Dict[str, Any]) -> None:
        self.options = options
        self.quit_called = False

    def quit(self) -> None:
        self.quit_called = True


def test_elo_cap_on_an_engine_without_uci_elo_says_to_pass_elo_null(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeUciEngine(options={"Threads": object()})
    monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci", lambda path: fake)

    with pytest.raises(ValueError, match=r"black\.elo=null"):
        UciPlayer(UciPlayerConfig(engine_path="lc0", elo=1800), name="black")
    assert fake.quit_called


@pytest.mark.skipif(shutil.which("stockfish") is None, reason="Stockfish is not on $PATH")
def test_stockfish_plays_a_short_game_against_itself() -> None:
    config = UciPlayerConfig(engine_path="stockfish", elo=1350, move_time=0.01)
    white, black = UciPlayer(config, "white"), UciPlayer(config, "black")
    try:
        result = play_game(white, black, board=chess.Board(ROOK_ENDGAME_FEN))
    finally:
        white.close()
        black.close()

    assert result.result in ("1-0", "1/2-1/2")


def test_random_player_plays_random_games_to_the_end() -> None:
    # Legal throughout (play_game raises otherwise) and ends with a recognised result
    for seed in range(5):
        white, black = RandomPlayer("white", random.Random(seed)), RandomPlayer("black", random.Random(seed + 100))
        result = play_game(white, black)
        assert result.result in {"1-0", "0-1", "1/2-1/2"}


def test_random_player_picks_among_all_legal_moves() -> None:
    board = chess.Board()
    player = RandomPlayer("white", random.Random(0))

    moves = {player.choose_move(board) for _ in range(400)}

    assert moves == set(board.legal_moves)  # All 20 opening moves, given enough draws


def test_make_player_builds_a_random_player() -> None:
    assert isinstance(make_player(RandomPlayerConfig(), "black"), RandomPlayer)
