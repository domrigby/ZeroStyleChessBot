"""
Players: anything that picks a move for one side of a game, behind one interface (Player), and the game loop that
plays two of them against each other (play_game). The loop holds no UI code, so a headless match runner can reuse it.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Protocol

import chess
import chess.engine

from zero_style_chess_engine.config import (
    HeuristicMctsPlayerConfig,
    PlayerConfig,
    UciPlayerConfig,
    ZeroStylePlayerConfig,
)
from zero_style_chess_engine.engine_interface import EngineInterface
from zero_style_chess_engine.search_insights import SearchSummary


class Player(Protocol):
    name: str

    def choose_move(self, board: chess.Board) -> chess.Move:
        """
        Pick a move for the side to move. The board is a copy carrying the whole game so far (its move stack), so a
        player that keeps state between moves works out the moves played since its last call from it
        :raises PlayerResignedError: if the player gives up (or has no move to offer)
        """
        ...

    def close(self) -> None:
        """Release what the player holds (processes, engines). Safe to call mid-search from another thread"""
        ...


class PlayerResignedError(Exception):
    """Raised by choose_move when the player resigns: play_game scores the game as a loss for that side"""


class IllegalMoveError(Exception):
    """Raised by play_game when a player returns a move that is not legal in the position"""


@dataclass(frozen=True)
class GameResult:
    result: str  # "1-0", "0-1" or "1/2-1/2"
    reason: str  # chess.Termination name in lower case (e.g. "checkmate", "threefold_repetition"), or "resignation"


def play_game(
    white: Player,
    black: Player,
    board: Optional[chess.Board] = None,
    on_move: Optional[Callable[[chess.Board], None]] = None,
) -> GameResult:
    """
    Play a game to the end from the board given (default: the starting position; the caller's board is not changed).
    Draws by threefold repetition and the fifty-move rule are claimed, as engine-match tools do, so two engines
    repeating moves don't play on to fivefold repetition or the 75-move rule
    :param on_move: called with a copy of the board after every move
    :raises IllegalMoveError: if a player returns an illegal move
    """
    board = chess.Board() if board is None else board.copy()
    players = {chess.WHITE: white, chess.BLACK: black}

    while not board.is_game_over(claim_draw=True):
        player = players[board.turn]
        try:
            move = player.choose_move(board.copy())
        except PlayerResignedError:
            return GameResult("0-1" if board.turn == chess.WHITE else "1-0", "resignation")
        if move not in board.legal_moves:
            raise IllegalMoveError(f"{player.name} played the illegal move {move.uci()} in {board.fen()}")
        board.push(move)
        if on_move is not None:
            on_move(board.copy())

    outcome = board.outcome(claim_draw=True)
    assert outcome is not None  # The game is over
    return GameResult(outcome.result(), outcome.termination.name.lower())


class ZeroStylePlayer:
    """
    An MCTS engine: the zero-style network's, or the heuristic baseline's. Keeps its search tree between moves while
    it sees the game continue
    """

    def __init__(
        self, engine: EngineInterface, name: str, on_progress: Optional[Callable[[SearchSummary], None]] = None
    ) -> None:
        """:param on_progress: called from the searching thread with snapshots of the search"""
        self.engine = engine
        self.name = name
        self.on_progress = on_progress

    def choose_move(self, board: chess.Board) -> chess.Move:
        engine_board = self.engine.current_board
        num_seen = len(engine_board.move_stack)
        # Same starting position and the engine's moves are a prefix of the game's: catch up and keep the tree.
        # Comparing the moves alone isn't enough: from a custom FEN both stacks start empty
        if board.root() == engine_board.root() and board.move_stack[:num_seen] == engine_board.move_stack:
            for move in board.move_stack[num_seen:]:
                self.engine.apply_user_move(move.uci())
        else:
            self.engine.reset(board)

        move_uci, _ = self.engine.get_engine_move(self.on_progress)
        return chess.Move.from_uci(move_uci)

    def close(self) -> None:
        self.engine.close()


class UciPlayer:
    """Any UCI engine (Stockfish, Lc0, ...), run as a subprocess and given a fixed time per move"""

    def __init__(self, config: UciPlayerConfig, name: str) -> None:
        self.name = name
        self.move_time = config.move_time
        self.engine = chess.engine.SimpleEngine.popen_uci(config.engine_path)
        if config.elo is not None:
            if "UCI_Elo" not in self.engine.options:
                self.engine.quit()
                raise ValueError(
                    f"{config.engine_path} has no UCI_Elo option, so its strength can't be capped: "
                    f"pass {name}.elo=null to play it at full strength"
                )
            self.engine.configure({"UCI_LimitStrength": True, "UCI_Elo": config.elo})

    def choose_move(self, board: chess.Board) -> chess.Move:
        result = self.engine.play(board, chess.engine.Limit(time=self.move_time))
        if result.move is None or result.resigned:
            raise PlayerResignedError(f"{self.name} resigned")
        return result.move

    def close(self) -> None:
        self.engine.quit()


def make_player(
    config: PlayerConfig, name: str, on_progress: Optional[Callable[[SearchSummary], None]] = None
) -> Player:
    """
    Build the player a config describes
    :param name: the side's name, e.g. "white"; used in error messages and command-line hints
    :param on_progress: search snapshots, for players that report them (the MCTS ones); ignored by the others
    """
    if isinstance(config, ZeroStylePlayerConfig):
        engine = EngineInterface(
            network_path=config.network_path, num_rollouts=config.num_rollouts, move_selection=config.move_selection
        )
        return ZeroStylePlayer(engine, name, on_progress)
    if isinstance(config, HeuristicMctsPlayerConfig):
        engine = EngineInterface(
            value_function=config.value_function,
            num_rollouts=config.num_rollouts,
            move_selection=config.move_selection,
        )
        return ZeroStylePlayer(engine, name, on_progress)
    return UciPlayer(config, name)
