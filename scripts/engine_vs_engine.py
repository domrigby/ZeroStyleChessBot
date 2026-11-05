import threading
from typing import Callable, List, Optional

import pygame
import chess
import hydra
from omegaconf import DictConfig

from zero_style_chess_engine.config import (
    EngineVsEngineConfig,
    HeuristicMctsPlayerConfig,
    RandomPlayerConfig,
    UciPlayerConfig,
    ZeroStylePlayerConfig,
    check_paths_exist,
    from_hydra,
    register_defaults,
)
from zero_style_chess_engine.players import Player, make_player, play_game
from zero_style_chess_engine.search_insights import SearchSummary
from zero_style_chess_engine.ui.visualiser import Arrow, MARGIN, Visualiser

register_defaults("engine_vs_engine_defaults", EngineVsEngineConfig)
register_defaults("zero_style", ZeroStylePlayerConfig, group="player")
register_defaults("heuristic_mcts", HeuristicMctsPlayerConfig, group="player")
register_defaults("uci", UciPlayerConfig, group="player")
register_defaults("random", RandomPlayerConfig, group="player")


@hydra.main(version_base="1.3", config_path="../conf", config_name="engine_vs_engine")
def main(hydra_config: DictConfig) -> None:
    config = from_hydra(EngineVsEngineConfig, hydra_config)
    check_paths_exist(config)

    import multiprocessing as mp

    try:
        mp.set_start_method("spawn", force=True)
        print("spawned")
    except RuntimeError:
        print("Failed to spawn")
        pass

    # ---- Configuration ----
    FPS = 30

    # Colors
    ARROW_COLORS = {chess.WHITE: (255, 0, 0), chess.BLACK: (0, 0, 255)}  # Last move of each side
    TEXT_COLOR = (0, 0, 0)

    # ---- State shared with the game thread ----
    # Replaced whole by the game thread after every move, so drawing never reads a board being pushed onto
    board = chess.Board()
    # Last move of each side
    arrows: List[Arrow] = []
    # Latest snapshot of the search of whichever zero-style side is thinking; None between moves
    search_summary: Optional[SearchSummary] = None
    status: Optional[str] = None  # Result or error once the game ends
    error: Optional[Exception] = None
    closing = False  # Window closed: errors from players shutting down mid-move are expected, not reported

    # Both sides' searches are shown in full, so the window is in analysis mode whenever it shows them
    visualiser = Visualiser(
        f"{config.white.type} (White) vs {config.black.type} (Black)",
        "analysis" if config.show_search else "play",
        config.show_search,
    )

    # Shared by both sides: only the side to move searches, and on_move clears its summary once it has moved. Each side's
    # searches are recorded as its own, so the graph shows each engine's view of the game
    def search_progress_of(searcher: chess.Color) -> Callable[[SearchSummary], None]:
        def show_search_progress(summary: SearchSummary) -> None:
            nonlocal search_summary
            search_summary = summary
            visualiser.history.record_search(summary, searcher)

        return show_search_progress

    def on_move(new_board: chess.Board) -> None:
        nonlocal board, arrows, search_summary
        move = new_board.peek()
        mover = not new_board.turn
        if len(arrows) == 2:
            arrows = []
        arrows = [*arrows, (move.from_square, move.to_square, ARROW_COLORS[mover])]
        search_summary = None  # The last summary is for the position before this move
        visualiser.history.push(new_board)
        board = new_board

    # ---- Players ----
    def on_progress(side: chess.Color) -> Optional[Callable[[SearchSummary], None]]:
        return search_progress_of(side) if config.show_search else None

    players: List[Player] = []
    try:
        players.append(make_player(config.white, "white", on_progress(chess.WHITE), config.move_time))
        players.append(make_player(config.black, "black", on_progress(chess.BLACK), config.move_time))
    except Exception:
        for player in players:
            player.close()
        raise
    white, black = players

    # Background game loop
    def run_game() -> None:
        nonlocal status, error
        try:
            result = play_game(white, black, on_move=on_move)
            status = f"{result.result} ({result.reason})"
            print("Game over:", status)
        except Exception as exc:
            if closing:
                return
            error = exc
            status = f"Error: {exc}"

    # ---- Pygame Setup ----
    status_font = pygame.font.SysFont("Arial", 24)
    clock = pygame.time.Clock()

    # Start the game thread
    threading.Thread(target=run_game, daemon=True).start()

    # ---- Main Loop ----
    try:
        running = True
        while running:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                visualiser.handle_event(ev)

            shown = board  # One board for the whole frame, whatever the game thread does meanwhile

            # While a zero-style side thinks, show the moves it is weighing
            visualiser.draw(shown, search_summary, arrows, show_candidates=status is None)

            # Status line
            text = status if status is not None else f"{'White' if shown.turn else 'Black'} thinking..."
            txt = status_font.render(text, True, TEXT_COLOR)
            visualiser.screen.blit(txt, (MARGIN, visualiser.board_rect.bottom - 24))

            pygame.display.flip()
            clock.tick(FPS)
    finally:
        # Cleanup: stop evaluator processes and engine subprocesses, even mid-move
        closing = True
        for player in players:
            player.close()
        pygame.quit()

    if error is not None:
        raise error


if __name__ == "__main__":
    main()
