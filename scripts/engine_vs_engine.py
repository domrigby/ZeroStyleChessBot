import math
import threading
from typing import List, Optional, Tuple

import pygame
import chess
import hydra
from omegaconf import DictConfig

from zero_style_chess_engine.config import (
    EngineVsEngineConfig,
    HeuristicMctsPlayerConfig,
    UciPlayerConfig,
    ZeroStylePlayerConfig,
    check_paths_exist,
    from_hydra,
    register_defaults,
)
from zero_style_chess_engine.players import Player, make_player, play_game
from zero_style_chess_engine.search_insights import SearchSummary
from scripts.search_panel import PANEL_WIDTH, SearchPanel, draw_candidate_arrows

register_defaults("engine_vs_engine_defaults", EngineVsEngineConfig)
register_defaults("zero_style", ZeroStylePlayerConfig, group="player")
register_defaults("heuristic_mcts", HeuristicMctsPlayerConfig, group="player")
register_defaults("uci", UciPlayerConfig, group="player")


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
    BOARD_SIZE = 400
    SQUARE_SIZE = BOARD_SIZE // 8
    MARGIN = 20
    PANEL_X = BOARD_SIZE + 2 * MARGIN  # Search panel (if shown) to the right of the board
    WINDOW_W = PANEL_X + PANEL_WIDTH + MARGIN if config.show_search else BOARD_SIZE + 2 * MARGIN
    WINDOW_H = BOARD_SIZE + 2 * MARGIN
    FPS = 30

    # Colors
    LIGHT_SQ = (240, 217, 181)
    DARK_SQ = (181, 136, 99)
    ARROW_COLORS = {chess.WHITE: (255, 0, 0), chess.BLACK: (0, 0, 255)}  # Last move of each side
    TEXT_COLOR = (0, 0, 0)

    # ---- State shared with the game thread ----
    # Replaced whole by the game thread after every move, so drawing never reads a board being pushed onto
    board = chess.Board()
    # List of (from_xy, to_xy, color) for drawing arrows
    arrows: List[Tuple[Tuple[float, float], Tuple[float, float], Tuple[int, int, int]]] = []
    # Latest snapshot of the search of whichever zero-style side is thinking; None between moves
    search_summary: Optional[SearchSummary] = None
    status: Optional[str] = None  # Result or error once the game ends
    error: Optional[Exception] = None
    closing = False  # Window closed: errors from players shutting down mid-move are expected, not reported

    # Shared by both sides: only the side to move searches, and on_move clears its summary once it has moved
    def show_search_progress(summary: SearchSummary) -> None:
        nonlocal search_summary
        search_summary = summary

    # Utility: pixel center of a square index
    def square_center(sq: chess.Square) -> Tuple[float, float]:
        file = chess.square_file(sq)
        rank = 7 - chess.square_rank(sq)
        x = MARGIN + file * SQUARE_SIZE + SQUARE_SIZE / 2
        y = MARGIN + rank * SQUARE_SIZE + SQUARE_SIZE / 2
        return (x, y)

    def on_move(new_board: chess.Board) -> None:
        nonlocal board, arrows, search_summary
        move = new_board.peek()
        mover = not new_board.turn
        if len(arrows) == 2:
            arrows = []
        arrows = [*arrows, (square_center(move.from_square), square_center(move.to_square), ARROW_COLORS[mover])]
        search_summary = None  # The last summary is for the position before this move
        board = new_board

    # ---- Players ----
    on_progress = show_search_progress if config.show_search else None
    players: List[Player] = []
    try:
        players.append(make_player(config.white, "white", on_progress))
        players.append(make_player(config.black, "black", on_progress))
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
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption(f"{config.white.type} (White) vs {config.black.type} (Black)")
    # Use a font that supports Unicode chess symbols: Apple Symbols on macOS, DejaVu Sans on Linux. Not bold: a
    # synthesised bold fills in the outlined white pieces.
    piece_font = pygame.font.SysFont("applesymbols,dejavusans", 36)
    status_font = pygame.font.SysFont("Arial", 24)
    search_panel = SearchPanel(pygame.Rect(PANEL_X, MARGIN, PANEL_WIDTH, BOARD_SIZE)) if config.show_search else None
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

            shown = board  # One board for the whole frame, whatever the game thread does meanwhile

            # Draw board
            screen.fill((0, 0, 0))
            for r in range(8):
                for f in range(8):
                    col = LIGHT_SQ if (r + f) % 2 == 0 else DARK_SQ
                    rect = pygame.Rect(MARGIN + f * SQUARE_SIZE, MARGIN + r * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
                    pygame.draw.rect(screen, col, rect)

            # Draw pieces
            for sq in chess.SQUARES:
                p = shown.piece_at(sq)
                if p:
                    x, y = square_center(sq)
                    sym = p.unicode_symbol()
                    surf = piece_font.render(sym, True, TEXT_COLOR)
                    rect = surf.get_rect(center=(x, y))
                    screen.blit(surf, rect)

            # Draw move arrows
            for a, b, c in arrows:
                pygame.draw.line(screen, c, a, b, 4)
                ang = math.atan2(b[1] - a[1], b[0] - a[0])
                h = 10
                p1 = (b[0] - h * math.cos(ang - math.pi / 6), b[1] - h * math.sin(ang - math.pi / 6))
                p2 = (b[0] - h * math.cos(ang + math.pi / 6), b[1] - h * math.sin(ang + math.pi / 6))
                pygame.draw.line(screen, c, b, p1, 4)
                pygame.draw.line(screen, c, b, p2, 4)

            # While a zero-style side thinks, show the moves it is weighing
            if status is None:
                draw_candidate_arrows(screen, search_summary, square_center)
            if search_panel is not None:
                search_panel.draw(screen, search_summary)

            # Status line
            text = status if status is not None else f"{'White' if shown.turn else 'Black'} thinking..."
            txt = status_font.render(text, True, TEXT_COLOR)
            screen.blit(txt, (MARGIN, WINDOW_H - MARGIN - 24))

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
