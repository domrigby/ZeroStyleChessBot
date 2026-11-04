"""The pygame window shared by human_vs_machine and engine_vs_engine: board, arrows, search panel and the game so far.

Every position of the game is kept with the last search of it, so the viewer can step back through the game (arrow
keys, Home/End, or a click on the graph) while the engines keep searching the live position. Analysis mode adds a graph
of the expected score over the game.
"""

import math
import threading
from typing import List, Optional, Sequence, Tuple

import chess
import pygame

from zero_style_chess_engine.config import DisplayMode
from zero_style_chess_engine.search_insights import SearchSummary
from zero_style_chess_engine.ui.search_panel import PANEL_WIDTH, SearchPanel, _scaled, draw_candidate_arrows

BOARD_SIZE = 400
SQUARE_SIZE = BOARD_SIZE // 8
MARGIN = 20
GRAPH_HEIGHT = 100

LIGHT_SQUARE = (240, 217, 181)
DARK_SQUARE = (181, 136, 99)
PIECE = (0, 0, 0)
GRAPH_BACKGROUND = (30, 30, 30)
GRAPH_LINE = (230, 230, 230)
GRAPH_MUTED = (100, 100, 100)
GRAPH_MARKER = (230, 180, 60)

Point = Tuple[float, float]
Colour = Tuple[int, int, int]
Arrow = Tuple[chess.Square, chess.Square, Colour]


def expected_score(summary: Optional[SearchSummary]) -> Optional[float]:
    """White's expected score in [0, 1] from a search: (1 + value) / 2. Draws and 50/50 positions both read 0.5"""
    if summary is None:
        return None
    return _scaled(summary.search_value, 1.0 if summary.white_to_move else -1.0)


class GameHistory:
    """Every position of the game, each with the latest search of it. Written by game and search threads, read by the UI"""

    def __init__(self, board: chess.Board) -> None:
        self._lock = threading.Lock()
        self._boards = [board.copy(stack=False)]
        self._summaries: List[Optional[SearchSummary]] = [None]

    def __len__(self) -> int:
        with self._lock:
            return len(self._boards)

    def push(self, board: chess.Board) -> None:
        """Add the position after a move"""
        with self._lock:
            self._boards.append(board.copy(stack=False))
            self._summaries.append(None)

    def record_search(self, summary: SearchSummary) -> None:
        """Keep a search of the latest position; a search of any other position (a stale one) is ignored"""
        with self._lock:
            # Placement and side to move only: the search's FEN and python-chess may differ on the other fields
            if summary.fen.split()[:2] == self._boards[-1].fen().split()[:2]:
                self._summaries[-1] = summary

    def board(self, ply: int) -> chess.Board:
        with self._lock:
            return self._boards[ply]

    def summary(self, ply: int) -> Optional[SearchSummary]:
        with self._lock:
            return self._summaries[ply]

    def expected_scores(self) -> List[Optional[float]]:
        with self._lock:
            return [expected_score(summary) for summary in self._summaries]


class Visualiser:
    """
    One window per game. The caller owns input and game logic: it passes events to handle_event first, calls draw
    each frame, draws its own extras on top, then flips. Moves should only be accepted while is_live
    :param left_width: space left of the board for the caller's own controls
    """

    def __init__(self, caption: str, display_mode: DisplayMode, show_search: bool, left_width: int = 0) -> None:
        pygame.init()  # Before any font is loaded, the search panel's included
        self.display_mode = display_mode
        self.board_rect = pygame.Rect(MARGIN + left_width, MARGIN, BOARD_SIZE, BOARD_SIZE)
        right = self.board_rect.right
        self.search_panel: Optional[SearchPanel] = None
        if show_search:
            self.search_panel = SearchPanel(pygame.Rect(right + MARGIN, MARGIN, PANEL_WIDTH, BOARD_SIZE))
            right = self.search_panel.rect.right
        self.width = right + MARGIN
        self.height = self.board_rect.bottom + MARGIN
        self.graph_rect: Optional[pygame.Rect] = None
        if display_mode == "analysis":
            self.graph_rect = pygame.Rect(self.board_rect.x, self.height, right - self.board_rect.x, GRAPH_HEIGHT)
            self.height = self.graph_rect.bottom + MARGIN

        self.history = GameHistory(chess.Board())
        self._view_ply: Optional[int] = None  # None follows the live position

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(caption)
        # Use a font that supports Unicode chess symbols: Apple Symbols on macOS, DejaVu Sans on Linux. Not bold: a
        # synthesised bold fills in the outlined white pieces.
        self.piece_font = pygame.font.SysFont("applesymbols,dejavusans", 36)
        self.small_font = pygame.font.SysFont("menlo,dejavusansmono,monospace", 12)

    @property
    def is_live(self) -> bool:
        return self._view_ply is None

    @property
    def viewed_ply(self) -> int:
        return len(self.history) - 1 if self._view_ply is None else self._view_ply

    def square_center(self, square: chess.Square) -> Point:
        return (
            self.board_rect.x + chess.square_file(square) * SQUARE_SIZE + SQUARE_SIZE / 2,
            self.board_rect.y + (7 - chess.square_rank(square)) * SQUARE_SIZE + SQUARE_SIZE / 2,
        )

    def square_at(self, pos: Tuple[int, int]) -> Optional[chess.Square]:
        if not self.board_rect.collidepoint(pos):
            return None
        file = (pos[0] - self.board_rect.x) // SQUARE_SIZE
        rank = 7 - (pos[1] - self.board_rect.y) // SQUARE_SIZE
        return chess.square(file, rank)

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Step through the game: left/right a ply, Home/End the start/live position, or a click on the graph"""
        last = len(self.history) - 1
        target: Optional[int] = None
        if event.type == pygame.KEYDOWN:
            keys = {pygame.K_LEFT: self.viewed_ply - 1, pygame.K_RIGHT: self.viewed_ply + 1, pygame.K_HOME: 0}
            target = last if event.key == pygame.K_END else keys.get(event.key)
        elif event.type == pygame.MOUSEBUTTONDOWN and self.graph_rect is not None:
            if self.graph_rect.collidepoint(event.pos):
                plot = self._plot_rect()
                target = round((event.pos[0] - plot.x) / plot.width * last)
        if target is None:
            return False
        target = max(0, min(target, last))
        self._view_ply = None if target == last else target
        return True

    def draw(
        self,
        live_board: chess.Board,
        live_summary: Optional[SearchSummary],
        arrows: Sequence[Arrow] = (),
        show_candidates: bool = False,
    ) -> None:
        """Draw the live position, or the position being viewed with the engine's last search of it"""
        self.screen.fill((0, 0, 0))
        board, summary = live_board, live_summary
        if not self.is_live:
            board, summary = self.history.board(self.viewed_ply), self.history.summary(self.viewed_ply)
            arrows = []  # They belong to the live position
            show_candidates = self.display_mode == "analysis"

        self._draw_board(board)
        for start, end, colour in arrows:
            draw_arrow(self.screen, self.square_center(start), self.square_center(end), colour, 4)
        if show_candidates:
            draw_candidate_arrows(self.screen, summary, self.square_center)
        if self.search_panel is not None:
            self.search_panel.draw(self.screen, summary)
        if self.graph_rect is not None:
            self._draw_graph(self.graph_rect)
        if not self.is_live:
            self._draw_banner()

    def _draw_board(self, board: chess.Board) -> None:
        for square in chess.SQUARES:
            x, y = self.square_center(square)
            light = (chess.square_file(square) + chess.square_rank(square)) % 2 == 1
            rect = pygame.Rect(0, 0, SQUARE_SIZE, SQUARE_SIZE)
            rect.center = (round(x), round(y))
            pygame.draw.rect(self.screen, LIGHT_SQUARE if light else DARK_SQUARE, rect)
            piece = board.piece_at(square)
            if piece:
                surface = self.piece_font.render(piece.unicode_symbol(), True, PIECE)
                self.screen.blit(surface, surface.get_rect(center=(x, y)))

    def _plot_rect(self) -> pygame.Rect:
        assert self.graph_rect is not None
        return pygame.Rect(self.graph_rect.x + 8, self.graph_rect.y + 20, self.graph_rect.width - 16, GRAPH_HEIGHT - 28)

    def _draw_graph(self, rect: pygame.Rect) -> None:
        pygame.draw.rect(self.screen, GRAPH_BACKGROUND, rect)
        self.screen.blit(self.small_font.render("Expected score (White)", True, GRAPH_LINE), (rect.x + 8, rect.y + 4))
        plot = self._plot_rect()
        pygame.draw.line(self.screen, GRAPH_MUTED, (plot.x, plot.centery), (plot.right, plot.centery))

        scores = self.history.expected_scores()
        last = len(scores) - 1

        def x_of(ply: int) -> float:
            return plot.x + plot.width * ply / max(last, 1)

        # Plies with no search (a UCI side, or no pondering) are skipped: the line joins the searched ones
        points = [
            (x_of(ply), plot.bottom - score * plot.height) for ply, score in enumerate(scores) if score is not None
        ]
        if len(points) >= 2:
            pygame.draw.lines(self.screen, GRAPH_LINE, False, points, 2)
        for point in points:
            pygame.draw.circle(self.screen, GRAPH_LINE, point, 2)
        marker_x = x_of(self.viewed_ply)
        pygame.draw.line(self.screen, GRAPH_MARKER, (marker_x, plot.y), (marker_x, plot.bottom), 1)

    def _draw_banner(self) -> None:
        text = f"Viewing ply {self.viewed_ply} of {len(self.history) - 1}: End returns to the game"
        surface = self.small_font.render(text, True, GRAPH_LINE)
        banner = pygame.Surface((self.board_rect.width, surface.get_height() + 8), pygame.SRCALPHA)
        banner.fill((0, 0, 0, 170))
        banner.blit(surface, (6, 4))
        self.screen.blit(banner, self.board_rect.topleft)


def draw_arrow(screen: pygame.Surface, start: Point, end: Point, colour: Colour, width: int) -> None:
    pygame.draw.line(screen, colour, start, end, width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    for side in (-math.pi / 6, math.pi / 6):
        tip = (end[0] - 10 * math.cos(angle + side), end[1] - 10 * math.sin(angle + side))
        pygame.draw.line(screen, colour, end, tip, width)
