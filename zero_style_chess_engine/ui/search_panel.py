"""Pygame view of a SearchSummary: a side panel of search statistics and arrows for the moves being considered."""

import math
from typing import Callable, List, Optional, Tuple

import chess
import pygame

from zero_style_chess_engine.search_insights import SearchSummary

PANEL_WIDTH = 280
TOP_ARROWS = 3
PV_PLIES = 10

BACKGROUND = (30, 30, 30)
TEXT = (230, 230, 230)
MUTED = (150, 150, 150)
BAR_WHITE = (240, 240, 240)
BAR_BLACK = (60, 60, 60)
CANDIDATE_ARROW = (40, 170, 70)

Point = Tuple[float, float]
Colour = Tuple[int, int, int]


class SearchPanel:
    """Draws into a fixed rect. All values are shown from White's point of view: positive is good for White."""

    def __init__(self, rect: pygame.Rect) -> None:
        self.rect = rect
        self.font = pygame.font.SysFont("menlo,dejavusansmono,monospace", 13)
        self.heading_font = pygame.font.SysFont("arial", 16, bold=True)

    def draw(self, screen: pygame.Surface, summary: Optional[SearchSummary]) -> None:
        pygame.draw.rect(screen, BACKGROUND, self.rect)
        x, y = self.rect.x + 10, self.rect.y + 8
        y = self._blit(screen, "Engine search", (x, y), self.heading_font)
        if summary is None:
            self._blit(screen, "Waiting for the first search", (x, y + 4), colour=MUTED)
            return

        # Flips side-to-move values to White's point of view
        sign = 1.0 if summary.white_to_move else -1.0
        board = chess.Board(summary.fen)

        y = self._draw_eval_bar(screen, (x, y + 4), _scaled(summary.search_value, sign))
        y = self._blit(
            screen,
            f"Network {_signed(summary.net_value, sign)}  Search {_signed(summary.search_value, sign)}",
            (x, y),
        )
        reused = summary.total_visits - summary.rollouts_this_search
        y = self._blit(screen, f"Visits  {summary.total_visits} ({reused} reused)", (x, y))
        y = self._blit(screen, f"Speed   {summary.rollouts_per_second:.0f} rollouts/s", (x, y))
        y = self._blit(screen, f"Tree    {summary.tree_size} nodes, depth {summary.max_depth}", (x, y))

        if not summary.candidates:
            # Play mode strips the moves from a search on the human's move
            self._blit(screen, "Pondering your move: moves hidden", (x, y + 6), colour=MUTED)
            return

        y = self._blit(screen, f"{'Move':<8}{'Visits':>7}{'Q':>7}{'Prior':>7}", (x, y + 6), colour=MUTED)
        for candidate in summary.candidates:
            row = (
                f"{_figurine_san(board, chess.Move.from_uci(candidate.move)):<8}"
                f"{candidate.visit_share:>7.0%}{sign * candidate.q:>+7.2f}{candidate.prior:>7.0%}"
            )
            y = self._blit(screen, row, (x, y))

        y = self._blit(screen, "Expected line", (x, y + 6), colour=MUTED)
        for line in self._wrap(_numbered_san_line(board, summary.principal_variation[:PV_PLIES])):
            y = self._blit(screen, line, (x, y))

    def _draw_eval_bar(self, screen: pygame.Surface, top_left: Tuple[int, int], white_share: Optional[float]) -> int:
        width, height = self.rect.width - 20, 12
        bar = pygame.Rect(top_left[0], top_left[1], width, height)
        pygame.draw.rect(screen, BAR_BLACK, bar)
        if white_share is not None:
            pygame.draw.rect(screen, BAR_WHITE, pygame.Rect(bar.x, bar.y, round(width * white_share), height))
        return bar.bottom + 4

    def _blit(
        self,
        screen: pygame.Surface,
        text: str,
        pos: Tuple[int, int],
        font: Optional[pygame.font.Font] = None,
        colour: Colour = TEXT,
    ) -> int:
        font = font or self.font
        screen.blit(font.render(text, True, colour), pos)
        return pos[1] + font.get_linesize() + 2

    def _wrap(self, words: List[str]) -> List[str]:
        lines: List[str] = []
        for word in words:
            if lines and self.font.size(f"{lines[-1]} {word}")[0] <= self.rect.width - 20:
                lines[-1] = f"{lines[-1]} {word}"
            else:
                lines.append(word)
        return lines


def draw_candidate_arrows(
    screen: pygame.Surface, summary: Optional[SearchSummary], square_center: Callable[[chess.Square], Point]
) -> None:
    """Arrows for the most visited root moves, thicker for more visits."""
    if summary is None:
        return
    for candidate in summary.candidates[:TOP_ARROWS]:
        move = chess.Move.from_uci(candidate.move)
        width = 2 + round(10 * candidate.visit_share)
        _draw_arrow(screen, square_center(move.from_square), square_center(move.to_square), width)


def _draw_arrow(screen: pygame.Surface, start: Point, end: Point, width: int) -> None:
    pygame.draw.line(screen, CANDIDATE_ARROW, start, end, width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    head = 8 + width
    for side in (-math.pi / 6, math.pi / 6):
        tip = (end[0] - head * math.cos(angle + side), end[1] - head * math.sin(angle + side))
        pygame.draw.line(screen, CANDIDATE_ARROW, end, tip, width)


def _figurine_san(board: chess.Board, move: chess.Move) -> str:
    """SAN led by the moving piece's symbol, pawns included: Nf3 -> ♞f3, exd5 -> ♟exd5. Castling stays O-O."""
    san = board.san(move)
    piece_type = board.piece_type_at(move.from_square)
    if piece_type is None or board.is_castling(move):
        return san
    # Filled symbols for both sides: the outlined ones are hard to read at panel font size
    symbol = chess.Piece(piece_type, chess.BLACK).unicode_symbol()
    return symbol + (san if piece_type == chess.PAWN else san[1:])


def _numbered_san_line(board: chess.Board, moves: List[str]) -> List[str]:
    """['e2e4', 'e7e5'] from the start position -> ['1.', '♟e4', '♟e5']"""
    board = board.copy(stack=False)
    words: List[str] = []
    for index, uci in enumerate(moves):
        if board.turn == chess.WHITE:
            words.append(f"{board.fullmove_number}.")
        elif index == 0:
            words.append(f"{board.fullmove_number}...")
        move = chess.Move.from_uci(uci)
        words.append(_figurine_san(board, move))
        board.push(move)
    return words


def _signed(value: Optional[float], sign: float) -> str:
    return "  n/a" if value is None else f"{sign * value:+.2f}"


def _scaled(value: Optional[float], sign: float) -> Optional[float]:
    """A value in [-1, 1] as White's share of the eval bar"""
    return None if value is None else (1 + sign * value) / 2
