import dataclasses
import sys
import threading
import pygame
import chess
import hydra
from omegaconf import DictConfig
from zero_style_chess_engine.config import HumanVsMachineConfig, check_paths_exist, from_hydra, register_defaults
from zero_style_chess_engine.engine_interface import EngineInterface  # Your engine interface module
from zero_style_chess_engine.search_insights import SearchSummary
from zero_style_chess_engine.ui.visualiser import MARGIN, Visualiser, draw_arrow
import multiprocessing as mp
from typing import Optional

register_defaults("human_vs_machine_defaults", HumanVsMachineConfig)


@hydra.main(version_base="1.3", config_path="../conf", config_name="human_vs_machine")
def main(hydra_config: DictConfig) -> None:
    config = from_hydra(HumanVsMachineConfig, hydra_config)
    check_paths_exist(config)

    try:
        mp.set_start_method("spawn", force=True)
        print("spawned")
    except RuntimeError:
        print("Failed to spawn")
        pass

    # ---- Configuration ----
    # Layout configuration
    BUTTON_WIDTH = 120  # Width of the left button panel
    BUTTON_HEIGHT = 40  # Button height
    BUTTON_SPACING = 20  # Vertical space between buttons

    BOARD_Y = MARGIN

    # Colors
    ARROW_COLOR = (255, 0, 0)
    ENGINE_ARROW_COLOR = (0, 0, 255)
    ENGINE_SIDE = chess.BLACK  # The human moves first
    BUTTON_COLOR = (70, 130, 180)
    BUTTON_TEXT_COLOR = (255, 255, 255)
    THINKING_COLOR = (255, 0, 0)

    # ---- Engine Initialization ----
    engine = EngineInterface(
        network_path=config.network_path, num_rollouts=config.num_rollouts, move_selection=config.move_selection
    )

    # ---- Pygame Initialization ----
    # The board sits right of the button panel
    visualiser = Visualiser("Play vs Chess Engine", config.display_mode, config.show_search, BUTTON_WIDTH + MARGIN)
    screen = visualiser.screen
    WINDOW_WIDTH, WINDOW_HEIGHT = visualiser.width, visualiser.height
    # The graph is drawn from the searches, so analysis mode reports them even without the panel
    track_search = config.show_search or config.display_mode == "analysis"

    button_font = pygame.font.SysFont("Arial", 24)
    status_font = pygame.font.SysFont("Arial", 28)

    # ---- Global Variables ----
    dragging = False
    start_square = None
    end_square = None
    current_drag_pos = None

    engine_thinking = False
    engine_arrow = None  # Tuple: (engine_move_start, engine_move_end)
    search_summary: Optional[SearchSummary] = None  # Latest snapshot of the engine's search, set by its thread
    ponder_summary: Optional[SearchSummary] = None  # Latest snapshot of the search on the human's move, while pondering

    clock = pygame.time.Clock()
    running = True

    # ---- Define Button Rectangles (vertical stack on left) ----
    # Placed in the left panel (x from MARGIN to MARGIN+BUTTON_WIDTH).
    queenside_button_rect = pygame.Rect(MARGIN, BOARD_Y, BUTTON_WIDTH, BUTTON_HEIGHT)
    confirm_button_rect = pygame.Rect(MARGIN, BOARD_Y + BUTTON_HEIGHT + BUTTON_SPACING, BUTTON_WIDTH, BUTTON_HEIGHT)
    kingside_button_rect = pygame.Rect(
        MARGIN, BOARD_Y + 2 * (BUTTON_HEIGHT + BUTTON_SPACING), BUTTON_WIDTH, BUTTON_HEIGHT
    )

    # ---- Utility Functions ----

    def draw_button(screen: pygame.Surface, text: str, rect: pygame.Rect) -> None:
        """Draw a simple button with the given text."""
        pygame.draw.rect(screen, BUTTON_COLOR, rect)
        text_surf = button_font.render(text, True, BUTTON_TEXT_COLOR)
        text_rect = text_surf.get_rect(center=rect.center)
        screen.blit(text_surf, text_rect)

    def display_endgame_box(message: str) -> None:
        """Display a semi‑transparent overlay box with an endgame message."""
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        box_rect = pygame.Rect(WINDOW_WIDTH // 4, WINDOW_HEIGHT // 3, WINDOW_WIDTH // 2, WINDOW_HEIGHT // 3)
        pygame.draw.rect(overlay, (255, 255, 255), box_rect)
        text_surf = status_font.render(message, True, (0, 0, 0))
        text_rect = text_surf.get_rect(center=box_rect.center)
        overlay.blit(text_surf, text_rect)
        screen.blit(overlay, (0, 0))
        pygame.display.flip()

    def get_promotion_choice() -> str:
        """Display an overlay for promotion choice and return the chosen letter."""
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        options = [("Queen", "q"), ("Rook", "r"), ("Bishop", "b"), ("Knight", "n")]
        num_options = len(options)
        option_width = 120
        option_height = 50
        margin = 20
        total_width = num_options * option_width + (num_options - 1) * margin
        start_x = (WINDOW_WIDTH - total_width) // 2
        y = (WINDOW_HEIGHT - option_height) // 2
        option_rects = []
        for i, (name, letter) in enumerate(options):
            rect = pygame.Rect(start_x + i * (option_width + margin), y, option_width, option_height)
            option_rects.append((rect, letter))
            pygame.draw.rect(overlay, BUTTON_COLOR, rect)
            text = button_font.render(name, True, BUTTON_TEXT_COLOR)
            text_rect = text.get_rect(center=rect.center)
            overlay.blit(text, text_rect)
        screen.blit(overlay, (0, 0))
        pygame.display.flip()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONDOWN:
                    pos = event.pos
                    for rect, letter in option_rects:
                        if rect.collidepoint(pos):
                            return letter
                elif event.type == pygame.QUIT:
                    pygame.quit()
                    engine.close()
                    sys.exit()

    # ---- Thread Function for Engine Move ----
    def show_search_progress(summary: SearchSummary) -> None:
        nonlocal search_summary
        search_summary = summary
        visualiser.history.record_search(summary, ENGINE_SIDE)

    def show_ponder_progress(summary: SearchSummary) -> None:
        nonlocal ponder_summary
        if config.display_mode == "play":
            # The moves searched are the human's candidates: showing them would be a hint
            summary = dataclasses.replace(summary, candidates=[], principal_variation=[])
        ponder_summary = summary
        visualiser.history.record_search(summary, ENGINE_SIDE)

    def start_pondering() -> None:
        if config.ponder:
            engine.start_pondering(config.ponder_max_rollouts, show_ponder_progress if track_search else None)

    def process_engine_move(user_move_str: str) -> None:
        nonlocal start_square, end_square, dragging, current_drag_pos, engine_thinking, engine_arrow, search_summary
        nonlocal ponder_summary
        engine.stop_pondering()  # The search tree is about to change under it
        ponder_summary = None
        if engine.apply_user_move(user_move_str):
            print("User move:", user_move_str)
            visualiser.history.push(engine.current_board)
            search_summary = None  # The last summary is for the previous position
            eng_move, _ = engine.get_engine_move(on_progress=show_search_progress if track_search else None)
            visualiser.history.push(engine.current_board)
            print("Engine move:", eng_move)
            if len(eng_move) >= 4:
                eng_from = eng_move[0:2]
                eng_to = eng_move[2:4]
                engine_arrow = (eng_from, eng_to)
        else:
            print("Invalid move:", user_move_str)
        start_pondering()  # The human is to move again either way
        start_square = None
        end_square = None
        dragging = False
        current_drag_pos = None
        engine_thinking = False

    start_pondering()  # The human moves first

    # ---- Main Loop ----
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif visualiser.handle_event(event):
                pass

            # Moves only on the live board, not while looking back through the game
            elif event.type == pygame.MOUSEBUTTONDOWN and visualiser.is_live:
                x, y = event.pos
                clicked = visualiser.square_at((x, y))
                # Check if click is in the board area and engine is not thinking.
                if clicked is not None and not engine_thinking:
                    start_square = chess.square_name(clicked)
                    dragging = True
                    current_drag_pos = (x, y)
                # Else if click is in the left panel area, check buttons.
                elif x < visualiser.board_rect.x:
                    if queenside_button_rect.collidepoint(x, y) and not engine_thinking:
                        if engine.current_board.turn:  # White's turn.
                            move_str = "e1c1"
                        else:
                            move_str = "e8c8"
                        engine_thinking = True
                        threading.Thread(target=process_engine_move, args=(move_str,)).start()
                    elif confirm_button_rect.collidepoint(x, y) and not engine_thinking:
                        if start_square and end_square:
                            move_str = start_square + end_square
                            # Check for promotion.
                            sq = chess.parse_square(start_square)
                            piece = engine.current_board.piece_at(sq)
                            if piece and piece.piece_type == chess.PAWN:
                                if (piece.color and end_square[1] == "8") or (not piece.color and end_square[1] == "1"):
                                    promotion_letter = get_promotion_choice()
                                    move_str += promotion_letter
                            engine_thinking = True
                            threading.Thread(target=process_engine_move, args=(move_str,)).start()
                    elif kingside_button_rect.collidepoint(x, y) and not engine_thinking:
                        if engine.current_board.turn:  # White.
                            move_str = "e1g1"
                        else:
                            move_str = "e8g8"
                        engine_thinking = True
                        threading.Thread(target=process_engine_move, args=(move_str,)).start()

            elif event.type == pygame.MOUSEBUTTONUP:
                if dragging:
                    # Only register the drop if it's within the board area.
                    dropped = visualiser.square_at(event.pos)
                    if dropped is not None:
                        end_square = chess.square_name(dropped)
                    dragging = False

            elif event.type == pygame.MOUSEMOTION:
                if dragging:
                    current_drag_pos = event.pos

        # ---- Drawing Section ----
        # Draw engine's move arrow.
        arrows = []
        if engine_arrow:
            arrows.append(
                (chess.parse_square(engine_arrow[0]), chess.parse_square(engine_arrow[1]), ENGINE_ARROW_COLOR)
            )

        # While the engine thinks, show the moves it is weighing; afterwards the panel explains the move it chose
        # While pondering, the panel follows the search on the human's move instead. Only analysis mode draws arrows
        shown_summary = search_summary if engine_thinking else ponder_summary or search_summary
        show_candidates = config.display_mode == "analysis" and (engine_thinking or ponder_summary is not None)
        visualiser.draw(engine.current_board, shown_summary, arrows, show_candidates)

        # Draw the user drag arrow if one exists.
        if visualiser.is_live and start_square and current_drag_pos:
            start_center = visualiser.square_center(chess.parse_square(start_square))
            draw_arrow(screen, start_center, current_drag_pos, ARROW_COLOR, 3)

        # Draw the buttons in the left panel.
        draw_button(screen, "Queenside Castle", queenside_button_rect)
        draw_button(screen, "Confirm Move", confirm_button_rect)
        draw_button(screen, "Kingside Castle", kingside_button_rect)

        # If the engine is thinking, display a message below the buttons.
        if engine_thinking:
            thinking_text = status_font.render("Thinking...", True, THINKING_COLOR)
            screen.blit(thinking_text, (MARGIN, BOARD_Y + 3 * (BUTTON_HEIGHT + BUTTON_SPACING)))

        # Check for checkmate.
        if engine.current_board.is_checkmate():
            display_endgame_box("Checkmate! Press any key to exit.")
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN or event.type == pygame.QUIT:
                        waiting = False
                clock.tick(30)
            running = False

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    engine.close()
    sys.exit()


if __name__ == "__main__":
    main()
