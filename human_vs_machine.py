import sys, math, threading, pygame, chess
from zero_style_chess_engine.engine_interface import EngineInterface  # Your engine interface module
import multiprocessing as mp

if __name__ == '__main__':
    try:
        mp.set_start_method('spawn', force=True)
        print("spawned")
    except RuntimeError:
        print("Failed to spawn")
        pass

    # ---- Configuration ----
    BOARD_SIZE = 400  # Board is 400x400 pixels
    SQUARE_SIZE = BOARD_SIZE // 8  # Each square is 50x50 pixels

    # Layout configuration
    MARGIN = 20  # General margin
    BUTTON_WIDTH = 120  # Width of the left button panel
    BUTTON_HEIGHT = 40  # Button height
    BUTTON_SPACING = 20  # Vertical space between buttons

    # Board offset: board is drawn to the right of the button panel
    BOARD_X = MARGIN + BUTTON_WIDTH + MARGIN  # Board starts after left panel + margin
    BOARD_Y = MARGIN

    WINDOW_WIDTH = BOARD_X + BOARD_SIZE + MARGIN  # Right margin at end of board
    WINDOW_HEIGHT = BOARD_Y + BOARD_SIZE + MARGIN  # Bottom margin

    # Colors
    LIGHT_SQUARE = (240, 217, 181)
    DARK_SQUARE = (181, 136, 99)
    ARROW_COLOR = (255, 0, 0)
    ENGINE_ARROW_COLOR = (0, 0, 255)
    BUTTON_COLOR = (70, 130, 180)
    BUTTON_TEXT_COLOR = (255, 255, 255)
    THINKING_COLOR = (255, 0, 0)

    # ---- Engine Initialization ----
    engine = EngineInterface(
        network_path=r"neural_nets/example_network.pt",
        parallel=False
    )

    # ---- Pygame Initialization ----
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Play vs Chess Engine")

    # Use a font that supports Unicode chess symbols; pieces will be bold.
    piece_font = pygame.font.SysFont("DejaVu Sans", 36, bold=True)
    button_font = pygame.font.SysFont("Arial", 24)
    status_font = pygame.font.SysFont("Arial", 28)

    # ---- Global Variables ----
    dragging = False
    start_square = None
    end_square = None
    current_drag_pos = None

    engine_thinking = False
    engine_arrow = None  # Tuple: (engine_move_start, engine_move_end)

    clock = pygame.time.Clock()
    running = True

    # ---- Define Button Rectangles (vertical stack on left) ----
    # Placed in the left panel (x from MARGIN to MARGIN+BUTTON_WIDTH).
    queenside_button_rect = pygame.Rect(MARGIN, BOARD_Y, BUTTON_WIDTH, BUTTON_HEIGHT)
    confirm_button_rect = pygame.Rect(MARGIN, BOARD_Y + BUTTON_HEIGHT + BUTTON_SPACING, BUTTON_WIDTH, BUTTON_HEIGHT)
    kingside_button_rect = pygame.Rect(MARGIN, BOARD_Y + 2 * (BUTTON_HEIGHT + BUTTON_SPACING), BUTTON_WIDTH, BUTTON_HEIGHT)


    # ---- Utility Functions ----

    def draw_board(screen, board):
        """Draw the chessboard and pieces using the current state from python‑chess."""
        # Draw squares with board offset.
        for row in range(8):
            for col in range(8):
                color = LIGHT_SQUARE if (row + col) % 2 == 0 else DARK_SQUARE
                rect = pygame.Rect(BOARD_X + col * SQUARE_SIZE, BOARD_Y + row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
                pygame.draw.rect(screen, color, rect)
        # Draw pieces.
        for row in range(8):
            for col in range(8):
                square_name = chr(ord('a') + col) + str(8 - row)
                try:
                    sq = chess.parse_square(square_name)
                except Exception:
                    continue
                piece = board.piece_at(sq)
                if piece:
                    symbol = piece.unicode_symbol()
                    text_surface = piece_font.render(symbol, True, (0, 0, 0))
                    text_rect = text_surface.get_rect(
                        center=(BOARD_X + col * SQUARE_SIZE + SQUARE_SIZE / 2,
                                BOARD_Y + row * SQUARE_SIZE + SQUARE_SIZE / 2)
                    )
                    screen.blit(text_surface, text_rect)


    def draw_button(screen, text, rect):
        """Draw a simple button with the given text."""
        pygame.draw.rect(screen, BUTTON_COLOR, rect)
        text_surf = button_font.render(text, True, BUTTON_TEXT_COLOR)
        text_rect = text_surf.get_rect(center=rect.center)
        screen.blit(text_surf, text_rect)


    def pixel_to_square(x, y):
        """Convert window pixel coordinates to a chess square in algebraic notation (accounting for board offset)."""
        col = int((x - BOARD_X) // SQUARE_SIZE)
        row = int((y - BOARD_Y) // SQUARE_SIZE)
        return chr(ord('a') + col) + str(8 - row)


    def square_center(square):
        """Return the pixel center (x,y) of the given chess square (e.g. 'e2') accounting for board offset."""
        col = ord(square[0]) - ord('a')
        row = 8 - int(square[1])
        return (BOARD_X + col * SQUARE_SIZE + SQUARE_SIZE / 2,
                BOARD_Y + row * SQUARE_SIZE + SQUARE_SIZE / 2)


    def draw_arrow(screen, start, end, color=ARROW_COLOR, width=3):
        """Draw an arrow from start to end. Start and end are (x,y) tuples."""
        pygame.draw.line(screen, color, start, end, width)
        angle = math.atan2(end[1] - start[1], end[0] - start[0])
        headlen = 10
        arrow_x1 = end[0] - headlen * math.cos(angle - math.pi / 6)
        arrow_y1 = end[1] - headlen * math.sin(angle - math.pi / 6)
        arrow_x2 = end[0] - headlen * math.cos(angle + math.pi / 6)
        arrow_y2 = end[1] - headlen * math.sin(angle + math.pi / 6)
        pygame.draw.line(screen, color, end, (arrow_x1, arrow_y1), width)
        pygame.draw.line(screen, color, end, (arrow_x2, arrow_y2), width)


    def display_endgame_box(message):
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


    def get_promotion_choice():
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
                    sys.exit()


    # ---- Thread Function for Engine Move ----
    def process_engine_move(user_move_str):
        global start_square, end_square, dragging, current_drag_pos, engine_thinking, engine_arrow
        if engine.apply_user_move(user_move_str):
            print("User move:", user_move_str)
            eng_move, _ = engine.get_engine_move()
            print("Engine move:", eng_move)
            if len(eng_move) >= 4:
                eng_from = eng_move[0:2]
                eng_to = eng_move[2:4]
                engine_arrow = (eng_from, eng_to)
        else:
            print("Invalid move:", user_move_str)
        start_square = None
        end_square = None
        dragging = False
        current_drag_pos = None
        engine_thinking = False


    # ---- Main Loop ----
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                # Check if click is in the board area (account for board offset) and engine is not thinking.
                if (BOARD_X <= x < BOARD_X + BOARD_SIZE and
                        BOARD_Y <= y < BOARD_Y + BOARD_SIZE and not engine_thinking):
                    start_square = pixel_to_square(x, y)
                    dragging = True
                    current_drag_pos = (x, y)
                # Else if click is in the left panel area, check buttons.
                elif x < BOARD_X:
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
                                if (piece.color and end_square[1] == '8') or (not piece.color and end_square[1] == '1'):
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
                    x, y = event.pos
                    # Only register the drop if it's within the board area.
                    if BOARD_X <= x < BOARD_X + BOARD_SIZE and BOARD_Y <= y < BOARD_Y + BOARD_SIZE:
                        end_square = pixel_to_square(x, y)
                    dragging = False

            elif event.type == pygame.MOUSEMOTION:
                if dragging:
                    current_drag_pos = event.pos

        # ---- Drawing Section ----
        screen.fill((0, 0, 0))
        draw_board(screen, engine.current_board)

        # Draw the user drag arrow if one exists.
        if start_square and current_drag_pos:
            start_center = square_center(start_square)
            draw_arrow(screen, start_center, current_drag_pos)

        # Draw engine's move arrow.
        if engine_arrow:
            from_center = square_center(engine_arrow[0])
            to_center = square_center(engine_arrow[1])
            draw_arrow(screen, from_center, to_center, color=ENGINE_ARROW_COLOR, width=4)

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
    sys.exit()
