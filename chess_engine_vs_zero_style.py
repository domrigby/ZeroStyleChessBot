import sys
import math
import threading
import time

import pygame
import chess
import chess.engine
from zero_style_chess_engine.engine_interface import EngineInterface

if __name__=="__main__":

    import multiprocessing as mp

    try:
        mp.set_start_method('spawn', force=True)
        print("spawned")
    except RuntimeError:
        print("Failed to spawn")
        pass

    # ---- Configuration ----
    BOARD_SIZE   = 400
    SQUARE_SIZE  = BOARD_SIZE // 8
    MARGIN       = 20
    WINDOW_W     = BOARD_SIZE + 2 * MARGIN
    WINDOW_H     = BOARD_SIZE + 2 * MARGIN
    FPS          = 30

    # Colors
    LIGHT_SQ       = (240, 217, 181)
    DARK_SQ        = (181, 136,  99)
    NET_COLOR      = (255,   0,   0)   # Your net moves
    STOCKFISH_COLOR= (  0,   0, 255)   # Stockfish moves
    TEXT_COLOR     = (  0,   0,   0)

    # ---- Engine Setup ----
    # Your network-based engine (we'll play White)
    net_engine = EngineInterface(
        network_path   =r"neural_nets/example_network.pt",
        num_rollouts   = 2000
    )

    # Stockfish (Black).  Adjust to your binary path:
    SF_PATH = "/home/dom/stockfish/stockfish-ubuntu-x86-64-avx2"
    sf_engine = chess.engine.SimpleEngine.popen_uci(SF_PATH)
    # Optionally cap strength:
    sf_engine.configure({
        "UCI_LimitStrength": True,
        "UCI_Elo":          1320,
    })
    SF_TIME = 10.0  # seconds per Stockfish move

    # Central board for display
    board = chess.Board()

    # List of (from_xy, to_xy, color) for drawing arrows
    from collections import deque
    arrows = []

    # Thread-safe thinking flag
    thinking = False

    # Utility: pixel center of a square index
    def square_center(sq):
        file = chess.square_file(sq)
        rank = 7 - chess.square_rank(sq)
        x = MARGIN + file * SQUARE_SIZE + SQUARE_SIZE/2
        y = MARGIN + rank * SQUARE_SIZE + SQUARE_SIZE/2
        return (x, y)

    # Background game loop
    def play_game():
        global thinking, arrows
        while not board.is_game_over():
            thinking = True

            if board.turn:
                # White: your network
                # get_engine_move() should both return AND apply internally
                uci_move, _ = net_engine.get_engine_move()
                mv = chess.Move.from_uci(uci_move)
                board.push(mv)
                if len(arrows) == 2:
                    arrows = []
                arrows.append((square_center(mv.from_square),
                               square_center(mv.to_square),
                               NET_COLOR))
            else:
                # Black: Stockfish
                res = sf_engine.play(board, chess.engine.Limit(time=SF_TIME))
                mv  = res.move
                board.push(mv)
                # Sync your net's internal board state:
                net_engine.apply_user_move(mv.uci())
                if len(arrows) == 2:
                    arrows = []
                arrows.append((square_center(mv.from_square),
                               square_center(mv.to_square),
                               STOCKFISH_COLOR))

            thinking = False
            time.sleep(0.1)

        print("Game over:", board.result())

    # ---- Pygame Setup ----
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("Network vs Stockfish")
    piece_font  = pygame.font.SysFont("DejaVu Sans", 36, bold=True)
    status_font = pygame.font.SysFont("Arial", 24)
    clock = pygame.time.Clock()

    # Start the game thread
    threading.Thread(target=play_game, daemon=True).start()

    # ---- Main Loop ----
    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False

        # Draw board
        screen.fill((0,0,0))
        for r in range(8):
            for f in range(8):
                col = LIGHT_SQ if (r+f)%2==0 else DARK_SQ
                rect = pygame.Rect(
                    MARGIN + f*SQUARE_SIZE,
                    MARGIN + r*SQUARE_SIZE,
                    SQUARE_SIZE, SQUARE_SIZE
                )
                pygame.draw.rect(screen, col, rect)

        # Draw pieces
        for sq in chess.SQUARES:
            p = board.piece_at(sq)
            if p:
                x,y = square_center(sq)
                sym = p.unicode_symbol()
                surf= piece_font.render(sym, True, TEXT_COLOR)
                rect= surf.get_rect(center=(x,y))
                screen.blit(surf, rect)

        # Draw move arrows
        for (a,b,c) in arrows:
            pygame.draw.line(screen, c, a, b, 4)
            ang = math.atan2(b[1]-a[1], b[0]-a[0])
            h   = 10
            p1  = (b[0] - h*math.cos(ang-math.pi/6),
                   b[1] - h*math.sin(ang-math.pi/6))
            p2  = (b[0] - h*math.cos(ang+math.pi/6),
                   b[1] - h*math.sin(ang+math.pi/6))
            pygame.draw.line(screen, c, b, p1, 4)
            pygame.draw.line(screen, c, b, p2, 4)

        # Status line
        status = "Thinking..." if thinking else f"Result: {board.result()}" if board.is_game_over() else ""
        txt    = status_font.render(status, True, TEXT_COLOR)
        screen.blit(txt, (MARGIN, WINDOW_H - MARGIN - 24))

        pygame.display.flip()
        clock.tick(FPS)

    # Cleanup
    sf_engine.quit()
    pygame.quit()
    sys.exit()