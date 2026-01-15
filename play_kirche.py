import pygame
import numpy as np
import sys
import time
from utils import *
from MCTS import MCTS
from kirche.KircheGame import KircheGame
from kirche.keras.NNet import NNetWrapper as NNet

# --- Constants ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800
BG_COLOR = (240, 240, 230)
GRID_COLOR = (50, 50, 50)
P1_COLOR = (50, 100, 200) # Blue
P2_COLOR = (200, 50, 50)  # Red
HIGHLIGHT_COLOR = (100, 255, 100)
VALID_MOVE_COLOR = (200, 255, 200)

class DummyNNet():
    """Fallback if no model is found."""
    def __init__(self, game):
        self.action_size = game.getActionSize()
    def predict(self, board):
        return np.ones(self.action_size)/self.action_size, 0

# --- Configuration ---
# --- Configuration ---
class VariantConfig:
    def __init__(self, name, n, priests, checkpoint_dir):
        self.name = name
        self.n = n
        self.priests = priests
        self.checkpoint_dir = checkpoint_dir

class DifficultyConfig:
    def __init__(self, name, sims):
        self.name = name
        self.sims = sims

VARIANTS = [
    VariantConfig("Variant 1 (5x5) - Optimized", 5, 1, './temp/variant1/'),
    VariantConfig("Repository AI (6x6, 2 Priests)", 6, 2, './temp/variant2/'),
    VariantConfig("Base AI (Untrained)", 6, 1, None),
]

DIFFICULTIES = [
    DifficultyConfig("Easy (10 Sims)", 10),
    DifficultyConfig("Medium (50 Sims)", 50),
    DifficultyConfig("Hard (400 Sims)", 400),
]

def draw_text(screen, text, size, x, y, color=(0, 0, 0)):
    font = pygame.font.SysFont("Arial", size)
    img = font.render(text, True, color)
    rect = img.get_rect(center=(x, y))
    screen.blit(img, rect)

def draw_menu_buttons(screen, title, options, start_y, spacing):
    """Generic helper to draw a menu with buttons."""
    screen.fill(BG_COLOR)
    draw_text(screen, "Lass die Kirche im Dorf", 60, SCREEN_WIDTH // 2, 100)
    draw_text(screen, title, 40, SCREEN_WIDTH // 2, 200)
    
    buttons = []
    for i, opt in enumerate(options):
        btn = pygame.Rect(0, 0, 400, 60)
        btn.center = (SCREEN_WIDTH // 2, start_y + i * spacing)
        
        # Simple color cycle
        color = (180, 200, 220) if i % 2 == 0 else (200, 220, 180)
        buttons.append((btn, opt, color))
        
        # Draw
        pygame.draw.rect(screen, color, btn, border_radius=10)
        pygame.draw.rect(screen, (0,0,0), btn, 2, border_radius=10)
        draw_text(screen, opt.name, 30, btn.centerx, btn.centery)
        
    pygame.display.flip()
    return buttons

def handle_menu_click(buttons, pos):
    for rect, opt, _ in buttons:
        if rect.collidepoint(pos):
            return opt
    return None

def select_variant_screen(screen):
    buttons = draw_menu_buttons(screen, "Select Game Variant:", VARIANTS, 300, 80)
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                choice = handle_menu_click(buttons, event.pos)
                if choice: return choice

def select_difficulty_screen(screen):
    buttons = draw_menu_buttons(screen, "Select Difficulty:", DIFFICULTIES, 300, 80)
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                choice = handle_menu_click(buttons, event.pos)
                if choice: return choice

def draw_board(screen, game_board, n, cell_size, selected_piece=None, valid_moves=None):
    screen.fill(BG_COLOR)

    # Draw grid
    for i in range(n):
        for j in range(n):
            rect = pygame.Rect(j * cell_size, i * cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, GRID_COLOR, rect, 1)

    # Highlight valid moves
    if valid_moves:
        for (r, c) in valid_moves:
             rect = pygame.Rect(c * cell_size + 2, r * cell_size + 2, cell_size - 4, cell_size - 4)
             pygame.draw.rect(screen, VALID_MOVE_COLOR, rect, 0)

    # Highlight selected piece
    if selected_piece:
        r, c = selected_piece
        rect = pygame.Rect(c * cell_size, r * cell_size, cell_size, cell_size)
        pygame.draw.rect(screen, HIGHLIGHT_COLOR, rect, 3)

    # Draw pieces
    for r in range(n):
        for c in range(n):
            player = game_board[r][c][0]
            ptype = game_board[r][c][1] # 0=Vert, 1=Horiz, 2=Priest

            if player != 0:
                color = P1_COLOR if player == 1 else P2_COLOR
                cx = c * cell_size + cell_size // 2
                cy = r * cell_size + cell_size // 2
                
                if ptype == 2: # Priest (Circle or Star)
                    pygame.draw.circle(screen, color, (cx, cy), cell_size // 3)
                    pygame.draw.circle(screen, (0,0,0), (cx, cy), cell_size // 3, 2)
                    # Label 'P'
                    draw_text(screen, "P", 20, cx, cy, (255,255,255))
                else:
                    if ptype == 0: # Vertical
                        w, h = cell_size * 0.4, cell_size * 0.8
                    else: # Horizontal
                        w, h = cell_size * 0.8, cell_size * 0.4
                    
                    piece_rect = pygame.Rect(0, 0, w, h)
                    piece_rect.center = (cx, cy)
                    pygame.draw.rect(screen, color, piece_rect, 0, border_radius=5)
                    pygame.draw.rect(screen, (0,0,0), piece_rect, 2, border_radius=5)

    pygame.display.flip()

def get_square_under_mouse(pos, n, cell_size):
    x, y = pos
    col = x // cell_size
    row = y // cell_size
    if 0 <= col < n and 0 <= row < n:
        return int(row), int(col)
    return None

def show_game_over_screen(screen, winner):
    """
    Displays the Game Over screen.
    """
    # Create semi-transparent overlay
    overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
    overlay.set_alpha(180)
    overlay.fill((255, 255, 255))
    screen.blit(overlay, (0, 0))

    msg = "Player 1 Wins!" if winner == 1 else "AI Wins!"
    color = P1_COLOR if winner == 1 else P2_COLOR
    
    draw_text(screen, "GAME OVER", 80, SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 50, (0, 0, 0))
    draw_text(screen, msg, 60, SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 50, color)
    draw_text(screen, "Click to Exit", 30, SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 150, (100, 100, 100))
    
    pygame.display.flip()
    
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                waiting = False


# --- Game Modes ---
class GameMode:
    PVE = "Player vs AI"
    PVP = "Player vs Player"

def select_mode_screen(screen):
    options = [
        dotdict({'name': GameMode.PVE}),
        dotdict({'name': GameMode.PVP})
    ]
    buttons = draw_menu_buttons(screen, "Select Game Mode:", options, 300, 80)
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                choice = handle_menu_click(buttons, event.pos)
                if choice: return choice.name

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Lass die Kirche im Dorf")

    # 1. Select Variant
    variant_cfg = select_variant_screen(screen)
    print(f"Selected Variant: {variant_cfg.name}")
    
    # 2. Select Mode
    mode = select_mode_screen(screen)
    print(f"Selected Mode: {mode}")

    # 3. Select Difficulty (Only if PVE)
    diff_cfg = None
    if mode == GameMode.PVE:
        diff_cfg = select_difficulty_screen(screen)
        print(f"Selected Difficulty: {diff_cfg.name}")
        pygame.display.set_caption(f"Lass die Kirche im Dorf | {variant_cfg.name} | {diff_cfg.name}")
    else:
        pygame.display.set_caption(f"Lass die Kirche im Dorf | {variant_cfg.name} | PvP")

    # 4. Setup Game
    n = variant_cfg.n
    cell_size = SCREEN_WIDTH // n
    game = KircheGame(n=n, num_priests=variant_cfg.priests)
    board = game.getInitBoard()
    
    # 5. Load Model / Setup AI (Only if PVE)
    mcts = None
    if mode == GameMode.PVE:
        if variant_cfg.checkpoint_dir:
            try:
                print(f"Loading model from {variant_cfg.checkpoint_dir}...")
                nnet = NNet(game)
                try:
                    nnet.load_checkpoint(variant_cfg.checkpoint_dir, 'best.pth.tar')
                except:
                    print("Best model not found, trying temp...")
                    nnet.load_checkpoint(variant_cfg.checkpoint_dir, 'temp.pth.tar')
            except Exception as e:
                print(f"Model load failed: {e}. Using Dummy AI.")
                nnet = DummyNNet(game)
        else:
            print("No checkpoint configured. Using Dummy AI.")
            nnet = DummyNNet(game)

        args = dotdict({'numMCTSSims': diff_cfg.sims, 'cpuct': 1.0})
        mcts = MCTS(game, nnet, args)

    player = 1
    selected_piece = None
    valid_destinations = []
    legal_moves_map = {} # (r,c) -> action_idx

    running = True
    while running:
        draw_board(screen, board, n, cell_size, selected_piece, valid_destinations)
        
        # Check Win
        if game.getGameEnded(board, player) != 0:
            winner = game.getGameEnded(board, player)
            # In PvP: 1 is P1 (Blue), -1 is P2 (Red)
            # logic remains valid since getGameEnded returns winning player ID
            print(f"Game Over. Winner: {winner}")
            show_game_over_screen(screen, winner)
            sys.exit()

        # AI Turn (Only in PVE and when player is -1)
        if mode == GameMode.PVE and player == -1: 
            print("AI thinking...")
            pygame.event.pump()
            # Run MCTS
            root_board = game.getCanonicalForm(board, player)
            pi = mcts.getActionProb(root_board, temp=0)
            action = np.argmax(pi)
            
            # Transform action back to real coordinates
            if player == -1:
                # Decode
                n = game.n
                src_idx = action // (n * n)
                dst_idx = action % (n * n)
                src_r, src_c = src_idx // n, src_idx % n
                dst_r, dst_c = dst_idx // n, dst_idx % n
                
                # Rotate 180 (Invert)
                real_src_r, real_src_c = n - 1 - src_r, n - 1 - src_c
                real_dst_r, real_dst_c = n - 1 - dst_r, n - 1 - dst_c
                
                # Encode
                real_src_idx = real_src_r * n + real_src_c
                real_dst_idx = real_dst_r * n + real_dst_c
                action = real_src_idx * (n * n) + real_dst_idx

            board, player = game.getNextState(board, player, action)
            print("AI moved.")
            selected_piece = None
            valid_destinations = []
            continue
            
        # Human Input (P1 or P2 in PvP)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
                
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                sq = get_square_under_mouse(event.pos, n, cell_size)
                if not sq: continue
                r, c = sq
                
                # A: Select own piece
                # In PvP: Player 1 touches 1, Player -1 touches -1.
                # Board logic: board[r][c][0] stores the owner (1 or -1).
                # So `board[r][c][0] == player` works for both P1 and P2!
                if board[r][c][0] == player:
                    selected_piece = (r, c)
                    # Calcs moves
                    # getValidMoves expects board from current player's perspective
                    canon_board = game.getCanonicalForm(board, player)
                    valids = game.getValidMoves(canon_board, 1) 
                    
                    valid_destinations = []
                    legal_moves_map = {}
                    
                    start_idx = r * n + c
                    
                    # If player is -1 (P2), their (r,c) is rotated in canonical form logic
                    # But getValidMoves returns actions in canonical (rotated) space for P2.
                    # Wait. Implementation detail:
                    # In getValidMoves:
                    #   moves = get_moves_for_square...
                    # It returns raw moves on the canonical board.
                    
                    # Let's map it carefully.
                    # If P2 (-1) clicks at real (r, c).
                    # Canonical board rotates this by 180.
                    # So we are looking for moves starting at (N-1-r, N-1-c) in canonical board.
                    
                    # Instead of complex mapping, let's rely on the fact that:
                    # 1. We have the canonical board.
                    # 2. We iterate through ALL valid moves from the canonical board.
                    # 3. We map each valid move BACK to real coordinates.
                    # 4. If the real start coord matches our click, show the real dest coord.
                    
                    for idx, v in enumerate(valids):
                        if v:
                            # Canonical Action -> Real Action
                            # Canonical Src/Dst
                            can_src = idx // (n*n)
                            can_dst = idx % (n*n)
                            can_src_r, can_src_c = can_src // n, can_src % n
                            can_dst_r, can_dst_c = can_dst // n, can_dst % n
                            
                            if player == 1:
                                real_src_r, real_src_c = can_src_r, can_src_c
                                real_dst_r, real_dst_c = can_dst_r, can_dst_c
                            else: # Player -1
                                real_src_r, real_src_c = n - 1 - can_src_r, n - 1 - can_src_c
                                real_dst_r, real_dst_c = n - 1 - can_dst_r, n - 1 - can_dst_c
                                
                            # Check if this move starts at my selected piece
                            if (real_src_r, real_src_c) == (r, c):
                                valid_destinations.append((real_dst_r, real_dst_c))
                                # We store the RAW CANONICAL action in the map?
                                # No, getNextState usually expects standard action for P1.
                                # But KircheGame.getNextState might call internal moves.
                                
                                # Actually, AlphaZero logic relies on getCanonicalForm -> MCTS -> Action.
                                # The Action is an index into the flattened policy vector (n*n*n*n).
                                
                                # If we pass the action index (from valids) to getNextState?
                                # getNextState:
                                #   if player == -1: move = rotate(move)
                                #   board.execute_move(move)
                                
                                # So getNextState expects the action to be relative to the player's perspective (canonical)??
                                # Let's check getNextState in KircheGame.py.
                                # "action takes int value of (x1,y1) -> (x2,y2)"
                                # "action is size n*n*n*n"
                                
                                # If getNextState expects the action index corresponding to the canonical board...
                                # Then `idx` is correct.
                                legal_moves_map[(real_dst_r, real_dst_c)] = idx
                                

                # B: Move to valid dest
                elif selected_piece and (r, c) in valid_destinations:
                    action = legal_moves_map[(r, c)]
                    board, player = game.getNextState(board, player, action)
                    selected_piece = None
                    valid_destinations = []
                
                else:
                    selected_piece = None
                    valid_destinations = []

if __name__ == "__main__":
    main()
