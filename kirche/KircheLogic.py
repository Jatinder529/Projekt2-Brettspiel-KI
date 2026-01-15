
import numpy as np

class Board():
    """
    Board class for 'Lass die Kirche im Dorf' (Refactored).
    
    State Representation:
    (n, n, 2)
    - Channel 0: Owner (1, -1)
    - Channel 1: Type/Orientation
        0 = Vertical House (Moves N/S)
        1 = Horizontal House (Moves E/W)
        2 = Priest (Moves All Directions)
    """

    # Constants
    VERTICAL = 0
    HORIZONTAL = 1
    PRIEST = 2

    def __init__(self, n=6):
        self.n = n
        # tensor of shape (n, n, 2)
        self.state = np.zeros((self.n, self.n, 2), dtype=int)

    def __getitem__(self, index): 
        return self.state[index]

    def get_legal_moves(self, color):
        moves = []
        for x in range(self.n):
            for y in range(self.n):
                if self.state[x][y][0] == color:
                    piece_type = self.state[x][y][1]
                    moves.extend(self.get_moves_for_square(x, y, piece_type))
        return moves

    def get_moves_for_square(self, x, y, piece_type):
        moves = []
        
        # Directions based on type
        # CORRECTED VECTORS:
        # Vertical (0) -> Move Up/Down (Change Row/x)
        # Horizontal (1) -> Move Left/Right (Change Col/y)
        
        if piece_type == self.VERTICAL:
            directions = [(1, 0), (-1, 0)]
        elif piece_type == self.HORIZONTAL:
            directions = [(0, 1), (0, -1)]
        elif piece_type == self.PRIEST:
            # Priest moves in all 4 directions
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        else:
            directions = []

        for dx, dy in directions:
            cur_x, cur_y = x + dx, y + dy
            
            # CHANGED: 1-Step Logic (No sliding)
            # This prevents 1-turn wins across the empty board.
            if 0 <= cur_x < self.n and 0 <= cur_y < self.n:
                if self.state[cur_x][cur_y][0] == 0:
                    moves.append(((x, y), (cur_x, cur_y)))
                
        return moves

    def execute_move(self, move, color):
        (start_x, start_y), (end_x, end_y) = move

        piece_color = self.state[start_x][start_y][0]
        piece_type = self.state[start_x][start_y][1]
        
        # Clear old
        self.state[start_x][start_y][0] = 0
        self.state[start_x][start_y][1] = 0 

        # Place new
        self.state[end_x][end_y][0] = piece_color
        
        # Update orientation/type
        if piece_type == self.PRIEST:
            # Priest doesn't rotate
            self.state[end_x][end_y][1] = self.PRIEST
        else:
            # Houses rotate
            self.state[end_x][end_y][1] = 1 - piece_type

    def is_win(self, color):
        # Win condition: Reach opposite edge
        # P1 (Starts Row 0) -> Wins at Row N-1
        # P2 (Starts Row N-1) -> Wins at Row 0
        
        if color == 1:
            for y in range(self.n):
               if self.state[self.n-1][y][0] == 1: return True
        elif color == -1:
            for y in range(self.n):
               if self.state[0][y][0] == -1: return True
        return False
