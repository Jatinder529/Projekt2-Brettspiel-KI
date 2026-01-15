from __future__ import print_function
import sys
sys.path.append('..')
from Game import Game
from .KircheLogic import Board
import numpy as np

class KircheGame(Game):
    """
    Game class for 'Lass die Kirche im Dorf' (Refactored).
    Supports variable board size and number of priests.
    """
    def __init__(self, n=6, num_priests=1):
        self.n = n
        self.num_priests = num_priests

    def getInitBoard(self):
        b = Board(self.n)
        
        # 1. Place Standard Houses (Corners/Edges)
        # P1 (1) - Bottom/Left? Let's say (0,0)
        b.state[0][0][0] = 1
        b.state[0][0][1] = Board.VERTICAL
        
        # P2 (-1) - Top/Right (n-1, n-1)
        b.state[self.n-1][self.n-1][0] = -1
        b.state[self.n-1][self.n-1][1] = Board.VERTICAL
        
        # 2. Place Priests (if any)
        # P1 Priests (Near 0,0)
        for i in range(self.num_priests):
            if 1+i < self.n:
                 b.state[0][1+i][0] = 1
                 b.state[0][1+i][1] = Board.PRIEST
        
        # P2 Priests (Near N-1, N-1)
        for i in range(self.num_priests):
            if self.n-2-i >= 0:
                 b.state[self.n-1][self.n-2-i][0] = -1
                 b.state[self.n-1][self.n-2-i][1] = Board.PRIEST
                 
        return b.state

    def getBoardSize(self):
        return (self.n, self.n, 2)

    def getActionSize(self):
        return (self.n * self.n) ** 2

    def getNextState(self, board, player, action):
        n = self.n
        src = action // (n * n)
        dst = action % (n * n)
        
        start_x, start_y = src // n, src % n
        end_x, end_y = dst // n, dst % n
        
        move = ((start_x, start_y), (end_x, end_y))
        
        b = Board(self.n)
        b.state = np.copy(board)
        b.execute_move(move, player)
        
        return (b.state, -player)

    def getValidMoves(self, board, player):
        b = Board(self.n)
        b.state = np.copy(board)
        legalMoves = b.get_legal_moves(player)
        
        valids = [0] * self.getActionSize()
        
        if len(legalMoves) == 0:
            return np.array(valids)
            
        for (start, end) in legalMoves:
            start_idx = start[0] * self.n + start[1]
            end_idx = end[0] * self.n + end[1]
            action_idx = start_idx * (self.n * self.n) + end_idx
            valids[action_idx] = 1
            
        return np.array(valids)

    def getGameEnded(self, board, player):
        b = Board(self.n)
        b.state = np.copy(board)

        if b.is_win(player):
            return 1
        if b.is_win(-player):
            return -1
            
        if len(b.get_legal_moves(player)) == 0:
            return -1 
            
        return 0

    def getCanonicalForm(self, board, player):
        if player == 1:
            return board
        
        # If player -1:
        # 1. Flip ownership (1 <-> -1)
        # 2. Rotate board 180 degrees so perspective is same as P1
        
        ret = np.copy(board)
        # Flip colors
        ret[:,:,0] = ret[:,:,0] * -1
        
        # Rotate 180 (k=2)
        # We rotate the spatial dimensions (0, 1), not the channel dimension (2)
        ret = np.rot90(ret, 2, axes=(0, 1))
        
        return ret

    def getSymmetries(self, board, pi):
        # Since we use 180 rotation for canonical form, 
        # let's include rotational symmetries if the game supports it.
        # But for 'Lass die Kirche im Dorf', the board might not be fully determining symmetric moves
        # due to orientation logic. 
        # Safest is Identity.
        return [(board, pi)]

    def stringRepresentation(self, board):
        return board.tostring()
