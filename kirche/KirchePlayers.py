
import numpy as np

class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        while valids[a] != 1:
            a = np.random.randint(self.game.getActionSize())
        return a

class GreedyKirchePlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        valids = self.game.getValidMoves(board, 1)
        candidates = []
        
        # 1. Look for winning moves
        for a in range(self.game.getActionSize()):
            if valids[a] == 0:
                continue
                
            next_board, _ = self.game.getNextState(board, 1, a)
            # Check if this move results in a win for me (1)
            # getGameEnded returns 1 if p1 wins, -1 if p2 wins, 0 else
            # But getNextState flips perspective to opponent (-1).
            # So if getGameEnded returns 1 (relative to previous turn's player), it's a win?
            # Actually getGameEnded(board, 1) returns 1 if Player 1 won.
            
            # Note: getNextState returns board from perspective of next player (-1).
            # So we need to check if the game ended. 
            # If I just moved, did I win?
            # getGameEnded checks win for a player.
            
            # Simple heuristic:
            # Score = Advancement.
            # Convert action to src, dst
            n = self.game.n
            dst = a % (n * n)
            dst_r = dst // n 
            # As P1 (1), we want to maximize Row index (reach N-1)
            
            score = dst_r 
            
            # If win, score = infinity
            if self.game.getGameEnded(next_board, 1) == 1: # I won?
                 # Wait, getGameEnded takes (board, player). 
                 # If next_board is from P2 perspective...
                 # It's complicated due to canonical forms.
                 # Let's stick to simple "Furthest Forward" heuristic.
                 score += 1000
            
            candidates.append((score, a))
            
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
