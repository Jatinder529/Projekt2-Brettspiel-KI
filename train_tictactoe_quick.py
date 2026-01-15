
import logging
import coloredlogs
from Coach import Coach
from tictactoe.TicTacToeGame import TicTacToeGame as Game
from tictactoe.keras.NNet import NNetWrapper as nn
from utils import *

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')

# Quick training configuration for the assignment
args = dotdict({
    'numIters': 50,              # 50 iterations for quick test
    'numEps': 25,               # 25 episodes per iteration
    'tempThreshold': 15,
    'updateThreshold': 0.6,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 25,
    'arenaCompare': 20,         # Compare fewer games
    'cpuct': 1,

    'checkpoint': './temp/tictactoe/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
})

def main():
    log.info('Loading %s...', Game.__name__)
    g = Game()

    log.info('Loading %s...', nn.__name__)
    nnet = nn(g)

    log.info('Loading the Coach...')
    c = Coach(g, nnet, args)

    log.info('Starting the learning process 🎉')
    c.learn()

if __name__ == "__main__":
    main()
