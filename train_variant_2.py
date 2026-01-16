from kirche.keras.NNet import args as nnet_args
import logging
import coloredlogs
from Coach import Coach
from kirche.KircheGame import KircheGame as Game
from kirche.keras.NNet import NNetWrapper as nn
from utils import *

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')

args = dotdict({
    'numIters': 25,            # 50 Iterations for long-term learning
    'numEps': 25,               # More games per iteration
    'tempThreshold': 15,
    'updateThreshold': 0.6,
    'maxlenOfQueue': 200000,
    
    # PARAMETER TUNING:
    # Increased simulations because this variant (2 Priests) has a larger branching factor 
    # and deeper strategy than the simplified version.
    'numMCTSSims': 80,          
    
    'arenaCompare': 20,
    'cpuct': 1,

    # CHECKPOINT:
    # Saving to a distinct folder for Variant 2
    'checkpoint': './temp/variant2/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

})

# Standard Epochs
nnet_args['epochs'] = 10
nnet_args['num_channels'] = 64 # Reduce from 512 for performance

def main():
    log.info('Loading %s...', Game.__name__)
    
    # VARIANT 2: Standard 6x6 Board, but with 2 Priests
    g = Game(n=6, num_priests=2)

    log.info('Loading %s...', nn.__name__)
    # Set checkpoint in NNet args for internal use (loss logging)
    nnet_args['checkpoint'] = args.checkpoint
    nnet = nn(g)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(g, nnet, args)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉 (Variant 2)')
    c.learn()

if __name__ == "__main__":
    main()
