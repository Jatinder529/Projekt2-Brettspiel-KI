
import argparse
import os
import shutil
import time
import random
import numpy as np
import math
import sys
sys.path.append('..')
from utils import *
from NeuralNet import NeuralNet

from .KircheNNet import KircheNNet as onnet

args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': False,
    'num_channels': 512,
})

class NNetWrapper(NeuralNet):
    def __init__(self, game):
        self.nnet = onnet(game, args)
        self.board_x, self.board_y, self.board_z = game.getBoardSize()
        self.action_size = game.getActionSize()

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        input_boards, target_pis, target_vs = list(zip(*examples))
        input_boards = np.asarray(input_boards)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)
        history = self.nnet.model.fit(x = input_boards, y = [target_pis, target_vs], batch_size = args.batch_size, epochs = args.epochs)

        # LOGGING LOSS
        # Create/Append to a csv file: iteration, loss, pi_loss, v_loss
        # Since we don't have explicit iteration count passed here easily, we'll just append lines.
        # Format: timestamp, loss, pi_loss, v_loss
        import csv
        
        checkpoint_dir = args.checkpoint if 'checkpoint' in args else '.'
        log_file = os.path.join(checkpoint_dir, 'loss_log.csv')
        
        # Check if we need header
        file_exists = os.path.isfile(log_file)
        
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['epoch', 'total_loss', 'pi_loss', 'v_loss'])
            
            # Write average loss for this 'train' call (which covers multiple epochs)
            # Or write each epoch? Let's write the final epoch's loss or average.
            # Usually users want to track loss over time.
            # history.history is a dict: {'loss': [e1, e2...], ...}
            
            for i in range(args.epochs):
                l = history.history['loss'][i]
                pi_l = history.history['pi_loss'][i] if 'pi_loss' in history.history else 0
                v_l = history.history['v_loss'][i] if 'v_loss' in history.history else 0
                writer.writerow([i, l, pi_l, v_l]) # 'i' is just epoch index within this batch

    def predict(self, board):
        """
        board: np array with board
        """
        # timing
        start = time.time()

        # preparing input
        # board is (N, N, 2)
        # Model expects (Batch, N, N, 2)
        board = board[np.newaxis, :, :, :]

        # run
        # Use __call__ with training=False for faster inference on small batches
        pi, v = self.nnet.model(board, training=False)

        return pi[0].numpy(), v[0].numpy()

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # change extension
        filename = filename.split(".")[0] + ".weights.h5"

        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        self.nnet.model.save_weights(filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # change extension
        filename = filename.split(".")[0] + ".weights.h5"

        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ValueError("No model in path '{}'".format(filepath))
        self.nnet.model.load_weights(filepath)
