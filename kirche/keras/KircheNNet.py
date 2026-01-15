
import sys
import os
sys.path.append('..')
from utils import *

import argparse
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *

class KircheNNet():
    """
    Neural Network for Kirche Game.
    Adapts the Othello/TicTacToe/Connect4 architecture.
    """
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y, self.board_z = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        # Neural Net
        self.input_boards = Input(shape=(self.board_x, self.board_y, self.board_z))

        # Block 1 - Conv > BN > Relu
        # Note: channel_axis=3 because input is (x, y, channels)
        x_image = Reshape((self.board_x, self.board_y, self.board_z))(self.input_boards)
        
        h_conv1 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='same', use_bias=False)(x_image))) 
        h_conv2 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='same', use_bias=False)(h_conv1))) 
        h_conv3 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='same', use_bias=False)(h_conv2))) 
        h_conv4 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='valid', use_bias=False)(h_conv3))) 
        
        h_conv4_flat = Flatten()(h_conv4)       
        
        s_fc1 = Activation('relu')(BatchNormalization(axis=1)(Dense(1024, use_bias=False)(h_conv4_flat))) 
        s_fc2 = Activation('relu')(BatchNormalization(axis=1)(Dense(512, use_bias=False)(s_fc1))) 
        
        self.pi = Dense(self.action_size, activation='softmax', name='pi')(s_fc2)   # policy
        self.v = Dense(1, activation='tanh', name='v')(s_fc2)                    # value

        self.model = Model(inputs=self.input_boards, outputs=[self.pi, self.v])
        self.model.compile(loss=['categorical_crossentropy','mean_squared_error'], optimizer=Adam(args.lr))
