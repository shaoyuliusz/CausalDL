import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Concatenate
from tensorflow.keras import regularizers
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
import numpy as np
"""
Define the representation layer class and hypothesis layer class for TARNet, CFRNet and DragonNet
"""
#code is based and modified from https://github.com/kochbj/Deep-Learning-for-Causal-Inference

class RepresentLayer(Layer):
    def __init__(self, num_units = 200, activation = 'elu', kernel_init = 'RandomNormal'):
        """
        Parameters:
            input_dim: input dimension
            num_units: number of neurons in each layer
            activation: dense layer activation function 
            kernel_initializer:
        """
        super().__init__()
        self.num_units = num_units
        self.dense_1 = Dense(units=num_units, activation=activation,
                             kernel_initializer=kernel_init,name='phi_1')
        self.dense_2 = Dense(units=num_units, activation=activation,
                             kernel_initializer=kernel_init,name='phi_2')
        self.dense_3 = Dense(units=num_units, activation=activation,
                             kernel_initializer=kernel_init,name='phi_3')
        
    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        x = self.dense_3(x)
        return x

class HypothesisLayer(Layer):
    def __init__(self, num_units = 100, activation = 'elu', kernel_reg = 'L2', reg_param = 0.01):
        """
        Parameters:
                input_dim: input dimension
                num_units: number of neurons in each layer
                activation: dense layer activation function
                kernel_reg: regularizer [L1,L2]
                reg_param: regularization penalty strength
        """
        super().__init__()
        if kernel_reg == 'L2':
            regularizer = regularizers.l2(reg_param)
        elif kernel_reg == 'L1':
            regularizer = regularizers.l1(reg_param)
        else:
            raise ValueError("Regularizer not correct, must be L1 or L2")
        
        self.regularizer = regularizer
        self.dense_y0_1 = Dense(units=num_units, activation= activation,
                                kernel_regularizer=regularizer,name='y0_hidden_1')
        self.dense_y0_2 = Dense(units=num_units, activation= activation, 
                                kernel_regularizer=regularizer,name='y0_hidden_2')
        self.dense_y1_1 = Dense(units=num_units, activation= activation,
                                kernel_regularizer=regularizer,name='y1_hidden_1')
        self.dense_y1_2 = Dense(units=num_units, activation= activation,
                                kernel_regularizer=regularizer,name='y1_hidden_2')
        self.pred_y0 = Dense(units=1, activation=None, 
                             kernel_regularizer=regularizers.l2(reg_param), 
                             name='y0_predictions')
        self.pred_y1 = Dense(units=1, activation=None, 
                             kernel_regularizer=regularizers.l2(reg_param), 
                             name='y1_predictions')
            
    def call(self, inputs):

        y0_hidden = self.dense_y0_1(inputs)
        y0_hidden = self.dense_y0_2(y0_hidden)
        y0_pred = self.pred_y0(y0_hidden)
        
        y1_hidden = self.dense_y1_1(inputs)
        y1_hidden = self.dense_y1_2(y1_hidden)
        y1_pred = self.pred_y1(y1_hidden)
        
        return y0_pred, y1_pred
    
class EpsilonLayer(Layer):
    """
    epsilon layer for Targeted Regularization
    this code follows Claudia Shi https://github.com/claudiashi57/dragonnet
    """
    def __init__(self):
        super(EpsilonLayer, self).__init__()

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.epsilon = self.add_weight(name='epsilon',
                                       shape=[1, 1],
                                       initializer='RandomNormal',
                                       #  initializer='ones',
                                       trainable=True)
        super(EpsilonLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs, **kwargs):
        #note there is only one epsilon were just duplicating it for conformability
        return self.epsilon * tf.ones_like(inputs)[:, 0:1]