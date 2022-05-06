import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Concatenate
from tensorflow.keras import regularizers
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer

from model.common_layer import RepresentLayer, HypothesisLayer, EpsilonLayer
from utils import config

class TARNet(Model):
    """
    baseline TARNet
    https://arxiv.org/pdf/1606.03976.pdf
    """
    def __init__(self, name='tarnet', **kwargs):
        super().__init__(name=name, **kwargs)
        self.repre_layer = RepresentLayer(num_units = config.num_units_rep, activation = config.actv, kernel_init = config.kernel_init)
        self.hypo_layer = HypothesisLayer(num_units = config.num_units_hypo, activation = config.actv, kernel_reg = config.kernel_reg, reg_param = config.reg_param)

    def call(self, inputs):
        #x = Input(shape=(input_dim,), name='input')
        phi = self.repre_layer(inputs)
        y0_pred, y1_pred = self.hypo_layer(phi)
        concat_pred = Concatenate(1)([y0_pred, y1_pred])
        return concat_pred
    
class CFRNet(Model):
    """
    CFRNet (with an additional IPW loss)
    https://arxiv.org/pdf/1606.03976.pdf
    """
    def __init__(self, name='cfrnet', **kwargs):
        super().__init__(name=name, **kwargs)
        self.repre_layer = RepresentLayer(num_units = config.num_units_rep, activation = config.actv, kernel_init = config.kernel_init)
        self.hypo_layer = HypothesisLayer(num_units = config.num_units_hypo, activation = config.actv, kernel_reg = config.kernel_reg, reg_param = config.reg_param)

    def call(self, inputs):
        #x = Input(shape=(input_dim,), name='input')
        phi = self.repre_layer(inputs)
        y0_pred, y1_pred = self.hypo_layer(phi)
        concat_pred = Concatenate(1)([y0_pred, y1_pred, phi])
        return concat_pred
    
class DragonNetAIPW(Model):
    """
    baseline DragonNet with AIPW (augmented inverse propensity weighting estimator) 
    https://arxiv.org/pdf/1906.02120.pdf
    """
    def __init__(self, name='dragonnet', **kwargs):
        super().__init__(name=name, **kwargs)
        self.repre_layer = RepresentLayer(num_units = config.num_units_rep, activation = config.actv, kernel_init = config.kernel_init)
        self.hypo_layer = HypothesisLayer(num_units = config.num_units_hypo, activation = config.actv, kernel_reg = config.kernel_reg, reg_param = config.reg_param)
        self.prop_layer = Dense(units=1,activation=None, name='t_prediction')

    def call(self, inputs):
        phi = self.repre_layer(inputs)
        y0_pred, y1_pred = self.hypo_layer(phi)
        #propensity prediction
        #Note that the activation is actually sigmoid, but we will squish it in the loss function for numerical stability reasons
        t_pred = self.prop_layer(phi) 
        concat_pred = Concatenate(1)([y0_pred, y1_pred,t_pred,phi])
        return concat_pred
        
class DragonNetTR(Model):
    """
    DragonNet with Targeted Regularization
    https://arxiv.org/pdf/1906.02120.pdf
    """
    def __init__(self, name='dragonnet', **kwargs):
        super().__init__(name=name, **kwargs)
        self.repre_layer = RepresentLayer(num_units = config.num_units_rep, activation = config.actv, kernel_init = config.kernel_init)
        self.hypo_layer = HypothesisLayer(num_units = config.num_units_hypo, activation = config.actv, kernel_reg = config.kernel_reg, reg_param = config.reg_param)
        self.prop_layer = Dense(units=1,activation=None, name='t_prediction')
        self.epsilon_layer = EpsilonLayer()
    
    def call(self, inputs):
        phi = self.repre_layer(inputs)
        y0_pred, y1_pred = self.hypo_layer(phi)
        #propensity prediction
        #Note that the activation is actually sigmoid, but we will squish it in the loss function for numerical stability reasons
        t_prediction = self.prop_layer(phi) 
        epsilons = self.epsilon_layer(t_prediction)
        concat_pred = Concatenate(1)([y0_pred, y1_pred,epsilons,phi])
        return concat_pred