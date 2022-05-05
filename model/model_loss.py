import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.metrics import binary_accuracy
from tensorflow.keras.losses import Loss
"""
Define custom loss functions for TarNet, CFRNet and DragonNet
"""

def MSE_Loss(concat_true, concat_pred):
    #standard MSE loss for TARNet
    
    #get individual vectors
    y_true = concat_true[:, 0]
    t_true = concat_true[:, 1]

    y0_pred = concat_pred[:, 0]
    y1_pred = concat_pred[:, 1]

    loss0 = tf.reduce_sum((1. - t_true) * tf.square(y_true - y0_pred))
    loss1 = tf.reduce_sum(t_true * tf.square(y_true - y1_pred))
    return loss0 + loss1

class Base_Dragon_Loss(Loss):
    #baseline dragonnet loss
    #initialize instance attributes
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        self.name='standard_loss'

    def split_pred(self,concat_pred):
        #generic helper to make sure we dont make mistakes
        preds={}
        preds['y0_pred'] = concat_pred[:, 0]
        preds['y1_pred'] = concat_pred[:, 1]
        preds['t_pred'] = concat_pred[:, 2]
        preds['phi'] = concat_pred[:, 3:]
        return preds

    #for logging purposes only
    def treatment_acc(self,concat_true,concat_pred):
        t_true = concat_true[:, 1]
        p = self.split_pred(concat_pred)
        #Since this isn't used as a loss, I've used tf.reduce_mean for interpretability
        return tf.reduce_mean(binary_accuracy(t_true, tf.math.sigmoid(p['t_pred']), threshold=0.5))

    def treatment_bce(self,concat_true,concat_pred):
        t_true = concat_true[:, 1]
        p = self.split_pred(concat_pred)
        lossP = tf.reduce_sum(binary_crossentropy(t_true,p['t_pred'],from_logits=True))
        return lossP
    
    def regression_loss(self,concat_true,concat_pred):
        #regular regression loss
        y_true = concat_true[:, 0]
        t_true = concat_true[:, 1]
        p = self.split_pred(concat_pred)
        loss0 = tf.reduce_sum((1. - t_true) * tf.square(y_true - p['y0_pred']))
        loss1 = tf.reduce_sum(t_true * tf.square(y_true - p['y1_pred']))
        return loss0+loss1

    def standard_loss(self,concat_true,concat_pred):
        lossR = self.regression_loss(concat_true,concat_pred)
        lossP = self.treatment_bce(concat_true,concat_pred)
        return lossR + self.alpha * lossP

    #compute loss
    def call(self, concat_true, concat_pred):        
        return self.standard_loss(concat_true,concat_pred)

#subclass of our base loss for targeted regularization DragonNet
class TarReg_Loss(Base_Dragon_Loss):
    #initialize instance attributes
    def __init__(self, alpha=1,beta=1):
        super().__init__()
        self.alpha = alpha
        self.beta=beta
        self.name='tarreg_loss'

    def split_pred(self,concat_pred):
        #generic helper to make sure we dont make mistakes
        preds={}
        preds['y0_pred'] = concat_pred[:, 0]
        preds['y1_pred'] = concat_pred[:, 1]
        preds['t_pred'] = concat_pred[:, 2]
        preds['epsilon'] = concat_pred[:, 3] #we're moving epsilon into slot three
        preds['phi'] = concat_pred[:, 4:]
        return preds

    def calc_hstar(self,concat_true,concat_pred):
        #step 2 above
        p=self.split_pred(concat_pred)
        y_true = concat_true[:, 0]
        t_true = concat_true[:, 1]

        t_pred = tf.math.sigmoid(concat_pred[:, 2])
        t_pred = (t_pred + 0.001) / 1.002 # a little numerical stability trick implemented by Shi
        y_pred = t_true * p['y1_pred'] + (1 - t_true) * p['y0_pred']

        #calling it cc for "clever covariate" as in SuperLearner TMLE literature
        cc = t_true / t_pred - (1 - t_true) / (1 - t_pred)
        h_star = y_pred + p['epsilon'] * cc
        return h_star

    def call(self,concat_true,concat_pred):
        y_true = concat_true[:, 0]

        standard_loss=self.standard_loss(concat_true,concat_pred)
        h_star=self.calc_hstar(concat_true,concat_pred)
        #step 3 above
        targeted_regularization = tf.reduce_sum(tf.square(y_true - h_star))

        # final
        loss = standard_loss + self.beta * targeted_regularization
        return loss

def pdist2sq(x,y):
    x2 = tf.reduce_sum(x ** 2, axis=-1, keepdims=True)
    y2 = tf.reduce_sum(y ** 2, axis=-1, keepdims=True)
    dist = x2 + tf.transpose(y2, (1, 0)) - 2. * x @ tf.transpose(y, (1, 0))
    return dist



class CFRNet_Loss(Loss):
      #initialize instance attributes
    def __init__(self, alpha=1.,sigma=1.):
        super().__init__()
        self.alpha = alpha # balances regression loss and MMD IPM
        self.rbf_sigma=sigma #for gaussian kernel
        self.name='cfrnet_loss'

    def split_pred(self,concat_pred):
        #generic helper to make sure we dont make mistakes
        preds={}
        preds['y0_pred'] = concat_pred[:, 0]
        preds['y1_pred'] = concat_pred[:, 1]
        preds['phi'] = concat_pred[:, 2:]
        return preds

    def rbf_kernel(self, x, y):
        return tf.exp(-pdist2sq(x,y)/tf.square(self.rbf_sigma))

    def calc_mmdsq(self, Phi, t):
        Phic, Phit =tf.dynamic_partition(Phi,tf.cast(tf.squeeze(t),tf.int32),2)

        Kcc = self.rbf_kernel(Phic,Phic)
        Kct = self.rbf_kernel(Phic,Phit)
        Ktt = self.rbf_kernel(Phit,Phit)

        m = tf.cast(tf.shape(Phic)[0],Phi.dtype)
        n = tf.cast(tf.shape(Phit)[0],Phi.dtype)

        mmd = 1.0/(m*(m-1.0))*(tf.reduce_sum(Kcc))
        mmd = mmd + 1.0/(n*(n-1.0))*(tf.reduce_sum(Ktt))
        mmd = mmd - 2.0/(m*n)*tf.reduce_sum(Kct)
        return mmd * tf.ones_like(t)

    def mmdsq_loss(self, concat_true,concat_pred):
        t_true = concat_true[:, 1]
        p=self.split_pred(concat_pred)
        mmdsq_loss = tf.reduce_mean(self.calc_mmdsq(p['phi'],t_true))
        return mmdsq_loss

    def regression_loss(self,concat_true,concat_pred):
        y_true = concat_true[:, 0]
        t_true = concat_true[:, 1]
        p = self.split_pred(concat_pred)
        loss0 = tf.reduce_mean((1. - t_true) * tf.square(y_true - p['y0_pred']))
        loss1 = tf.reduce_mean(t_true * tf.square(y_true - p['y1_pred']))
        return loss0+loss1

    def cfr_loss(self,concat_true,concat_pred):
        lossR = self.regression_loss(concat_true,concat_pred)
        lossIPM = self.mmdsq_loss(concat_true,concat_pred)
        return lossR + self.alpha * lossIPM

        #return lossR + self.alpha * lossIPM

    #compute loss
    def call(self, concat_true, concat_pred):        
        return self.cfr_loss(concat_true,concat_pred)