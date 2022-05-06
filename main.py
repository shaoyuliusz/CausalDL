import os
import math
import time
import datetime
import numpy as np 
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Concatenate
from tensorflow.keras import regularizers
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN
from tensorflow.keras.optimizers import SGD, Adam

from utils import config
from utils.data_reader import read_IHDP_data, read_SIPP_data

from model.model_metrics import AIPW_Metrics, TarReg_Metrics, Base_Metrics
from model.model_loss import Base_Dragon_Loss, TarReg_Loss, MSE_Loss, CFRNet_Loss
from model.models import TARNet, CFRNet, DragonNetAIPW, DragonNetTR
from model.common_layer import RepresentLayer, HypothesisLayer, EpsilonLayer


i = 123
np.random.seed(i)
    
if config.dataset == "SIPP":
    data=read_SIPP_data(file = './data/SIPP/sipp1991.dta')
elif config.dataset == "IHDP":
    data=read_IHDP_data(training_data='./data/IHDP/ihdp_npci_1-100.train.npz',testing_data='./data/IHDP/ihdp_npci_1-100.test.npz')
    
print("DATASET USED",config.dataset)    

if config.model == "tarnet":
    model = TARNet()
elif config.model == "cfrnet":
    model = CFRNet()
elif config.model == "dragonnet":
    model = DragonNetAIPW()
elif config.model == "dragonnetTR":
    model = DragonNetTR()
    
print("MODEL USED",config.model)

yt = np.concatenate([data['ys'], data['t']], 1) #we'll use both y and t to compute the loss

basic_callback = [
        TerminateOnNaN(),
        EarlyStopping(monitor='val_loss', patience=40, min_delta=0.), 
        #40 is Shi's recommendation for this dataset, but you should tune for your data 
        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=config.verbose, mode='auto',
                          min_delta=0., cooldown=0, min_lr=0),
    ]

if config.model == "tarnet":
    model.compile(optimizer=SGD(learning_rate=config.lr, momentum=config.momentum, nesterov=config.nesterov),
                    loss=MSE_Loss,
                    metrics=MSE_Loss)
    
    start_time = time.time()
    model.fit(x=data['x'],y=yt,
                    callbacks=basic_callback,
                    validation_split=config.val_split,
                    epochs=config.num_epoch,
                    batch_size=config.batch_size,
                    verbose=config.verbose)
    elapsed_time = time.time() - start_time
    print("*********training_time is**********: ", elapsed_time)

elif config.model == "dragonnetTR":
    basic_callback.append(TarReg_Metrics(data, verbose=config.verbose)) 
    tarreg_loss=TarReg_Loss(alpha=1)

    model.compile(optimizer=SGD(learning_rate=config.lr, momentum=config.momentum, nesterov=config.nesterov),
                          loss=tarreg_loss,metrics=[tarreg_loss,tarreg_loss.regression_loss,tarreg_loss.treatment_acc])
    start_time = time.time()
    model.fit(x=data['x'],y=yt,
                     callbacks=basic_callback,
                      validation_split=config.val_split,
                      epochs=config.num_epoch,
                      batch_size=config.batch_size,
                      verbose=config.verbose)
    elapsed_time = time.time() - start_time
    print("*********training_time is**********: ", elapsed_time)
elif config.model == "dragonnet":
    basic_callback.append(AIPW_Metrics(data,verbose=config.verbose))
    aipw_loss=Base_Dragon_Loss(alpha=1.0)
    model.compile(optimizer=SGD(learning_rate = config.lr, momentum=config.momentum, nesterov=config.nesterov),
                        loss=aipw_loss,
                        metrics=[aipw_loss,aipw_loss.regression_loss,aipw_loss.treatment_acc]
                       )
    start_time = time.time()
    model.fit(x=data['x'],y=yt,
                      callbacks=basic_callback,
                      validation_split=config.val_split,
                      epochs=config.num_epoch,
                      batch_size=config.batch_size,
                      verbose=config.verbose)
    elapsed_time = time.time() - start_time
    print("*********training_time is**********: ", elapsed_time)

elif config.model == "cfrnet":
    adam_callbacks = [
        TerminateOnNaN(),
        EarlyStopping(monitor='val_loss', patience=2, min_delta=0.),
        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=config.verbose, mode='auto',
                          min_delta=1e-8, cooldown=0, min_lr=0),
        Base_Metrics(data,verbose=config.verbose)
    ]
    
    cfrnet_loss=CFRNet_Loss(alpha=1.0)

    model.compile(optimizer=Adam(learning_rate=config.lr),
                          loss=cfrnet_loss,
                     metrics=[cfrnet_loss,cfrnet_loss.regression_loss,cfrnet_loss.mmdsq_loss])
    
    start_time = time.time()
    model.fit(x=data['x'],y=yt,
                     callbacks=adam_callbacks,
                      validation_split=config.val_split,
                      epochs=config.num_epoch,
                      batch_size=config.batch_size,
                      verbose=config.verbose)
    
    elapsed_time = time.time() - start_time
    print("*********training_time is**********: ", elapsed_time)

# model.save('./save/trained_models/{}'.format(config.model+'_'+config.dataset))
# print('model saved!')
          
def compute_prediction(data, model):
    """
    Compute model predicted ATE(average treatment effect) and compare to ground truth (if exists)
    """
    concat_pred=model.predict(data['x'])
    #dont forget to rescale the outcome before estimation!
    y0_pred = data['y_scaler'].inverse_transform(concat_pred[:, 0].reshape(-1, 1))
    y1_pred = data['y_scaler'].inverse_transform(concat_pred[:, 1].reshape(-1, 1))
    
    #predicted CATE
    cate_pred=y1_pred-y0_pred
    
    data_save = {}      
    data_save['cate_pred'] = cate_pred
    
    if 'mu_1' in data.keys() or 'mu_0' in data.keys():
        cate_true=data['mu_1']-data['mu_0'] #Hill's noiseless true values
        data_save['cate_true'] = cate_true
        print("Actual ATE:", cate_true.mean(),'\n\n')
    ate_pred=tf.reduce_mean(cate_pred)
    print("Estimated ATE:", ate_pred.numpy(),'\n\n')
    data_save['ate_pred'] = ate_pred
    
    return data_save

data_save = compute_prediction(data, model)

np.save('./save/pred_result/pred_result_{}.npy'.format(config.model + '_' + config.dataset), data_save) 
