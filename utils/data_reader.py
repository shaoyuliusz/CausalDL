import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
"""
read IHDP data function
"""
# curPath = os.path.abspath(os.path.dirname(__file__))
# rootPath = os.path.split(curPath)[0]
# sys.path.append(rootPath)
def read_SIPP_data(file):
    data = pd.read_stata(file)
    y = np.array(data['net_tfa']).astype('float32')
    x = np.array(data[["age", "inc", "educ", "fsize", "marr", "twoearn", "db", "pira","hown"]]).astype('float')
    t = np.array(data['e401']).astype('float32')
    
    data = {'x':x, 't':t, 'y':y}
    data['t']=data['t'].reshape(-1,1)
    data['y']=data['y'].reshape(-1,1)
    #rescaling y between 0 and 1 often makes training of DL regressors easier
    data['y_scaler'] = StandardScaler().fit(data['y'])
    data['ys'] = data['y_scaler'].transform(data['y'])
    return data

def read_IHDP_data(training_data,testing_data,i=7):
    with open(training_data,'rb') as trf, open(testing_data,'rb') as tef:
        train_data=np.load(trf); test_data=np.load(tef)
        y=np.concatenate(   (train_data['yf'][:,i],   test_data['yf'][:,i])).astype('float32') #most GPUs only compute 32-bit floats
        t=np.concatenate(   (train_data['t'][:,i],    test_data['t'][:,i])).astype('float32')
        x=np.concatenate(   (train_data['x'][:,:,i],  test_data['x'][:,:,i]),axis=0).astype('float32')
        mu_0=np.concatenate((train_data['mu0'][:,i],  test_data['mu0'][:,i])).astype('float32')
        mu_1=np.concatenate((train_data['mu1'][:,i],  test_data['mu1'][:,i])).astype('float32')
 
        data={'x':x,'t':t,'y':y,'t':t,'mu_0':mu_0,'mu_1':mu_1}
        data['t']=data['t'].reshape(-1,1) #we're just padding one dimensional vectors with an additional dimension 
        data['y']=data['y'].reshape(-1,1)
        
        #rescaling y between 0 and 1 often makes training of DL regressors easier
        data['y_scaler'] = StandardScaler().fit(data['y'])
        data['ys'] = data['y_scaler'].transform(data['y'])
 
    return data