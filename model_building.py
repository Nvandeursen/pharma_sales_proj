# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 11:38:32 2020

@author: nkraj
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import warnings

from sklearn.metrics import mean_squared_error, mean_absolute_error

# read in the weekly data file
df=pd.read_csv('salesweekly.csv')

# set pharmaceutical categories
cats = ['M01AB','M01AE','N02BA','N02BE','N05B','N05C','R03','R06']

# define function for Mean Absolute Percentage Error
def getMAPE(y_true, y_preds):
    y_true, y_preds = np.array(y_true), np.array(y_preds)
    return np.mean(np.abs((y_true - y_preds) / y_true)) * 100


# naive forecast for each category
for cat in cats:
    ds=df[cat]
    dataframe = pd.concat([ds.shift(1), ds], axis=1)
    dataframe.columns = ['t+1', 't-1']
    X = dataframe['t-1']
    Y = dataframe['t+1']
    size = len(dataframe)-50
    test, predictions = X[size:len(X)], Y[size:len(Y)]
    error = mean_squared_error(test, predictions)
    pct_error = getMAPE(test, predictions)
    print(cat+' (MSE=' + str(round(error,2))+', MAPE='+ str(round(pct_error,2)) +'%)')

# average method forecasting
for cat in cats:
    X=df[cat].values
    size = len(X)-50
    test = X[size:len(X)] 
    mean = np.mean(X[0:size])
    predictions = np.full(50,mean)
    error = mean_squared_error(test, predictions)
    pct_error = getMAPE(test, predictions)
    print(cat+' (MSE=' + str(round(error,2))+', MAPE='+ str(round(pct_error,2)) +'%)')
    
# seasonal naive forecasting
for cat in cats:
    X=df[cat].values
    size = len(X)-52
    test = X[size:len(X)]
    train = X[0:size]
    predictions=list()
    history = [x for x in train]
    for i in range(len(test)):
        obs=list()
        for y in range(1,5):
            obs.append(train[-(y*52)+i])
        yhat = np.mean(obs)
        predictions.append(yhat)
        history.append(test[i])
    error = mean_squared_error(test, predictions)
    pct_error = getMAPE(test, predictions)
    print(cat+' (MSE=' + str(round(error,2))+', MAPE='+ str(round(pct_error,2)) +'%)')
    

### Rolling Forecast Model

# optimize parameters p, d, q
# create manual Grid Search to evaluate on MSE
def evaluate_arima_model(X, arima_order):
    train_size = int(len(X) * 0.67)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    predictions = []
    for t in range(len(test)):
        warnings.filterwarnings("ignore")
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit()
        y_hat = model_fit.forecast(steps=1)
        predictions.append(y_hat)
        history.append(test[t])
    error = mean_squared_error(test, predictions)
    warnings.filterwarnings("default")
    return error

def evaluate_models(cat, dataset, p_values, d_values, q_values):
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    mse = evaluate_arima_model(dataset, order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                except:
                    continue
    print(cat+' - Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))

# specify parameters for search
p_values = range(0, 3)
d_values = range(0, 2)
q_values = range(0, 3)

#### WARNING: loop below takes approx. 3 min per category to run
ts = time.time()
for cat in cats:
    evaluate_models(cat, df[cat].values, p_values, d_values, q_values)
    # evaluate_models('N02BE', df['N02BE'].values, p_values, d_values, q_values)
time.time() - ts

### Long Term Model
# ARIMA Model
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import auto_arima

ts = time.time()
# ARIMA Results
for cat in cats:
    # define X based on pharmaceutical category
    X = df[cat].values
    # define train and test sets (split 67/33)
    train, test = X[:int(len(X)*0.67)], X[int(len(X)*0.67):]
    # train model
    model_auto = auto_arima(train, max_p=5, max_q=5, max_d=1, max_order=11, 
                            random_state=42, seasonal=False, start_q=0, 
                            start_p = 0, start_d = 0, suppress_warnings=True)
    
    # make predictions
    predictions = model_auto.predict(n_periods=len(test))
    # calculate MSE and extract order
    error = mean_squared_error(test, predictions)
    best_order = model_auto.order
    print(cat+' - Best ARIMA%s MSE=%.3f' % (best_order, error))
time.time() - ts


### Long Term Forecast
# optimize parameters p, d, q
# create manual Grid Search to evaluate on MSE
def evaluate_arima_model_long(X, arima_order):
    train_size = int(len(X) * 0.67)
    train, test = X[0:train_size], X[train_size:]
    warnings.filterwarnings("ignore")
    model = ARIMA(train, order=arima_order)
    model_fit = model.fit()
    predictions = model_fit.predict(1,len(test))
    error = mean_squared_error(test, predictions)
    warnings.filterwarnings("default")
    return error

def evaluate_models_long(cat, dataset, p_values, d_values, q_values):
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    mse = evaluate_arima_model_long(dataset, order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                except:
                    continue
    print(cat+' - Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))

p_values = range(0,6)
d_values = range(0,2)
q_values = range(0,6)
    
ts = time.time()
for cat in cats:
    evaluate_models_long(cat, df[cat].values, p_values, d_values, q_values)
    # evaluate_models_long('N02BE', df['N02BE'].values, p_values, d_values, q_values)
time.time() - ts

# Plot Long Term Forecasts

# set series
M01AB = {'series':'M01AB', 'p':0,'d':0, 'q':0}
M01AE = {'series':'M01AE', 'p':0,'d':0, 'q':0}
N02BA = {'series':'N02BA', 'p':0,'d':0, 'q':0}
N02BE = {'series':'N02BE', 'p':0,'d':0, 'q':0}
N05B = {'series':'N05B', 'p':0,'d':0, 'q':0}
N05C = {'series':'N05C', 'p':0,'d':0, 'q':0}
R03 = {'series':'R03', 'p':0,'d':0, 'q':0}
R06 = {'series':'R06', 'p':5,'d':1, 'q':5}

resultsLongtermdf = pd.DataFrame(index=['ARIMA MSE', 'ARIMA MAPE'],  
                                 columns = cats)

subplotindex = 0
numrows = 3
numcols = 3
fig, ax = plt.subplots(numrows, numcols, figsize=(18,12))
plt.subplots_adjust(wspace=0.1, hspace=0.3)

for x in [M01AB, M01AE, N02BA, N02BE, N05B, N05C, R03, R06]:
    rowindex=math.floor(subplotindex/numcols)
    colindex=subplotindex-(rowindex*numcols)
    X = df[x['series']].values
    size = int(len(X) - 50)
    train, test = X[0:size], X[size:len(X)]
    model = ARIMA(train, order=(x['p'],x['d'],x['q']))
    model_fit = model.fit()
    forecast = model_fit.predict(1,len(test))
    error = mean_squared_error(test, forecast)
    resultsLongtermdf.loc['ARIMA MSE',x['series']]=error
    # resultsLongtermdf.loc['ARIMA MAPE',x['series']]=perror
    ax[rowindex,colindex].set_title(x['series']+' (MSE=' + str(round(error,2))+')')
    ax[rowindex,colindex].legend(['Real', 'Predicted'], loc='upper left')
    ax[rowindex,colindex].plot(test)
    ax[rowindex,colindex].plot(forecast, color='red')
    subplotindex=subplotindex+1
plt.show()

# Plot Rolling Forecasts

# set series
M01AB = {'series':'M01AB', 'p':0,'d':0, 'q':0}
M01AE = {'series':'M01AE', 'p':2,'d':0, 'q':0}
N02BA = {'series':'N02BA', 'p':5,'d':1, 'q':1}
N02BE = {'series':'N02BE', 'p':2,'d':0, 'q':0}
N05B = {'series':'N05B', 'p':0,'d':0, 'q':5}
N05C = {'series':'N05C', 'p':0,'d':0, 'q':1}
R03 = {'series':'R03', 'p':5,'d':1, 'q':1}
R06 = {'series':'R06', 'p':1,'d':0, 'q':1}

# create results dataframe
resultsRollingdf = pd.DataFrame(index=['ARIMA MSE'], columns = cats)

subplotindex = 0
numrows = 4
numcols = 2
fig, ax = plt.subplots(numrows, numcols, figsize=(18,15))
plt.subplots_adjust(wspace=0.1, hspace=0.3)

for x in [M01AB, M01AE, N02BA, N02BE, N05B, N05C, R03, R06]:
    rowindex=math.floor(subplotindex/numcols)
    colindex=subplotindex-(rowindex*numcols)
    X = df[x['series']].values
    size = len(X)-50
    train, test = X[0:size], X[size:len(X)]
    history = [x for x in train]
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=(x['p'],x['d'],x['q']))
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
    error = mean_squared_error(test, predictions)
    resultsRollingdf.loc['ARIMA MSE',x['series']]=error
    ax[rowindex,colindex].set_title(x['series']+' (MSE=' + str(round(error,2))+')')
    ax[rowindex,colindex].legend(['Real', 'Predicted'], loc='upper left')
    ax[rowindex,colindex].plot(test)
    ax[rowindex,colindex].plot(predictions, color='red')
    subplotindex=subplotindex+1
plt.show()