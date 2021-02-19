# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 11:38:32 2020

@author: nkraj
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

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
    
# ARIMA Model
from statsmodels.tsa.arima_model import ARIMA

# # optimize parameters p, d, q
# # create manual Grid Search to evaluate on MSE
# # **warning** the below code takes extremely long to run
# def evaluate_arima_model(X, arima_order):
#     train_size = int(len(X) * 0.66)
#     train, test = X[0:train_size], X[train_size:]
#     history = [x for x in train]
#     predictions = list()
#     for t in range(len(test)):
#         model = ARIMA(history, order=arima_order)
#         model_fit = model.fit(disp=0)
#         yhat = model_fit.forecast()[0]
#         predictions.append(yhat)
#         history.append(test[t])
#     error = mean_squared_error(test, predictions)
#     return error

# def evaluate_models(cat, dataset, p_values, d_values, q_values):
#     best_score, best_cfg = float("inf"), None
#     for p in p_values:
#         for d in d_values:
#             for q in q_values:
#                 order = (p,d,q)
#                 try:
#                     mse = evaluate_arima_model(dataset, order)
#                     if mse < best_score:
#                         best_score, best_cfg = mse, order
#                 except:
#                     continue
#     print(cat+' - Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))

# # specify parameters for search
# p_values = range(0, 6)
# d_values = range(0, 2)
# q_values = range(0, 6)

# for cat in cats:
#     evaluate_models(cat, df[cat].values, p_values, d_values, q_values)
