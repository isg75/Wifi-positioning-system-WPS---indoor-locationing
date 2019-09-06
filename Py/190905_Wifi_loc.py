# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 09:37:22 2019

@author: Paul
"""
## Check toolkits
#pip freeze

## Load packages
#import sys

import os
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from pathlib import Path
from Dora import dora      #for exploration

## Set path
#absFilePath = os.path.abspath(__file__)
my_path = os.path.abspath(os.path.dirname(__file__))
parentDir = os.path.dirname(my_path) 
DataDir = os.path.join(parentDir, 'Raw_data')
filename_train = os.path.abspath(os.path.join(DataDir, 'trainingData.csv'))
filename_val = os.path.abspath(os.path.join(DataDir, 'validationData.csv'))

## Load data
wifi_train_raw = pd.read_csv(filename_train)
wifi_val_raw = pd.read_csv(filename_train)

## Transformations
## 1. Change RSSI
wifi_train = wifi_train_raw
type(wifi_train)
wifi_train.dtypes

wifi_train.iloc[:, 0:520] = wifi_train.iloc[:, 0:520] + 105
wifi_train.iloc[:, 0:520] = wifi_train.iloc[:, 0:520].replace(to_replace = 205, value = 0)


## 2. Change scales
plt.scatter(wifi_train.iloc[:, 520] , wifi_train.iloc[:, 521])
#LONGITUDE
wifi_train.iloc[:, 520] = wifi_train.iloc[:, 520] + 7700
#LATITUDE
wifi_train.iloc[:, 521] = wifi_train.iloc[:, 521] - 4864700

plt.scatter(wifi_train.iloc[:, 520] , wifi_train.iloc[:, 521])

## 3. Transformation - Rotation of axis
# Find angel
g = 4864930 - 4864750
a = 7690 - 7340
pi = np.arctan(g/a)
pi_deg = np.arctan(g/a)*180/np.pi

# Set angel
theta = np.radians(27.5)

# Define transformation matrix
tM = np.matrix([[np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]])
tM

# Select coordination columns
v = np.matrix(wifi_train_raw[['LONGITUDE', 'LATITUDE', 'FLOOR']])
v.shape
v = np.transpose(v)

type(tM)
type(v)

# Transform coordination
tMv = tM * v
type(tMv)
plt.scatter(np.array(tMv[0,:]), np.array(tMv[1,:]))
np.transpose(tMv)

wifi_train[['LONGITUDE', 'LATITUDE', 'FLOOR']] = np.transpose(tMv)
        
#### Exploration
## Histogram
wifi_expl = wifi_train

# RSSI for all WAPs
WAPs = wifi_expl.filter(like = "WAP")
WAPsmelted = pd.melt(WAPs)
WAPsmelted_wo0 = WAPsmelted.loc[(WAPsmelted.iloc[:,1] != 0)]
plt.hist(WAPsmelted_wo0['value'], bins = 21)

# Number of WAPs detected
wifi_expl['WAPs_det'] = wifi_expl.iloc[:,0:520].gt(0).sum(axis=1)
#wifi_expl['WAPs_det'] = wifi_expl.iloc[:,0:520].lt(0).sum(axis=1)
#wifi_expl['WAPs_det'] = wifi_expl.iloc[:,0:520].eq(0).sum(axis=1)

plt.hist(wifi_expl['WAPs_det'], bins = 520)
plt.ylabel('Frequency')
plt.xlabel('Number of WAPs detected')
plt.axis([0, 50, 0, 1600])
plt.suptitle('Frequency of WAPs detected')

# Histogram for PHONEID and USERID
plt.hist(wifi_expl['USERID'], bins = 18)
plt.hist(wifi_expl['PHONEID'], bins = 18)

# PHONE ID 3D Scatter
import plotly.express as px
from plotly.offline import plot

fig = px.scatter_3d(wifi_expl, x='LONGITUDE', y='LATITUDE', z='FLOOR',
              color='PHONEID', size_max = 10, opacity = 0.7)
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
plot(fig)

#### Baseline model ####
## Data split
#features
features = wifi_train.iloc[:,0:520]

#Training Set (Feature Space: X Training)
X_train_long = (features[: round(features.shape[0]*0.75)])
X_train_lat = (features[: round(features.shape[0]*0.75)])
X_train_floor = (features[: round(features.shape[0]*0.75)])
X_train_long.head()

#Dependent Variable Training Set (y Training)
y_train_long = wifi_train['LONGITUDE'][: round(features.shape[0]*0.75)]
y_train_lat = wifi_train['LATITUDE'][: round(features.shape[0]*0.75)]
y_train_floor = wifi_train['FLOOR'][: round(features.shape[0]*0.75)]

y_train_long_count = len(y_train_long.index)
print('The number of observations in the Y training set are:',str(y_train_long_count))
y_train_long.head()

#Testing Set (X Testing)
X_test_long = features[-round(features.shape[0]*0.25):]
X_test_lat = features[-round(features.shape[0]*0.25):]
X_test_floor = features[-round(features.shape[0]*0.25):]

X_test_long_count = len(X_test_long.index)
print('The number of observations in the feature testing set is:',str(X_test_long_count))
print(X_test_long.head())

#Ground Truth (y_test) 
y_test_long = wifi_train['LONGITUDE'][-round(features.shape[0]*0.25):]
y_test_lat = wifi_train['LATITUDE'][-round(features.shape[0]*0.25):]
y_test_floor = wifi_train['FLOOR'][-round(features.shape[0]*0.25):]

y_test_long_count = len(y_test_long.index)
print('The number of observations in the Y training set are:', str(y_test_long_count))
y_test_long.head()

## Modelling
# Load Estimators
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import RadiusNeighborsRegressor

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import GradientBoostingRegressor

# Define models
modelkNN_long = KNeighborsRegressor()
modelkNN_rad_long = RadiusNeighborsRegressor()
modelLR_long = LinearRegression()
modelRF_long = RandomForestRegressor()
modelGBR_long = GradientBoostingRegressor()

modelkNN_lat = KNeighborsRegressor()
modelkNN_rad_lat = RadiusNeighborsRegressor()
modelLR_lat = LinearRegression()
modelRF_lat = RandomForestRegressor()
modelGBR_lat = GradientBoostingRegressor()

modelkNN_floor = KNeighborsRegressor()
modelkNN_rad_floor = RadiusNeighborsRegressor()
modelLR_floor = LinearRegression()
modelRF_floor = RandomForestRegressor()
modelGBR_floor = GradientBoostingRegressor()

# Fit models - Baseline
modelkNN_long.fit(X_train_long, y_train_long)
#KNeighborsRegressor(algorithm='auto', leaf_size=30, metric='minkowski',
#                    metric_params=None, n_jobs=None, n_neighbors=5, p=2,
#                    weights='uniform')

modelkNN_rad_long.fit(X_train_long, y_train_long)
#RadiusNeighborsRegressor(algorithm='auto', leaf_size=30, metric='minkowski',
#                         metric_params=None, n_jobs=None, p=2, radius=1.0,
#                         weights='uniform')
modelLR_long.fit(X_train_long, y_train_long) #LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)
modelRF_long.fit(X_train_long, y_train_long) 
#RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
#                      max_features='auto', max_leaf_nodes=None,
#                      min_impurity_decrease=0.0, min_impurity_split=None,
#                      min_samples_leaf=1, min_samples_split=2,
#                      min_weight_fraction_leaf=0.0, n_estimators=10,
#                      n_jobs=None, oob_score=False, random_state=None,
#                      verbose=0, warm_start=False)
modelGBR_long.fit(X_train_long, y_train_long)
#GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
#                          learning_rate=0.1, loss='ls', max_depth=3,
#                          max_features=None, max_leaf_nodes=None,
#                          min_impurity_decrease=0.0, min_impurity_split=None,
#                          min_samples_leaf=1, min_samples_split=2,
#                          min_weight_fraction_leaf=0.0, n_estimators=100,
#                          n_iter_no_change=None, presort='auto',
#                          random_state=None, subsample=1.0, tol=0.0001,
#                          validation_fraction=0.1, verbose=0, warm_start=False)#n_estimators = 100

modelkNN_lat.fit(X_train_lat, y_train_lat)
modelkNN_rad_lat.fit(X_train_lat, y_train_lat)
modelLR_lat.fit(X_train_lat, y_train_lat)
modelRF_lat.fit(X_train_lat, y_train_lat)
modelGBR_lat.fit(X_train_lat, y_train_lat)

modelkNN_floor.fit(X_train_lat, y_train_lat)
modelkNN_rad_floor.fit(X_train_lat, y_train_lat)
modelLR_floor.fit(X_train_lat, y_train_lat)
modelRF_floor.fit(X_train_lat, y_train_lat)
modelGBR_floor.fit(X_train_lat, y_train_lat)


# CV -score
from sklearn.model_selection import cross_val_score

# CV value = 5
CV_score_mkNN_long = cross_val_score(modelkNN_long, X_train_long, y_train_long)
#CV_score_kNNrad_long = cross_val_score(modelkNN_rad_long, X_train_long, y_train_long) # ?
CV_score_mLR_long = cross_val_score(modelLR_long, X_train_long, y_train_long)
CV_score_mRF_long = cross_val_score(modelRF_long, X_train_long, y_train_long)
CV_score_mGBR_long = cross_val_score(modelGBR_long, X_train_long, y_train_long)

print(CV_score_mkNN_long)
print(CV_score_mLR_long)
print(CV_score_mkNN_long, CV_score_mLR_long, CV_score_mRF_long, CV_score_mGBR_long)


CV_score_mkNN_lat = cross_val_score(modelkNN_lat, X_train_lat, y_train_lat)
#CV_score_kNNrad_lat = cross_val_score(modelkNN_rad_lat, X_train_lat, y_train_lat)
CV_score_mLR_lat = cross_val_score(modelLR_lat, X_train_lat, y_train_lat)
CV_score_mRF_lat = cross_val_score(modelRF_lat, X_train_lat, y_train_lat)
CV_score_mGBR_lat = cross_val_score(modelGBR_lat, X_train_lat, y_train_lat)

CV_score_mkNN_floor = cross_val_score(modelkNN_floor, X_train_floor, y_train_floor)
#CV_score_kNNrad_floor = cross_val_score(modelkNN_rad_floor, X_train_floor, y_train_floor)
CV_score_mLR_floor = cross_val_score(modelLR_floor, X_train_floor, y_train_floor)
CV_score_mRF_floor = cross_val_score(modelRF_floor, X_train_floor, y_train_floor)
CV_score_mGBR_floor = cross_val_score(modelGBR_floor, X_train_floor, y_train_floor)

# Score
modelkNN_long.score(X_train_long, y_train_long)
#0.9951014567058754
modelLR_long.score(X_train_long, y_train_long)
#0.9554266250008222
modelRF_long.score(X_train_long, y_train_long)
#0.9959336416063866
modelGBR_long.score(X_train_long, y_train_long)
#0.9700627335401973

modelkNN_lat.score(X_train_lat, y_train_lat)
#0.9872333175157133
modelLR_lat.score(X_train_lat, y_train_lat)
#0.8886766112468676
modelRF_lat.score(X_train_lat, y_train_lat)
#0.9912928103181751
modelGBR_lat.score(X_train_lat, y_train_lat)
#0.910277736594106

modelkNN_floor.score(X_train_floor, y_train_floor)
#-44443.302461260566
modelLR_floor.score(X_train_floor, y_train_floor)
#-44308.920466185664
modelRF_floor.score(X_train_floor, y_train_floor)
#-44370.35980526809
modelGBR_floor.score(X_train_floor, y_train_floor)
#-44216.78694179182

# Load metric function and define calculations
from math import sqrt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error #(y_true, y_pred)
from sklearn.metrics import max_error #(y_true, y_pred)

# Make predictions
pred_long_kNN = modelkNN_long.predict(X_test_long)
pred_long_LR = modelLR_long.predict(X_test_long)
pred_long_RF = modelRF_long.predict(X_test_long)
pred_long_GBR = modelGBR_long.predict(X_test_long)

# Calculate perf. metrics
perf_kNN_long = pd.DataFrame()
def performance(predictions, y_test, df):
    df['Rsq'] = r2_score(y_test, predictions)
    df['RMSE'] = sqrt(mean_squared_error(y_test, predictions))
    df['MAE'] = mean_absolute_error(y_test, predictions)
    df['Emax'] = max_error(y_test, predictions)
    return df
performance(pred_long_kNN, y_test_long, perf_kNN_long)
    
    
print('R Squared: %.3f' % predRsquared)
# R Squared: 0.984
print('RMSE: %.3f' % predRMSE)
# RMSE: 15.962

## Visualize
type(y_test)
predictions = pd.DataFrame(predictions)
type(predictions)
plt.scatter(y_test, predictions, alpha = 0.5) #,color=['blue','green'])
plt.xlabel('Ground Truth')
plt.ylabel('Predictions')


