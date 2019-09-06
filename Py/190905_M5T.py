# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 17:53:30 2019

@author: Paul
"""

## Install
pip install numpy
pip install pandas
pip install matplotlib
pip install scipy

## Load toolkits
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt

## Data types
# FUnction:     functionName()
# Variable:     variableName = 25
# Dictionary    tel = {'jack': 4098, 'sape': 4139}     like lists, but key-value pair: {key1 : value1, ...}
#               tel['jack']

## Check data
# type(df)

# Coloumn names
header = wifi_train.dtypes.index
print(header)

## Explore data
wifi_train.head()
wifi_train.describe()
wifi_train.info()

## Visualization
## Histogram
plt.hist(wifi_train['LONGITUDE'], bins = 4)
plt.show() #for Jupyter

## Two subplots sharing axes
f, (hist1, hist2) = plt.subplots(2, sharey=True)
hist1.hist(wifi_train['LONGITUDE'], bins = 4)
hist2.hist(wifi_train['LATITUDE'], bins = 4)
hist1.set_title('Sharing y-axes')

## Line plots
# illustrate the range and mode of any given numerical variable
# preventing bias and model overfit and even identifying collinearity.
plt.plot(wifi_train['LONGITUDE'])

## Scatter 
# understanding any possible relationships between the data, but as you are 
# aware, does not always point to any causal relationship.
x = wifi_train['LONGITUDE']
y = wifi_train['LATITUDE']
plt.scatter(x,y)

##  Box/Whisker Plots
# identifying feature with outliers or extreme variances in the observations
# lines extending the 'box' can demonstrate the existence of outliers if point 
# go beyond lines' end; doesn't necessarily mean there is an issue 
bp = wifi_train['LATITUDE']
plt.boxplot(bp, 0, 'gD')

## Correlation & Covariance matrix
corrMat = wifi_train.corr()
print(corrMat)

covMat = wifi_train.cov()
print(covMat)

#### Modelling
## Load toolkits
#Estimators
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn import linear_model

#Model metrics
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

#cross validation
from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import LeaveOneOut
from sklearn.cross_validation import LeaveOneOut

## Selecting data
#variableName = wifi_train.iloc[rows,columns]
#variableName = wifi_train['coumnName’]
#variableName = wifi_train.iloc[:,0:4]

#features
features = wifi_train.iloc[:,0:520]
print('Summary of feature sample')
features.head()

#dependent variable
depVar = wifi_train['LONGITUDE']
depVar.head()

#Training Set (Feature Space: X Training)
X_train = (features[: round(features.shape[0]*0.75)])
X_train.head()

#Dependent Variable Training Set (y Training)
y_train = depVar[: round(depVar.shape[0]*0.75)]
y_train_count = len(y_train.index)
print('The number of observations in the Y training set are:',str(y_train_count))
y_train.head()

#Testing Set (X Testing)
X_test = features[-round(features.shape[0]*0.25):]
X_test_count = len(X_test.index)
print('The number of observations in the feature testing set is:',str(X_test_count))
print(X_test.head())

#Ground Truth (y_test) 
y_test = depVar[-round(features.shape[0]*0.25):]
y_test_count = len(y_test.index)
print('The number of observations in the Y training set are:',str(y_test_count))
y_test.head()

## Cross validation 
#Cross Validation to prevent overfitting
#Sci-Kit Learn there is a pre-built function
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train)
X_train.shape, X_test.shape

## Training
# Define modle variables
modelRF = RandomForestRegressor()
modelSVR = SVR()
modelLR = LinearRegression()

# Train models
modelRF.fit(X_train,y_train)
modelSVR.fit(X_train,y_train)
modelLR.fit(X_train,y_train)

#### Evaluation
# CV-score
from sklearn.model_selection import cross_val_score
#    1st value:  The score array for test scores on each cv split. (Higher is an indicator of a better performing model)
#    2nd value:  The time for fitting the estimator on the train set for each cv split.
#    3rd Value:  The time for scoring the estimator on the test set for each cv split. 
print(cross_val_score(modelRF, X_train, y_train)) 
#[0.85305472 0.98426452 0.98762407]
print(cross_val_score(modelSVR, X_train, y_train))
#[-0.04113565 -0.44301372 -0.05024049]
print(cross_val_score(modelLR, X_train, y_train))
#[-4.83328824e+16 -3.73577939e+17 -2.91841573e+14]

## Performance metrics
# Default metrics
modelRF.score(X_train,y_train)
#0.996097009613414
modelSVR.score(X_train,y_train)
#0.010806370512635888
modelLR.score(X_train,y_train)
#0.95241931585585
#numerous other ways to check for model overfit and underfit

## Predicting
# Load metric function and define calculations
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Make predictions
predictions = modelRF.predict(X_test)

# Calculate perf. metrics
predRsquared = r2_score(y_test,predictions)
predRMSE = sqrt(mean_squared_error(y_test, predictions))
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


