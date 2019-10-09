# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 15:25:11 2019

@author: Paul W.
"""
# =============================================================================
#### Initialysing
# =============================================================================

## Load toolkits
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load scalers
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer

# Load estimators (Regression & Classification)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from xgboost import XGBRegressor
from xgboost import XGBClassifier

# Load performance metrics
from math import sqrt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import max_error
from sklearn.metrics import explained_variance_score

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score

# Visualization
import plotly.express as px
from plotly.offline import plot

# Resampling
from sklearn.utils.random import sample_without_replacement


## Set path
my_path = os.path.abspath(os.path.dirname('__file__'))
parentDir = os.path.dirname(my_path) 
DataDir = os.path.join(parentDir, 'Raw_data')
SubmissionDir = os.path.join(parentDir, 'Submission')


## Set file names
filename_train = os.path.abspath(os.path.join(DataDir, 'trainingData.csv'))
filename_val = os.path.abspath(os.path.join(DataDir, 'validationData.csv'))


## Load data
wifi_train_raw = pd.read_csv(filename_train)
wifi_val_raw = pd.read_csv(filename_val)



# =============================================================================
#### Data preparation  
# =============================================================================

## 1. Transformation of RSSI
wifi_train = wifi_train_raw.copy()
wifi_val = wifi_val_raw.copy()

wifi_train.iloc[:, :520] = pd.DataFrame(wifi_train.iloc[:, :520] + 105).replace(to_replace = 205, value = 0)
wifi_val.iloc[:, :520] = pd.DataFrame(wifi_val.iloc[:, :520] + 105).replace(to_replace = 205, value = 0)


## 2. Transformation of 3D-Space (Rotation around z-axis)
# Find angel
pi_deg = np.arctan((wifi_train['LATITUDE'].max() - wifi_train['LATITUDE'].min())/
                   (abs(wifi_train['LONGITUDE'].min()) - abs(wifi_train['LONGITUDE'].max())))*180/np.pi

# Set angel
theta = np.radians(27.5)

# Define transformation matrix
tM = np.matrix([[np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]])

# Select coordination columns
v_train = np.matrix(wifi_train[['LONGITUDE', 'LATITUDE', 'FLOOR']])
v_train.shape
v_train = np.transpose(v_train)

type(tM)
type(v_train)

v_val = np.matrix(wifi_val_raw[['LONGITUDE', 'LATITUDE', 'FLOOR']])
v_val.shape
v_val = np.transpose(v_val)

# Transform coordination
tMv_train = tM * v_train
tMv_val = tM * v_val

type(tMv_train)

wifi_train[['LONGITUDE', 'LATITUDE', 'FLOOR']] = np.transpose(tMv_train)
wifi_val[['LONGITUDE', 'LATITUDE', 'FLOOR']] = np.transpose(tMv_val)


## 3. Change scales
# LONGITUDE
wifi_train.iloc[:, 520] = wifi_train.iloc[:, 520] + 2253200
wifi_val.iloc[:, 520] = wifi_val.iloc[:, 520] + 2253200

# LATITUDE
wifi_train.iloc[:, 521] = wifi_train.iloc[:, 521] - 4311690
wifi_val.iloc[:, 521] = wifi_val.iloc[:, 521] - 4311690

plt.scatter(wifi_train.iloc[:, 520] , wifi_train.iloc[:, 521])
plt.scatter(wifi_val.iloc[:, 520] , wifi_val.iloc[:, 521])



# =============================================================================
#### Data exploration
# =============================================================================

## 3D Scatterplot of raw training data
fig_raw = px.scatter_3d(wifi_train_raw, x='LONGITUDE', y='LATITUDE', z='FLOOR',
                        color = 'BUILDINGID', size_max = 5, opacity = 0.7)
fig_raw.update_layout(margin=dict(l=0, r=0, b=0, t=0))
plot(fig_raw)


## 3D Scatterplot of raw training data
fig_train = px.scatter_3d(wifi_train, x='LONGITUDE', y='LATITUDE', z='FLOOR',
                          color = 'BUILDINGID', size_max = 5, opacity = 0.7)
fig_train.update_layout(margin=dict(l=0, r=0, b=0, t=0))
plot(fig_train)


## Histogram: Frequency of RSSI signal strength over all WAPs
# Change to merged format
WAPsmelted_train = pd.melt(wifi_train.filter(like = "WAP"))
WAPsmelted_wo100_train = WAPsmelted_train.loc[(WAPsmelted_train.iloc[:,1] != 0)]

WAPsmelted_val = pd.melt(wifi_val.filter(like = "WAP"))
WAPsmelted_wo100_Val = WAPsmelted_val.loc[(WAPsmelted_val.iloc[:,1] != 0)]

# Plot histogram for train and test set
plt.rcParams.update({'font.size': 10})
fig_RSSI, (ax1, ax2) = plt.subplots(1, 2, tight_layout=True, figsize=(10, 3))

ax1.hist(WAPsmelted_wo100_train['value'], bins = 105)
ax1.set_title('Frequency of RSSI - Train')
ax1.set_xlabel('Signal strength (RSSI) [dbm]')
ax1.set_ylabel('Frequency')

ax2.hist(WAPsmelted_wo100_Val['value'], bins = 105)
ax2.set_title('Frequency of RSSI - Validation')
ax2.set_xlabel('Signal strength (RSSI) [dbm]')
ax2.set_ylabel('Frequency')


## Histgram: Frequency of number of WAPs detected
plt.rcParams.update({'font.size': 10})

fig_det, (ax1, ax2) = plt.subplots(1, 2, tight_layout=True, figsize=(10, 3))

ax1.hist(wifi_train.iloc[:,0:520].gt(0).sum(axis=1), bins = (wifi_train.iloc[:,0:520].gt(0).sum(axis=1)).max())
ax1.set_title('WAP detection per observation - Train')
ax1.set_xlabel('Count of WAPs detected')
ax1.set_ylabel('Frequency')

ax2.hist(wifi_val.iloc[:,0:520].gt(0).sum(axis=1), bins = (wifi_val.iloc[:,0:520].gt(0).sum(axis=1)).max())
ax2.set_title('WAP detection per observation - Validation')
ax2.set_xlabel('Count of WAPs detected')
ax2.set_ylabel('Frequency')


## BAR - USERID
bar = plt.figure(2)
plt.bar(x = wifi_train['USERID'].value_counts().index, height = wifi_train['USERID'].value_counts())
plt.xticks(list(range(0,19)))
plt.title('Frequency of USERID - Training')
plt.xlabel('USERID')
plt.ylabel('Frequency')
plt.interactive(False)
plt.show()


## Histogram of PHONEID
# Define function creating subplots of histograms
def draw_histograms(df, n_rows, n_cols):
    fig = plt.figure()
    plt.rcParams.update({'font.size': 10})
    p_id = list(df['PHONEID'].sort_values().unique())
    for i in range(0, len(df['PHONEID'].sort_values().unique())):
        ax = fig.add_subplot(n_rows,n_cols,i+1)
        melted = pd.melt(df.filter(like = "WAP").loc[df['PHONEID'] == p_id[i], :]) 
        melted.loc[melted['value'] > 0, 'value'].hist(bins = 105)
        ax.set_title("PHONEID" + str(p_id[i]))
        ax.set_xlabel('RSSI signal strength')
        ax.set_ylabel('Frequency')
    fig.tight_layout()  # Improves appearance a bit.
    plt.show()

# Create subplot of histograms - training
x = wifi_train['PHONEID'].value_counts().reset_index()
plt.bar(x = x['index'], height = x.loc[:,'PHONEID'])
plt.title('Frequency of PHONEID used - Training')
plt.xlabel('PHONEID')
plt.ylabel('Frequency')
draw_histograms(wifi_train, 4, 4)

# Create subplot of histograms - validation
xv = wifi_val['PHONEID'].value_counts().reset_index()
plt.bar(x = xv['index'], height = xv.loc[:,'PHONEID'])
plt.title('Frequency of PHONEID used - Validation')
plt.xlabel('PHONEID')
plt.ylabel('Frequency')
draw_histograms(wifi_val, 3, 4)



# =============================================================================
#### Pre-processing
# =============================================================================

### Identify all unique ref. points in train
wifi_train_groups = wifi_train.copy()

# Round LONG./LAT. to meters
d = 0
wifi_train_groups['LONGITUDE'] = round(wifi_train_groups['LONGITUDE'], d)
wifi_train_groups['LATITUDE'] = round(wifi_train_groups['LATITUDE'], d)

# Dicover all unique reference points (refP)
grouping_train = wifi_train_groups.groupby(['FLOOR', 'LONGITUDE', 'LATITUDE'])

# Store unique refPs and create identifier
wifi_train_refP = grouping_train.size().reset_index().rename(columns={0:'count'})
wifi_train_refP['index'] = list(range(0, 926))
wifi_train_refP['setID'] = 'train'
wifi_train_refP['REFPOINTID'] = list(range(0, 926))



### Identify all unique (rounded) refP in val
wifi_val_groups = wifi_val.copy()

# Round LONG./LAT. to meters and create identifier
wifi_val_groups['LONGITUDE'] = round(wifi_val_groups['LONGITUDE'], d)
wifi_val_groups['LATITUDE'] = round(wifi_val_groups['LATITUDE'], d)
wifi_val_groups['index'] = wifi_val_groups.index
wifi_val_groups['setID'] = 'val'

# Identify redundant data (which is in train and val)
wifi_grouped_inner = pd.merge(wifi_train_refP, wifi_val_groups[['FLOOR', 'LONGITUDE', 'LATITUDE', 'index', 'setID']],  how = 'inner', left_on=['FLOOR', 'LONGITUDE', 'LATITUDE'], right_on = ['FLOOR', 'LONGITUDE', 'LATITUDE'])

# Seperate redundant val data
val_4_train = wifi_val_groups.loc[~wifi_val_groups['index'].isin(wifi_grouped_inner['index_y'])]         # 859 rows
val_4_val = wifi_val_groups.loc[wifi_val_groups['index'].isin(wifi_grouped_inner['index_y'])]            # 252 rows

# Create new validation
wifi_val_add = wifi_train.loc[:, :'TIMESTAMP'].sample(1750, replace = False, random_state = 123)
wifi_val_new = pd.concat([wifi_val_add, val_4_val.loc[:, :'TIMESTAMP']], axis = 0)  ## 1750 + 252

# wifi_train_new: 19937 - 1750 (wifi_val_add) + 859 (val_4_train) = 19046
wifi_train_new = pd.concat([wifi_train.loc[~wifi_train.index.isin(wifi_val_add.index), :'TIMESTAMP'], val_4_train.loc[:, :'TIMESTAMP']], axis = 0)



### Address duplicated/empty rows and empty columns

## Duplicated rows
# Identify rows
duplicates_train = wifi_train_new.loc[wifi_train_new.duplicated(subset=None, keep='first') == True,:]  
duplicates_val = wifi_val_new.loc[wifi_val_new.duplicated(subset=None, keep='first') == True,:]        
len(duplicates_train)                       #571 duplicated rows in train
len(duplicates_val)                         #15 duplicated rows in validation

# Remove rows
wifi_train_new = wifi_train_new.drop_duplicates(subset=None, keep='first', inplace=False)
wifi_val_new = wifi_val_new.drop_duplicates(subset=None, keep='first', inplace=False)
len(wifi_train_new)                                               # 18475 rows


## Empty columns (undetected WAPs) in train
# Identify empty columns
duplicates_col_train = pd.DataFrame(wifi_train_new.sum(axis = 0) == 0)
duplicates_col_train.columns = ['duplicates']
duplicates_col_train = duplicates_col_train.loc[duplicates_col_train['duplicates'] == True]
len(duplicates_col_train.index)                                   # 8 empty columns


## Find undetected WAPs in validation set
duplicates_col_val = pd.DataFrame(wifi_val_new.sum(axis = 0) == 0)
duplicates_col_val.columns = ['duplicates']
duplicates_col_val = duplicates_col_val.loc[duplicates_col_val['duplicates'] == True]
len(duplicates_col_val.index)                                     # 72 undetected WAPs (empty columns)


## Drop 8 WAPs not detected in train_new
wifi_train_unique512 = wifi_train_new.drop(columns=duplicates_col_train.index)
wifi_val_unique512 = wifi_val_new.drop(columns=duplicates_col_train.index)      #Drop same columns in validation set

# Check
wifi_train_new.shape[1] - wifi_train_unique512.shape[1]           # 8 dropped
(wifi_train_unique512.sum(axis = 0) == 0).value_counts()          # False = 521; No True anymore
(wifi_val_unique512.sum(axis = 0) == 0).value_counts()            # False = 449; True = 72; --> still 72 in validation


## Empty rows (w/o WAP detection)
# Discover rows with zero WAPs detected
WAPs_det = wifi_train_unique512.filter(like = 'WAP').gt(0).sum(axis = 1)
plt.hist(WAPs_det, bins = 512)

zero_WAPs_det_id = WAPs_det.loc[WAPs_det == 0].index           
len(zero_WAPs_det_id)                                             # 70 rows with 0 WAPs detected

WAPs_det_val = wifi_val_unique512.filter(like = 'WAP').gt(0).sum(axis = 1)
zero_WAPs_det_id_val = WAPs_det_val.loc[WAPs_det_val == 0].index  # 4 rows with 0 WAPs detected
 
# Drop rows with zero WAPs detected
wifi_train_unique  = wifi_train_unique512.drop(index = zero_WAPs_det_id) 
wifi_val_unique  = wifi_val_unique512.drop(index = zero_WAPs_det_id_val) 



### Set data types

## train
wifi_train_unique.dtypes
wifi_train_unique['FLOOR'] = wifi_train_unique['FLOOR'].astype('category')
wifi_train_unique['BUILDINGID'] = wifi_train_unique['BUILDINGID'].astype('category')
wifi_train_unique['SPACEID'] = wifi_train_unique['SPACEID'].astype('category')
wifi_train_unique['RELATIVEPOSITION'] = wifi_train_unique['RELATIVEPOSITION'].astype('category')
wifi_train_unique['USERID'] = wifi_train_unique['USERID'].astype('category')
wifi_train_unique['PHONEID'] = wifi_train_unique['PHONEID'].astype('category')

## Val
wifi_val_unique.dtypes
wifi_val_unique['FLOOR'] = wifi_val_unique['FLOOR'].astype('category')
wifi_val_unique['BUILDINGID'] = wifi_val_unique['BUILDINGID'].astype('category')
wifi_val_unique['SPACEID'] = wifi_val_unique['SPACEID'].astype('category')
wifi_val_unique['RELATIVEPOSITION'] = wifi_val_unique['RELATIVEPOSITION'].astype('category')
wifi_val_unique['USERID'] = wifi_val_unique['USERID'].astype('category')
wifi_val_unique['PHONEID'] = wifi_val_unique['PHONEID'].astype('category')



### Data splitting

## Training set
# Features (X_train)
X_train = wifi_train_unique.filter(like = 'WAP', axis = 1)

# Dependent Variable Training Set (y Training)
y_train_long = wifi_train_unique['LONGITUDE']
y_train_lat = wifi_train_unique['LATITUDE']
y_train_floor = wifi_train_unique['FLOOR']
y_train_building = wifi_train_unique['BUILDINGID']
y_train_long_count = len(y_train_long.index)
print('The number of observations in the Y training set are:',str(y_train_long_count))


## Testing Set
# Features (X_val)
X_val = wifi_val_unique.loc[:, X_train.columns]
X_val_count = len(X_val.index)
print('The number of observations in the feature testing set is:',str(X_val_count))

# Ground Truth (y_val) 
y_val_long = wifi_val_unique['LONGITUDE']
y_val_lat = wifi_val_unique['LATITUDE']
y_val_floor = wifi_val_unique['FLOOR']
y_val_building = wifi_val_unique['BUILDINGID']
y_val_long_count = len(y_val_long.index)
print('The number of observations in the Y training set are:', str(y_val_long_count))



### Scaling

## Normalize rows
# Train
scaler_normObs = Normalizer(norm = 'l2', copy = True)
X_train_normObs = scaler_normObs.fit_transform(X_train)
X_train_normObs = pd.DataFrame(X_train_normObs, index = X_train.index, columns = X_train.columns)

# Val
X_val_normObs = scaler_normObs.transform(X_val)
X_val_normObs = pd.DataFrame(X_val_normObs, index = X_val.index, columns = X_val.columns)



# =============================================================================
#### Modelling
# =============================================================================

### Building
## Define models
modelkNN_build = KNeighborsClassifier()
modelRF_build = RandomForestClassifier()


## Fit models
modelkNN_build.fit(X_train_normObs, y_train_building) 
modelRF_build.fit(X_train_normObs, y = y_train_building)


## Make predictions
pred_build_kNN = modelkNN_build.predict(X_val_normObs)
pred_build_RF = modelkNN_build.predict(X_val_normObs)


## Calculate performance metrics accuracy and kappa
# Class report
acc_kNN = classification_report(y_val_building, pred_build_kNN, labels = [0, 1, 2])   
acc_RF = classification_report(y_val_building, pred_build_RF, labels = [0, 1, 2])
print(acc_kNN, acc_RF)

# Accuracy & Kappa
print(accuracy_score(y_val_building, pred_build_kNN), cohen_kappa_score(y_val_building, pred_build_kNN))
print(accuracy_score(y_val_building, pred_build_RF), cohen_kappa_score(y_val_building, pred_build_RF))

## Concat X_train and prediction for building
X_train_normObs_casc = pd.concat([X_train_normObs, y_train_building], axis = 1)
X_val_normObs_casc = pd.concat([X_val_normObs, y_val_building], axis = 1)



### LONGITUDE, LATITUDE and FLOOR

## Define models
modelkNN_long = KNeighborsRegressor(n_neighbors=7, weights='distance')
modelRF_long = RandomForestRegressor()

modelkNN_lat = KNeighborsRegressor(n_neighbors=2, weights='distance')
modelRF_lat = RandomForestRegressor()

modelkNN_floor = KNeighborsClassifier(n_neighbors=2, weights='distance')
modelRF_floor = RandomForestClassifier()


## Fit models
modelkNN_long.fit(X_train_normObs_casc, y_train_long)
modelRF_long.fit(X_train_normObs_casc, y_train_long)

modelkNN_lat.fit(X_train_normObs_casc, y_train_lat)
modelRF_lat.fit(X_train_normObs_casc, y_train_lat)

modelkNN_floor.fit(X_train_normObs_casc, y_train_floor)
modelRF_floor.fit(X_train_normObs_casc, y_train_floor)


## Make predictions
pred_long_kNN = modelkNN_long.predict(X_val_normObs_casc)
pred_long_RF = modelRF_long.predict(X_val_normObs_casc)

pred_lat_kNN = modelkNN_lat.predict(X_val_normObs_casc)
pred_lat_RF  = modelRF_lat.predict(X_val_normObs_casc)

pred_floor_kNN = modelkNN_floor.predict(X_val_normObs_casc)
pred_floor_RF = modelRF_floor.predict(X_val_normObs_casc)



### Calculation of Performance metrics

## LONG & LAT
# Define function
def performance(predictions_long, predictions_lat, y_val_long, y_val_lat):
    
    longlat = {'Rsq': [r2_score(y_val_long, predictions_long), r2_score(y_val_lat, predictions_lat)], 
               'RMSE': [sqrt(mean_squared_error(y_val_long, predictions_long)), sqrt(mean_squared_error(y_val_lat, predictions_lat))], 
               'MAE': [mean_absolute_error(y_val_long, predictions_long), mean_absolute_error(y_val_lat, predictions_lat)], 
               'Emax': [max_error(y_val_long, predictions_long), max_error(y_val_lat, predictions_lat)]}
    
    perf_model = pd.DataFrame(longlat)
    return perf_model

# Calculate metrics - kNN
perf_kNN = performance(pred_long_kNN, pred_long_RF, y_val_long, y_val_long)
newcolumns = pd.DataFrame({'Var': ['long', 'long'], 'Alg': ['kNN', 'RF']})
perf_kNN = pd.concat([perf_kNN, newcolumns], axis = 1)

# Calculate metrics - RF
perf_LR = performance(pred_lat_kNN, pred_lat_RF, y_val_lat, y_val_lat)
newcolumns = pd.DataFrame({'Var': ['lat', 'lat'], 'Alg': ['kNN', 'RF']})
perf_LR = pd.concat([perf_LR, newcolumns], axis = 1)

perf_longlat = pd.concat([perf_kNN, perf_LR], axis = 0)

# Calculate accuracy - Floor
acc_kNN_floor = classification_report(y_val_floor, pred_floor_kNN, labels = [0, 1, 2, 3, 4])     
acc_RF_floor = classification_report(y_val_floor, pred_floor_RF, labels = [0, 1, 2, 3, 4])   

print(acc_kNN_floor, acc_RF_floor)

print(accuracy_score(y_val_floor, pred_floor_kNN), cohen_kappa_score(y_val_floor, pred_floor_kNN))
print(accuracy_score(y_val_floor, pred_floor_RF), cohen_kappa_score(y_val_floor, pred_floor_RF))



# =============================================================================
#### Preprocessing - Tests data
# =============================================================================

## Load test data
filename_test = os.path.abspath(os.path.join(DataDir, 'testData.csv'))
testData = pd.read_csv(filename_test)


## Transformation (1. Change RSSI)
testData.iloc[:, 0:520] = testData.iloc[:, 0:520] + 105
testData.iloc[:, 0:520] = testData.iloc[:, 0:520].replace(to_replace = 205, value = 0)

testData_unique = testData.loc[:, X_train_normObs.columns]


## Scaling - Normalize rows
scaler_normObs_TEST = Normalizer(norm = 'l2', copy = True)
X_TEST_normObs = scaler_normObs_TEST.fit_transform(testData_unique)
X_TEST_normObs = pd.DataFrame(X_TEST_normObs, index = testData_unique.index, columns = testData_unique.columns)



# =============================================================================
#### Prediction
# =============================================================================

## Predict building
pred_TESTbuild_kNN = modelkNN_build.predict(X_TEST_normObs)
pred_TESTbuild_RF = modelkNN_build.predict(X_TEST_normObs)


## Compare prediction of building and create new df with building
X_TEST_normObs_casckNN = pd.concat([X_TEST_normObs, pd.Series(pred_TESTbuild_kNN)], axis = 1)
X_TEST_normObs_casckNN.rename(columns={0:'BUILDINGID'}, inplace=True)

X_TEST_normObs_cascRF = pd.concat([X_TEST_normObs, pd.Series(pred_TESTbuild_RF)], axis = 1)
X_TEST_normObs_cascRF.rename(columns={0:'BUILDINGID'}, inplace=True)
# --> both predictions for building are the same

# X_TEST to be used to predict
X_TEST_normObs_casc = X_TEST_normObs_casckNN.copy()


## Predict LONG and LAT
pred_TESTlong_kNN = modelkNN_long.predict(X_TEST_normObs_casc)
pred_TESTlong_RF = modelRF_long.predict(X_TEST_normObs_casc)

pred_TESTlat_kNN = modelkNN_lat.predict(X_TEST_normObs_casc)
pred_TESTlat_RF  = modelRF_lat.predict(X_TEST_normObs_casc)

pred_TESTfloor_kNN = modelkNN_floor.predict(X_TEST_normObs_casc)
pred_TESTfloor_RF = modelRF_floor.predict(X_TEST_normObs_casc)


## Create submission dataset
Prediction_TEST_kNN = pd.concat([pd.Series(pred_TESTlat_kNN), pd.Series(pred_TESTlong_kNN), pd.Series(pred_TESTfloor_kNN)], axis = 1)

Prediction_TEST_kNN.columns = ['LATITUDE', 'LONGITUDE', 'FLOOR']


## Create 3D-Scatterplot of prediction
fig = px.scatter_3d(Prediction_TEST_kNN, x='LONGITUDE', y='LATITUDE', z='FLOOR',
                    size_max = 10, opacity = 0.7)
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
plot(fig)



# =============================================================================
#### Postprocessing - Retransformation of prediction
# =============================================================================

## 1. Return to original scale
#LONGITUDE
Prediction_TEST_kNN['LONGITUDE'] = Prediction_TEST_kNN['LONGITUDE'] - 2253200

#LATITUDE
Prediction_TEST_kNN['LATITUDE'] = Prediction_TEST_kNN['LATITUDE'] + 4311690


## 2. Transformation - Rotation of axis
# Set angel
theta = np.radians(-27.5)

# Define transformation matrix
tM = np.matrix([[np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]])
tM

# Select coordination columns
v_TEST = np.matrix(Prediction_TEST_kNN[['LONGITUDE', 'LATITUDE', 'FLOOR']])
v_TEST.shape
v_TEST = np.transpose(v_TEST)

# Rotation
tMv_TEST = tM * v_TEST

Prediction_TEST_kNN_It9_Output = pd.DataFrame(np.transpose(tMv_TEST), columns = ['LONGITUDE', 'LATITUDE', 'FLOOR'])
Prediction_TEST_kNN_It9_Output = pd.concat([Prediction_TEST_kNN_It9_Output['LATITUDE'], 
                                            Prediction_TEST_kNN_It9_Output['LONGITUDE'], 
                                            Prediction_TEST_kNN_It9_Output['FLOOR']], axis = 1)

    
## Create 3D-Scatterplot of final submission
fig2 = px.scatter_3d(Prediction_TEST_kNN_It9_Output, x='LONGITUDE', y='LATITUDE', z='FLOOR',
                     size_max = 10, opacity = 0.7)
fig2.update_layout(margin=dict(l=0, r=0, b=0, t=0))
plot(fig2)


## Export DataFrame to CSV
filename_export = os.path.abspath(os.path.join(SubmissionDir, 'Prediction_TEST_kNN_It9_Output_Paul.csv'))
Prediction_TEST_kNN_It9_Output.to_csv(filename_export, index = None, header = True)