# **Wifi indoor positioning system**  
  
Date: 07/10/2019  
  
## Project summary
**Project goal:** Developing a wifi indoor positioning system based on wifi fingerprinting and applied machine learning and investigate its performance in terms of positioning accuracy.  
  
**Data characteristics:** The data provided contains locational information and the corresponding wifi-fingerprints gathered in a multi-building and multi-floor setup on an university campus. The location to be predicted is given by three variables, i.e. LONGITUDE and LATITUDE coordinates and the floor number defining the exact position of each location (reference points). Wifi-fingerprints, on the other hand, consist of 520 columns each representing a single 'Wifi access point' (WAP) and the signal intensity (RSSI) measured for a given observation.

The data is separated into three datasets varying in the way of data collection and data provided. In detail these datasets contain the following data: 
* 'Train': Is used in model training phase. Contains almost 20k observations measured on predefined positions throughout the campus. In addition to the exact location and WAP signals measured columns are provided identifying the building, user, and phone ('BUILDINGID', 'USERID', 'PHONEID') are provided. 
* 'Validation': Is considered to validate the models trained with unseen observations. Covers over 1k observations but provides no additional columns. Observations in this dataset have been taken randomly throughout the campus to simulate real-world data. Furthermore, it has to be mentioned that the validation set contains several observations taken in areas not covered by the trainset. 
* 'Test': This dataset is not available through training and validation phase since it is considered to test the best models identified before. Likewise, the testset includes only real-world data to evaluate its positioning accuracy in a real-world setup.  
  
## Technical approach
The project follows an iterative approach to understand the data and select the best performing model using different pre-processing techniques, algorithms, and trainset configurations. 
  
**1. Exploration and preparation of data** 
- Visualize distribution of reference points throughout the campus
- Visualize user and phone distribution and behaviour
- Rotate coordinates and change scales of LONGITUDE and LATITUDE
- Change negative RSSI scale to positive and set value for 'not detected' to zero
- Check and exclude duplicates and empty rows (Observations with zero WAP detection)
  
**2. Pre-processing**    
- Apply and test different scaling methods: Normalization, Standardization, etc.
- Develop and test different approaches to get rid of high spread of wifi-signals and anomalies in phone detection behaviour
- Address possible influences caused by different users
  
**3. Modelling**  
- Train different models using a diversity of appropriate algorithms (e.g. kNN, RF, XGBT)
- Perform cross-validation and bootstrap aggregation (bagging)
- Hyperparameter tuning
  
**4. Evaluation of approaches and models**  
- Evaluate models employing respective performance measures for regression and classification (e.g. RMSE, MAE, Accuracy, Kappa)
- Conduct error analysis (distribution of errors, error intervals, etc.)
- Identify and address weak-spots of each model
  
**5. Applying selected models on 'Test'-set for submission**  
  
## Overview of selected iterations

- 'Baseline' model: MVP model without pre-processing and hyperparameter tuning
- 'WAPs selection': Only those WAPs (columns) used that are at least once detected in train and validation set
- 'Scaling': Testing different scaling functions --> (Vector-) normalization by row addressing different detection behaviour of phones and possible user influence
- 'Cascade': Using certain predicted variables (e.g. BUILDINGID which is not available in test) for prediction of remaining variables
- 'WAP-Profiling': Identify distinctive WAP-profiles (e.g. 3 WAPS with highest signal intensity) for each reference point 
- 'Enrichment': Transfer those positions from validation to train that are not covered within train
- 'Hyperparameter': Tune hyperparameter of final model selection
