# **Wifi indoor positioning system**  
  
Date: 07/10/2019  
  
## Project summary
**Project goal:** Developing a wifi indoor positioning system based on wifi fingerprinting and applied machine learning and investigate its performance in terms of positioning accuracy.  
  
**Data characteristics:** The data provided contains locational information and the corresponding wifi-fingerprints gathered in a multi-building and multi-floor setup on an university campus. The location is given by three variables, i.e. LONGITUDE and LATITUDE coordinates and the floor number defining the exact position of each location (reference points). Wifi-fingerprints, on the other hand, consist of 520 columns each representing a single 'Wifi access point' (WAP) and the signal intensity (RSSI) measured for a given observation.

The data is seperated into three datasets varying in the way of data collection and data provided. In detail these datasets contain the following data: 
* 'Train': Is considered to be used in model training phase. Contains almost 20k observations measured on predefined positions throughout the campus. In addition to the exact location and WAP signals measured columns are provided identifying the building, user, and phone ('BUILDINGID', 'USERID', 'PHONEID') are provided. 
* 'Validation': Is considered to validate the models trained with unseen observations. Covers over 1k obersations but provides no additional columns. Observations in this dataset have been taken randomly troughout the campus to simulate real-world data. Furthermore, it has to be meantioned that the validation set contains several obsevations taken in areas not covered by the trainset. 
* 'Test': This dataset is not available through training and validation phase as it is consider to test the best models identifyed before. Likewise to 'validation', the testset includes only real-world data to evaluate its positioning accuracy in a real-world setup.  
  
## Technical approach of analysis  
  
**1. Exploration and preparation of data** 
- Set time-zone, format, and intervals 
- Identify, pad, and fill NAs
- Aggregate and check different granularity
  
**2. Stationarity, autocorrelation, and transformation**    
- Create and analysis (multi-seasonal) decomposition
- Check stationarity and autocorrelation
- Ensure stationarity/conduct transformations
  
**3. (Multi-seasonal) Time series and data splitting**  
- Create time series and split data 
  
**4. Train models, evaluate, forecast**  
- Train several forecasting models (e.g. ARIMA, prophet)
- Perform cross-validation
- Evaluate models based on several performance measures such as RMSE, MAE, MAPE
- Forecast energy consumption
  
**5. Create visualization and dashboard**  
  
