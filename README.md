<img width="950" alt="earthquake img" src="https://user-images.githubusercontent.com/36497538/56088846-64309b80-5ea6-11e9-9092-c19c3062c495.png">

Predicting the time remaining before laboratory earthquakes occur from real-time seismic data.

### Description
Forecasting earthquakes is one of the most important problems in Earth science because of their
devastating consequences. Current scientific studies related to earthquake forecasting focus on
three key points: when the event will occur, where it will occur, and how large it will be.
The goal of the challenge is to capture the physical state of the laboratory fault and how close
it is from failure from a snapshot of the seismic data it is emitting. You will have to build a model
that predicts the time remaining before failure from a chunk of seismic data, like we have done in
our first paper above on easier data.

### Problem Statement:
To predict the time remaining before laboratory earthquakes occur from real-time seismic data.
Sources:
<br>
https://www.kaggle.com/c/LANL-Earthquake-Prediction
<br>
https://www.kaggle.com/c/LANL-Earthquake-Prediction/discussion

### Data
train.csv - A single, continuous training segment of experimental data.

### Data Overview
train.csv contains 2 columns: 
- acoustic_data - the seismic signal [int16] 
- time_to_failure - the time (in seconds) until the next laboratory earthquake [float64]
- Number of rows in Train.csv = 629145480

### Type of Machine Learning Problem
It is a Regression problem, for a given chunk of seismic data we need to predict the time remaining
before laboratory earthquakes occur

### Performance Metric 

Source: https://www.kaggle.com/c/LANL-Earthquake-Prediction#evaluation 
<br>
Metric(s): Mean Absolute Error

I have used several kernels from kaggle and ideas from discussion threads . 

<br>
https://www.kaggle.com/vettejeep/masters-final-project-model-lb-1-392 
<br>
https://www.kaggle.com/allunia/shaking-earth 
<br>
https://www.kaggle.com/gpreda/lanl-earthquake-eda-and-prediction
<br>
https://www.kaggle.com/c/LANL-Earthquake-Prediction/discussion
<br>
https://www.kaggle.com/c/LANL-Earthquake-Prediction/kernels

## Exploratory Data Analysis
1. It is given that the earthquake occurs when the time_to_failure hits 0, hence we can count that there are 16 occurences of earthquake in the whole training data

<p align="center">
  <img src="Images/eqp1.png" width="600" title="Train Distibution">
</p>

2. If we zoom into the data we can see that the acoustic data has a peak just before the earthquake occurs and the whole training data follows the same pattern..

<p align="center">
  <img src="Images/eqp2.png" width="600" title="Train Distibution">
</p>

3. If we plot the data for 1000000 points we can see that the graph is continously decreasing but if we zoom into it we can see that the time_to_failure stops decreasing for a while when it reaches ~4000 samples.
It is due to the fact that the data is recorded in bins of 4096 samples and the recording device stops for 12 microseconds after each bin.

<p align="center">
  <img src="Images/eqp6.png" width="600" title="">
</p>

## Featurization
Since the test data has 150000 samples in each segment, we convert the train data into samples of size 150000 and hence we get 4194 samples.
<br>
Since the datasize is too small, We split the 6.2m train data into 6 slices, take 4000 random samples,each of size 150000 from each slice. Hence now we have 24000 training data. 
It takes huge time to run therefore we use multiprocessing.

1> Statistical Features: Quantiles, mean, max/min, Sta/lta, std, skew, kurtosis, exponential moving averages, moving average, rolling std, rolling mean etc

2> Signal Processing features: Low pass, high pass fiters, FFT, peaks, entropy, hjorth parameters.

## Machine Learning Models

### LGBM

We use the default and approximate values for parameters since we came to know that cv is not reliable.
Light GBM, to its advantage, can handle the large size of data and takes lower memory to run. Training speed is much faster than other ensemble models and can also be interpretable. we get a score of 1.340. We can see below how well the model is predicting.

<p align="center">
  <img src="Images/eqp3.png" width="450" title="Train Distibution">
</p>

### XGBOOST
Since we want to maximize the score, xgboost has demonstrated successful for kaggle competitions. We were able to reach a mae of 1.314. We can see below that the model is not overfitting to train data and is generalising well.

<p align="center">
  <img src="Images/eqp4.png" width="450" title="Train Distibution">
</p>

### Stacking
Stacking models are very powerful and since interpretability is not important, we stack lgbm and xgboost model and use Linear Regression as meta regressor. We find that the mae is 1.379

## Feature Selection
SKlearn selectkbest: selectkbest selects top features and gives us the feature scores. we use top 300 features and apply LGBM and compare the results. We also tried out autoencoders, pearson correlation to reduce the dimensions but the performance dropped.
We can see that rolling features and peaks are the most important features.

<p align="center">
  <img src="Images/eqp5.png" width="450" title=" Distibution">
</p>

## Results
XGB gives the highest score of 1.314 which is currently at the 27th position at the kaggle public leaderboard.

<p align="center">
  <img src="Images/earthquake_prediction_results.png" width="700" title="Train ">
</p>


At the time of submission, the score was at top 1% of kaggle public leaderboard. As it is public lb the leaderboard positions might change.

<p align="center">
<img src="https://user-images.githubusercontent.com/36497538/58457650-d272aa00-8144-11e9-9e2c-43a6c61fdb85.PNG" width="700">
 </p>

