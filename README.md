# Earthquake-Prediction

<img width="950" alt="earthquake img" src="https://user-images.githubusercontent.com/36497538/56088846-64309b80-5ea6-11e9-9092-c19c3062c495.png">

Predicting the time remaining before laboratory earthquakes occur from real-time seismic data.

### 1.1 Description
Forecasting earthquakes is one of the most important problems in Earth science because of their
devastating consequences. Current scientific studies related to earthquake forecasting focus on
three key points: when the event will occur, where it will occur, and how large it will be.
The goal of the challenge is to capture the physical state of the laboratory fault and how close
it is from failure from a snapshot of the seismic data it is emitting. You will have to build a model
that predicts the time remaining before failure from a chunk of seismic data, like we have done in
our first paper above on easier data.

### 1.2 Problem Statement:
To predict the time remaining before laboratory earthquakes occur from real-time seismic data.
Sources:
<br>
https://www.kaggle.com/c/LANL-Earthquake-Prediction
<br>
https://www.kaggle.com/c/LANL-Earthquake-Prediction/discussion

### 2 Data
train.csv - A single, continuous training segment of experimental data.

### 2.1.1 Data Overview
train.csv contains 2 columns: 
- acoustic_data - the seismic signal [int16] 
- time_to_failure - the time (in seconds) until the next laboratory earthquake [float64]
- Number of rows in Train.csv = 629145480

### 2.2.1 Type of Machine Learning Problem
It is a Regression problem, for a given chunk of seismic data we need to predict the time remaining
before laboratory earthquakes occur

### 2.2.2 Performance Metric 

Source: https://www.kaggle.com/c/LANL-Earthquake-Prediction#evaluation 
<br>
Metric(s): Mean Absolute Error

At the time of submission, the score was at top 1% of kaggle public leaderboard. As it is public lb the leaderboard positions might change.

![latest__score (2)](https://user-images.githubusercontent.com/36497538/58366132-82e57180-7eeb-11e9-929a-9aee90beaf57.PNG)

