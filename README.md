# Traffic Volume Prediction Using LSTM


## Table of Contents
1. [Introduction](#introduction)
2. [Data Description](#data-description)
3. [Data Preprocessing](#data-preprocessing)
4. [Model Building](#model-building)
5. [Model Evaluation](#model-evaluation)
6. [Conclusion](#conclusion)
7. [References](#references)

## Introduction
The dataset used in this project contains hourly traffic volume data for a city in the US. The data was collected from 2012 to 2018. The goal of this project is to predict the traffic volume for the next hour based on the historical data. The dataset contains the following columns:
- `holiday`: Categorical US National holidays plus regional holiday, Minnesota State Fair
- `temp`: Numeric Average temp in kelvin
- `rain_1h`: Numeric Amount in mm of rain that occurred in the hour
- `snow_1h`: Numeric Amount in mm of snow that occurred in the hour
- `clouds_all`: Numeric Percentage of cloud cover
- `weather_main`: Categorical Short textual description of the current weather
- `weather_description`: Categorical Longer textual description of the current weather
- `date_time`: DateTime Hour of the data collected in local CST time
- `traffic_volume`: Numeric Hourly I-94 ATR 301 reported westbound traffic volume

## Data Description

The dataset contains 48204 rows and 9 columns. The columns are as follows:
- `holiday`: 61 non-null object
- `temp`: 48204 non-null float64
- `rain_1h`: 48204 non-null float64
- `snow_1h`: 48204 non-null float64
- `clouds_all`: 48204 non-null int64
- `weather_main`: 48204 non-null object
- `weather_description`: 48204 non-null object
- `date_time`: 48204 non-null object
- `traffic_volume`: 48204 non-null int64

## Data Preprocessing

The following steps were performed to preprocess the data:
- Removed the columns `holiday`, `weather_main`, `weather_description`, and `date_time` as they are not required for the model.
- Converted the `date_time` column to a datetime object and set it as the index.
- Created a new column `hour` to store the hour of the day.
- Created a new column `dayofweek` to store the day of the week.
- Created a new column `month` to store the month of the year.
- Created a new column `year` to store the year.
- Created a new column `dayofyear` to store the day of the year.
- Created lag features for the `traffic_volume` column.
- Split the data into training and testing sets.

## Model Building

The model used in this project is an LSTM (Long Short-Term Memory) neural network. The model was built using the pytorch library.
In Detail: model.py

## Model Evaluation

The model was evaluated using the Mean Squared Error (MSE) metric.

## Conclusion


## References

https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume
https://www.kaggle.com/code/sayamkumar/metro-interstate-traffic-volume-prediction