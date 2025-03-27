# Time Seriess Forecasting 🌦

This lab lecture explores how to use a weather time series dataset from the Max Planck Institute for Biogeochemistry to solve forecasting problems.
The dataset spans from 2009 to 2016 and includes features such as ambient temperature, air pressure, and other environmental variables, recorded at a 10-minute frequency.
The goal is to predict future weather conditions based on historical data.

The first step involves preprocessing the dataset, which includes handling missing or null values, resampling the data from a 10-minute frequency to a 1-hour frequency, normalizing the features for consistent scaling,
and addressing any anomalies that may exist in the data.

Once the data is preprocessed, various deep learning models are applied to forecast weather.
Initially, the lab demonstrates how to predict a single timestamp, such as temperature, using historical data as input.
Models like Long Short-Term Memory (LSTM), Conv1D (1D Convolutional Neural Networks), and Multi-Layer Perceptron (MLP) are used for this task.
The lab then extends these models to handle multiple feature outputs, meaning all available features are predicted simultaneously.

Finally, the models are modified to perform multi-timestamp forecasting, where, instead of predicting just one future value, they can forecast weather conditions for several future timestamps.
For example, given data from the past 24 hours, the model can predict weather for the next 24 hours.

## Usage 📝

The `.ipynb` notebook it's ready for start, you just need to run it using your virtualenv or just upload it on
[Google Colab](https://colab.research.google.com/).
