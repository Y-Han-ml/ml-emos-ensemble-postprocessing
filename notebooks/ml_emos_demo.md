# ML-EMOS workflow Demo

This shows the workflow of a machine learning-based EMOS for postprocessing ensemble rainfall forecasts, based on my Msc dissertation at the University of Birmingham

## Problem statement

While numerical forecast systems like NEPS-G are powerful, but they often struggle with systematic biases and dispersion errors. The raw ensembles need to be postprocessed for adapting to real-world applications like flood assessment.

## Data Structure (Illustrative)

- Ensemble forecasts: 23 NEPS-G ensemble members during the 2018-2022 monsoon seasons (Jun-Sep) with a lead time of 1 day.
- observations: 0.25 * 0.25 degrees gridded datasets for 2018-2022
- weather regimes: daily frequency of 7 weather regimes to strength forecast accuarcy

## Methodology

This research employs Multilayer percetron (MLP) to postprocess NEPS-G rainfall forecasts at nine locations representing different geographical regions across India.

The model transitions from a baseline configuration of 23 raw ensemble inputs to a 30-variable input incorporating 7 weather regimes to evaluate whether capturing atmospheric patterns can enhance forecast accuracy.

ML-EMOS implements CRPS as the loss funtion for running the backwards process to update weighs and parameters in MLP.

## Setup

Data analytic tools like numpy, pandas, especially Xarray, are used to handle large meteorological datasets.

Deep learning framework Pytorch is used for this work.

## Data engineering

How to convert traditional .nc format data into Tensors that can be processed by neural networks. We need to handle two key logical steps here:

Spatio-temporal alignment: Ensure that the forecast dates correspond one-to-one with the observation dates.
Feature preprocessing: Sort the 23 ensemble members. In meteorological statistics, sorting the ensemble members can significantly improve the model's ability to learn the tails of the distribution.

## Model Structure

Our model is not merely a black box. To align with meteorological physical common sense, we incorporated a key design into the network:

Linear mapping:  Y = W X + b, which learns the weighted coefficients between ensemble members.

ReLU activation function: This serves as our physical constraint. Since precipitation cannot be negative, ReLU ensures that the model output is always greater than or equal to zero.

## Five-folds cross-validaiton

Thinking: Meteorological data exhibit strong seasonality and interannual variability. To thoroughly validate the model's generalization capability, we employ a Leave-One-Year-Out cross-validation strategy:

We conduct 5 iterations in total.
In each iteration, one year is held out as the validation set, while the remaining four years serve as the training set.
This approach allows us to assess how well the model performs on "unseen anomalous weather years" that it has never encountered during training.

## Evaluation 

- Frequency distribution: The
- Rank histogram: used for reliability assessment of ensemble forecasts
- Continuous Ranked Probability Skill Score (CRPSS): overall forecast performance of ensemble members (Thinking: Traditional MSE (Mean Squared Error) is only suitable for evaluating deterministic forecasts. In probabilistic forecasting, what we pursue is the "reliability" and "sharpness" of the predictions.); In this work, CRPS was also used as the loss function for training (via backpropagation).
- Brier Skill Score (BSS): performance of predicting extreme precipitation events

To evaluate a machine learning model, it is essential to ensure trainning stability and prevent overfitting.
Therefore, this research starting from a single layer structure to validate model before scaling to more complex architectures and hyperparameter optimizations.   

This study compared the performance of machine learning models with and without incorporating weather regimes.

## Fine tuning

In this study, we explored MLP architectures with different depths and tuned various hyperparameters, including the learning rate and number of epochs, to obtain the optimized model.
