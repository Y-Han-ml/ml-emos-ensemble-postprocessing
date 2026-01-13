# ML-EMOS workflow Demo

This shows the workflow of a machine learning-based EMOS for postprocessing ensemble rainfall forecasts, based on my Msc dissertation at the University of Birmingham

## Data Structure (Illustrative)

- Ensemble forecasts: 23 NEPS-G ensemble members during the 2018-2022 monsoon seasons (Jun-Sep) with a lead time of 1 day.
- observations: 0.25 * 0.25 degrees gridded datasets for 2018-2022
- weather regimes: daily frequency of 7 weather regimes to strength forecast accuarcy

## Methodology

This research employs Multilayer percetron (MLP) to postprocess NEPS-G rainfall forecasts at nine locations representing different geographical regions across India.
The model transitions from a baseline configuration of 23 raw ensemble inputs to a 30-variable input incorporating 7 weather regimes to evaluate whether capturing atmospheric patterns can enhance forecast accuracy.
ML-EMOS implements CRPS as the loss funtion for running the backwards process to update weighs and parameters in MLP.

## Evaluation 

- Frequency distribution: The
- Rank histogram: used for reliability assessment of ensemble forecasts
- Continuous Ranked Probability Skill Score (CRPSS): overall forecast performance of ensemble members
- Brier Skill Score (BSS): performance of predicting extreme precipitation events

To evaluate a machine learning model, it is essential to ensure trainning stability and prevent overfitting.
Therefore, this research starting from a single layer structure to validate model before scaling to more complex architectures and hyperparameter optimizations.   
