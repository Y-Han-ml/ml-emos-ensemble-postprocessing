# Machine Learning-Based EMOS for Postprocessing Ensemble rainfall forecasts

This repo contains research code developed during my MSc dissertation in Applied Meteorology and Climatology at the university of Birmingham.

The project is strongly based on the ongoing programmee named "HEPPL-ML", which is a major Government-University (UK Met office & University of Birmingham) collaborative program. 

The project focues on applying MLP-based EMOS with different structures and hyperparameter settings to reduce forecast bias existing in NEPS-G system(A numerical ensemble forecast system with 23-members in India).

## Background 
The skills of Numerical Weather prediction is often limited by computional constraints and growth of initial value uncertainty a chaotic system, particularly in India with the affects of the Monsoon Seasons. Traditional postprocessing method EMOS is statistical, while this work explores ML-EMOS to capture the non-linear relationship between observations and forecasts with potential addtional input weather regimes showing the different monsoon patterns in India.

## Methods
- Multi-layer Perceptron (MLP)-based EMOS
- Validation metrics using:
  - Rank histograms
  - Continuous Ranked Probability Skill Score (CRPSS)
  - Brier Skill Score (BSS)

## Notes
Due to data sharing restrictions, raw datasets are not included. There are two versions: Simplified ML-EMOS (with virtual data samples) and Detailed ML-EMOS (No datasets but complete codes and settings)

## Author
Yuteng Han
MSc Applied Meteorology and Climatology
University of Birmingham
