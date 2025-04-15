# Bachelor Thesis: Analysing the Foundation Models for Time-Series Forecasting

**University of Münster**  
Department of Information Systems  
Bachelor Thesis submitted by Chair for Data Science: Machine Learning and Data Engineering  
**Submission Date:** 21.01.2024
by Ellen Parthum

## Overview

This repository contains the code and datasets for my bachelor thesis titled *"Analysing the Foundation Models for Time-Series Forecasting"*. The work explores various foundation models, including TimesFM and TimeGPT-1, for forecasting time-series data and compares their performance in different experimental settings.

## Table of Contents

1. [Introduction](#introduction)
2. [Background](#background)
    - 2.1 Neural Networks
    - 2.2 Transformer Architecture
    - 2.3 Time-Series Foundation Models
3. [Related Works](#related-works)
4. [Methodology](#methodology)
5. [Experimental Results](#experimental-results)
    - 5.1 Dataset
    - 5.2 Evaluation Parameters
    - 5.3 Experimental Settings
    - 5.4 Fine-Tuning
6. [Discussion](#discussion)
7. [Conclusion](#conclusion)
8. [Appendix](#appendix)
    - A.1 Table of Sources
    - A.2 Table of Categorised Sources
    - A.3 Graphs for zero-shot
    - A.4 Graphs for fine-tuning
    - A.5 Graphs comparing MSE, MAE, and RMSE
    - A.6 Graphs for ARIMA
9. [Code](#code)
    - B.1 Code for TimesFM
    - B.2 Code for TimeGPT-1
    - B.3 Code for ARIMA
10. [Datasets](#datasets)
    - C.1 Dataset Code
    - C.2 Dataset Zero-shot
    - C.3 Dataset Fine-tuning

## Project Description

The goal of this research is to evaluate and compare the performance of different foundation models in the context of time-series forecasting. Specifically, it investigates the *TimesFM* and *TimeGPT-1* models, focusing on their ability to forecast time-series data with and without covariates, and their performance when fine-tuned for specific datasets.

The thesis includes an analysis of various machine learning methods, such as neural networks and transformers, and applies them to real-world time-series data. Key metrics such as MSE, MAE, and RMSE are used to evaluate model performance.

## Prerequisites

Before running the code, you’ll need to have the following dependencies installed:

- Python 3.x
- TensorFlow
- PyTorch
- Scikit-learn
- Matplotlib
- NumPy
- Pandas

## Installing TimesFM and TimeGPT
Instructions on how to install TimesFM and TimeGPT-1 can be found here:

TimesFM
https://github.com/google-research/timesfm

TimeGPT-1
https://github.com/Nixtla/nixtla
