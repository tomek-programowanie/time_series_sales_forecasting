# Time Series Sales Forecasting

This repository contains code and data for forecasting product sales using various models, including a baseline model, RandomForestRegressor, and SARIMAX. The primary objective is to provide accurate sales predictions to help plan stock levels.

## Table of Contents

- [Task Description](#task-description)
- [Datasets](#datasets)
- [Import Libraries](#import-libraries)
- [Data Collection and Preprocessing](#data-collection-and-preprocessing)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Feature Engineering](#feature-engineering)
- [Model Training and Evaluation](#model-training-and-evaluation)
  - [Baseline Model](#baseline-model)
  - [RandomForestRegressor](#randomforestregressor)
  - [SARIMAX](#sarimax)
- [Results](#results)
- [Summary of Results](#summary-of-results)
- [Conclusions](#conclusions)
- [Limitations](#limitations)

## Task Description

The goal of this project is to predict product sales to assist in stock level planning.

## Datasets

The following datasets are used in this project:

- **sales.csv**: Contains product-level weekly sales data.
  - `week_starting_date`: First day of the week (YYYYMMDD format)
  - `product_id`: Unique product identifier
  - `sales`: Weekly sales in pieces

- **categories.csv**: Contains product category assignments.
  - `product_id`: Unique product identifier
  - `category_id`: Unique category identifier

- **traffic.csv**: Contains weekly product display data on the website.
  - `week_starting_date`: First day of the week (YYYYMMDD format)
  - `product_id`: Unique product identifier
  - `traffic`: Weekly product displays on the website

## Import Libraries

The necessary libraries for data manipulation, visualization, and modeling are imported.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import random

from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
```

## Data Collection and Preprocessing

1. **Load datasets**:
   - `sales.csv`, `categories.csv`, `traffic.csv` are loaded into Pandas DataFrames.

2. **Data Cleaning and Merging**:
   - Convert date columns to datetime format.
   - Merge datasets on `product_id` and `week_starting_date`.
   - Handle missing values by filling them with 0.

## Exploratory Data Analysis (EDA)

1. **Visualizations**:
   - Plot sales and traffic over time.
   - Scatter plot to examine the correlation between sales and traffic.
   - Sales per category over time.
   - Distribution of the number of weeks of data available per product.

2. **Insights**:
   - Product data is not uniformly available. Only products with at least 2 years of data are considered.

## Feature Engineering

1. **Date Range Filtering**:
   - Filter products with a minimum of 2 years of data.

2. **Create Features**:
   - Generate lag features, rolling statistics, and decomposed time series components (trend, seasonal, residual).
   - Extract date features (day of week, week of year, month).
   - Create target variables for 1, 2, and 3 weeks ahead forecasts.

## Model Training and Evaluation

### Baseline Model

1. **Preparation**:
   - Shift sales by one week to create the `last_week_sales` feature.

2. **Evaluation**:
   - Calculate MAE, MSE, and WAPE for the baseline model.

### RandomForestRegressor

1. **Training**:
   - Train a RandomForestRegressor for 1, 2, and 3 weeks ahead forecasts.

2. **Evaluation**:
   - Evaluate the model using MAE, MSE, and WAPE metrics.

### SARIMAX

1. **Training**:
   - Train a SARIMAX model for 1, 2, and 3 weeks ahead forecasts using normalized features.

2. **Evaluation**:
   - Evaluate the model using MAE, MSE, and WAPE metrics.

## Results

The performance of each model is compared using bar plots for MAE, MSE, and WAPE metrics.

## Summary of Results

- **RandomForestRegressor** and **SARIMAX** models achieved better results compared to the baseline model.
- **SARIMAX** showed lower average forecast errors for most products, especially for 3-week forecasts.

## Conclusions

- Advanced models like **RandomForestRegressor** and **SARIMAX** are more effective for sales forecasting than simple baseline approaches.
- **SARIMAX** is particularly suitable for long-term forecasts, despite its complexity.

## Limitations

1. Models were trained only on products with at least 104 weeks of data.
2. A random sample of 150 products was used for training due to computational constraints, which may affect results.
