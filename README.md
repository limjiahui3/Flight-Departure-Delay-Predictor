# Flight Departure Delay Predictor

## Overview

**Project Goal:** Develop a supervised classification model to predict whether a flight will be delayed (delay >= 15 minutes) based on weather and flight-related features. Explore both LSTM and XGBoost models to evaluate performance across temporal and tabular data.

 **Models:** LSTM (Long Short-Term Memory), XGBoost

**Evaluation Metrics:** F1-score

## Data Preprocessing

**Weather Data Source:** https://observablehq.com/@observablehq/noaa-weather-data-by-major-u-s-city

**Flight Data Source:** https://www.transtats.bts.gov/ontime/Departures.aspx

**Selected Data:** Delta Air Lines Inc. (DL) flights departing from John F. Kennedy International Airport (JFK) from 2016 to 2021

**Key Data Preprocessing (Weather Data):**
- Date format standardisation
- Feature selection
- Handling missing values via linear interpolation
- Extraction of time-based features using cyclical encoding (sine and cosine transformations)

**Key Data Preprocessing (Train and Test Data):**
- Train–test split
  - Split flight data chronologically
  - Combined a random sample of 20,000 non-delayed flights with all 26,029 delayed flights to maintain a more balanced training set
- Feature scaling
  - Fit the scaler only on the training data to avoid data leakage
  - Applied the fitted scaler to the entire dataset for consistency
- Categorical encoding
- Weather window extraction

## Key Evaluation Steps & Insights

- Threshold tuning for F1-score

**Model Results:**
- LSTM model (5-hour weather sequences) and XGBoost model (same sequences flattened into fixed-length vectors) both achieved ~0.32 F1-score
- Suggests temporal dependencies within the 5-hour weather window may not be significant, or models failed to capture them effectively

**Observations:**
- Low F1-score indicates need for more informative features
- Basic weather and flight features alone may be insufficient to model flight delay causation

**Potential improvements:**
- Incorporate additional flight-level and operational features (e.g., airport congestion, airline schedules, prior flight delays)
- Further tune model architectures and preprocessing strategies

## Files
- `Flight_Departure_Delay_Predictor.ipynb`: Data preprocessing, model training, tuning, and evaluation
- `2016-2021_flight.csv`: Raw flight data
- `2016-2021_weather.csv`: Raw weather data
- `2016-2021_weather_preprocessed.csv`: Preprocessed weather data
- `README.md`: Project overview
