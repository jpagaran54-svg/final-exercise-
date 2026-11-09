# Wine Quality Prediction System

## Project Description
This project predicts wine quality using a Random Forest
Machine Learning classification model.

## Problem
A wine producer needs a preliminary system to screen
wine samples using physicochemical measurements.

## Users
- Quality Control Staff
- Production Supervisor
- Data Analyst

## Dataset
UCI Wine Quality Dataset - Red Wine.

https://archive.ics.uci.edu/dataset/186/wine+quality

## Prediction Rule
- Quality >= 6: GOOD QUALITY
- Quality < 6: NEEDS IMPROVEMENT

## Technologies
- Python
- Pandas
- Scikit-learn
- Random Forest
- Streamlit

## Installation

pip install -r requirements.txt

## Run Model

python train_model.py

## Run Dashboard

streamlit run app.py

## Limitations
The model is trained using a particular red-wine dataset.
It may not generalize to other wine producers, regions,
wine types, or testing conditions.

Predictions are preliminary screening results only.
