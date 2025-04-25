# Crop Recommendation System.

## Project overview.

This project aims to develop a machine learning-based crop recommendation system that suggests the most suitable crops based on soil and environmental conditions. Using a dataset containing essential agricultural parameters — including nitrogen, phosphorus, potassium, soil pH, temperature, humidity, and rainfall — the system predicts which crops are most likely to thrive in specific conditions.
By leveraging various machine learning techniques, the project provides a data-driven approach to optimize agricultural productivity and reduce potential losses.

**[Click here to try the app live!](https://crop-soil-prediction-jgcnjn3mjl4lhuj7u4jvgm.streamlit.app/)**

## Methodology
Exploratory Data Analysis (EDA) was performed to understand distributions, correlations, and detect any outliers in the dataset.

Multiple classification models were trained:

Baseline models such as Decision Tree Classifier.

Ensemble models like the Random Forest, which performed well without signs of overfitting or underfitting.

Cross-validation was used to assess generalization and performance on unseen data.

Additionally, a deep neural network was implemented to capture complex, non-linear relationships between the features and target labels. The model included:

Multiple dense layers with ReLU activation.

Early stopping based on validation loss, with a patience of 5 epochs.

A final softmax output layer for multi-class classification.

The deep learning model showed strong performance — validation accuracy improved consistently across epochs, while loss decreased, indicating effective learning and generalization.

This project demonstrates how ML and deep learning can be used to recommend crops tailored to specific environmental conditions. Such systems have the potential to assist farmers, agronomists, and policymakers in making informed decisions to boost yield and sustainability.