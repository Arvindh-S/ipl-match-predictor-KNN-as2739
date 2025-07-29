# IPL Match Outcome Predictor

## Project Overview

This project develops a machine learning-powered web application to predict the outcome of Indian Premier League (IPL) cricket matches. Leveraging historical match data, player statistics, and real-time match conditions, the application aims to provide accurate predictions for match results (Win/Loss).

The core of this project involves:
- **Data Collection & Preprocessing**: Gathering and cleaning extensive IPL match, delivery, batting, and bowling statistics.
- **Feature Engineering**: Creating insightful features from raw data to capture critical match dynamics (e.g., current run rate, required run rate, team strengths).
- **Model Development**: Implementing and evaluating two supervised machine learning models: K-Nearest Neighbors (KNN) and Logistic Regression.
- **Model Optimization**: Applying hyperparameter tuning (using GridSearchCV) to enhance model performance and generalization.
- **Web Application**: Building a user-friendly interface using Streamlit for interactive predictions.
- **Deployment**: Hosting the application for public access.