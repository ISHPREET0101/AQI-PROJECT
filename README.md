AQI Prediction and Pollution Analysis
This project utilizes machine learning to analyze environmental pollutants and predict the Air Quality Index (AQI).
By evaluating features like PM2.5, PM10, NO2, and CO, the project identifies key drivers of air pollution and compares various regression
and classification models to achieve high accuracy.
## Project Overview
The goal of this study is two-fold
:Regression: Predicting the exact numerical value of the AQI.
Classification: Categorizing air quality into levels such as "Good," "Moderate," "Unhealthy," etc.
## Key Insights from Data
Feature Importance: PM2.5 is the most significant predictor of air quality, followed by Carbon Monoxide (CO) and PM10.
Correlation: There is a very strong positive correlation (0.86) between PM2.5 and PM10, indicating they often rise and fall together.
Temporal Trends: Interestingly, the "Year" feature shows a slight negative correlation with AQI, suggesting a potential long-term shift in recorded levels.
## Model Performance
### Regression AnalysisWe compared four different models using $R^2$ scores via Cross-Validation.
Random Forest and Gradient Boosting emerged as the top performers, significantly outperforming simple Linear Regression.
ModelAverage R2 ScoreRandom Forest~0.92Gradient Boosting~0.91Linear Regression~0.88Decision Tree~0.85
### Classification ResultsThe classification model shows high precision, especially for the "Unhealthy" and "Very Unhealthy" categories.
High True Positives: The model correctly identified 1,079 "Unhealthy" instances.
Common Confusion: The model occasionally confuses "Good" with "Unhealthy," which suggests some overlapping feature ranges in those specific data points.
## Visualizations 
Below are the key analytical plots generated during this project.
1. Correlation HeatmapIdentifies the linear relationship between pollutants.<img width="1000" height="800" alt="correlation_heatmap" src="https://github.com/user-attachments/assets/ba18c3c9-08e3-40a9-b9bd-d0b1a013ce81" />
2. Feature ImportanceRanking the impact of each pollutant on the final AQI prediction.<img width="800" height="500" alt="feature_importance" src="https://github.com/user-attachments/assets/b6bf32c2-9efb-45f7-96b2-afde24e17bfb" />
3. Confusion MatrixEvaluates the classification accuracy across different AQI categories.<img width="600" height="500" alt="confusion_matrix" src="https://github.com/user-attachments/assets/2806480d-481d-4b6d-8f58-45770dc07e20" />
## Tech StackLanguage: Python
Libraries: Pandas, Scikit-learn, Matplotlib, Seaborn
Models: Random Forest, Gradient Boosting, Decision Trees, Linear Regression
