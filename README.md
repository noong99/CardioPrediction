# CardioPrediction: A Machine Learning Approach to Cardiovascular Disease Prediction
R Project

### Overview
CardioPredict is a data analysis and machine learning project designed to predict the risk of cardiovascular disease.
The project utilizes a dataset of health-related features to train and evaluate predictive models, aiming to identify individuals at high risk for cardiovascular conditions

### Features
- Data Exploration:
  Loaded and explored the dataset to understand its structure, summary statistics, and check for missing values.
- Modeling:
  Built and evaluated predictive models using decision trees and cross-validation techniques.
  Focused on variables like gender, cholesterol, glucose levels, smoking, alcohol use, and physical activity.
- Visualization:
  Created insightful plots using ggplot2 for better data understanding and model performance analysis.

### Used
- R Programming Language: For data manipulation, visualization, and modeling.
- tidymodels: For streamlined machine learning workflows.
- dplyr and ggplot2: For data wrangling and visualization.
- readr: For efficient data loading.

### Dataset
- Source:
  - cardiovascular_disease_dataset.csv
- Features:
  - Health metrics (e.g., cholesterol, glucose, physical activity)
  - Behavioral factors (e.g., smoking, alcohol use)
  - Target variable: Presence or absence of cardiovascular disease.

### Steps
1. Data Preparation:
 - Loaded the dataset and verified data integrity (e.g., no missing values).
 - Identified key features and ensured they were free from outliers.
2. Modeling:
 - Implemented a decision tree model.
 - Used cross-validation for reliable model evaluation.
 - Decision Tree, RandomForest
3. Evaluation:
 - Assessed model performance using metrics like accuracy and precision.
4. Visualization:
 - Created plots to analyze variable importance and model performance.

### Future Enhancements
- Explore feature engineering techniques to improve prediction accuracy.
- Integrate additional datasets to expand the scope of analysis.
