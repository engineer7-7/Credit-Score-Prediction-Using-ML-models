# Machine Learning Project: KNN and SVM Model Evaluation

## Overview
This project focuses on the implementation and evaluation of two machine learning models:
- **K-Nearest Neighbors (KNN)**
- **Support Vector Machine (SVM)**

The goal is to compare the performance of these models using metrics such as accuracy, recall, precision, and F1-score. The dataset used contains multiple categorical and numerical features, and the target variable consists of three classes: *Poor*, *Standard*, and *Good*.

## Models Used
- **KNN (K-Nearest Neighbors)**: A classification model that assigns a class based on the majority vote of its k-nearest neighbors.
- **SVM (Support Vector Machine)**: A classification model that finds the optimal hyperplane which separates the data into different classes.

## Dataset
- **Target Variable**: Credit Score (`Poor`, `Standard`, `Good`)
- **Features**: Multiple numerical and categorical features, including age, loan type, etc.
- **Data Preprocessing**:
  - Label encoding was used to convert categorical features into numerical values.
  - Data scaling (StandardScaler) was applied to ensure that all features are on the same scale.

## Performance Evaluation
We evaluated the models using the following metrics:
- **Accuracy**: Proportion of correct predictions.
- **Recall**: Ability of the model to capture all the true positives.
- **Precision**: Proportion of correct positive predictions.
- **F1-score**: Harmonic mean of precision and recall.

### KNN Performance:
- **Recall**: 0.7572
- **Precision**: 0.7607
- **F1-score**: 0.7584

### SVM Performance:
- **Recall**: 0.6358
- **Precision**: 0.6882
- **F1-score**: 0.6550

## Getting Started

### Prerequisites
You will need the following Python libraries to run the project:
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`

Install the necessary libraries using pip:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
