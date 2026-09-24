# Employee Attrition Prediction — End-to-End ML Project

An end-to-end machine learning project that predicts whether an employee is likely to leave a company using the IBM HR Analytics Employee Attrition dataset.

## Problem Statement

Employee attrition can affect productivity, hiring costs, and workforce planning. This project builds a binary classification model to estimate the probability of employee attrition.

## Dataset

- Dataset: IBM HR Analytics Employee Attrition & Performance
- Records: 1,470 employees
- Features: 35
- Target: Attrition
- Attrition rate: approximately 16.1%

## Project Workflow

1. Data loading and inspection
2. Data cleaning
3. Exploratory Data Analysis
4. Feature engineering
5. Data preprocessing
6. Model comparison
7. Hyperparameter tuning
8. Decision-threshold tuning
9. Final model evaluation
10. Model saving
11. Flask prediction application

## Feature Engineering

The project creates additional features including:

- IncomePerJobLevel
- TenureRatio
- AvgYearsPerCompany
- PromotionGap
- OverallSatisfaction

## Machine Learning Models

The following classification models were compared:

- Logistic Regression
- Random Forest
- Gradient Boosting

Five-fold stratified cross-validation was used for model comparison.

## Model Results

Logistic Regression achieved the highest cross-validation ROC-AUC among the tested models and was selected for hyperparameter tuning.

- Best parameter: C = 0.1
- Cross-validation ROC-AUC: 0.834
- Decision threshold: 0.54
- Test accuracy: 0.81
- Test ROC-AUC: 0.805
- Attrition precision: 0.43
- Attrition recall: 0.64
- Attrition F1-score: 0.52

## Application

The trained model is saved using Joblib and can be used through a Flask web application.

Project structure:

```text
attrition/
├── app.py
├── data/
│   └── employee_attrition.csv
├── models/
│   ├── attrition_model.joblib
│   └── meta.json
├── reports/
├── src/
│   └── train.py
├── templates/
│   └── index.html
├── requirements.txt
└── README.md
```

## Key Findings

Exploratory analysis showed differences in observed attrition across factors such as overtime, job role, business travel, marital status, and employee characteristics.

These findings describe patterns in the dataset and should not be interpreted as causal relationships.

## Limitations

- The dataset is a historical HR analytics dataset.
- The model's performance may differ on real company data.
- Model predictions represent estimated risk, not certainty.
- Additional validation would be required before using such a system for real HR decisions.

## Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Flask
- Joblib
- Google Colab

## Author

Vikas Kumar

BTech Computer Science Engineering — AI/ML