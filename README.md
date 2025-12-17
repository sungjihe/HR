# HR Attrition Analysis

## Overview
This project analyzes employee attrition using supervised machine learning.
Logistic Regression with L2 regularization and Random Forest models were compared
to evaluate linear vs non-linear decision boundaries.

## Modeling
- Logistic Regression (Ridge)
- Random Forest Classifier
- Stratified train-test split
- Evaluation: Precision, Recall, F1-score, ROC-AUC

## Key Findings
- Logistic Regression achieved ROC-AUC 0.96 with stable performance.
- Random Forest improved recall and F1-score for the attrition class.
- Model selection depends on interpretability vs predictive performance.

## Files
- `hr_dataset.py`: data preprocessing, modeling, and evaluation
