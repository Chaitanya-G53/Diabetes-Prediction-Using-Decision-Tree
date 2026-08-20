<div align="center">

🩺 Diabetes Classification Using Decision Tree

A machine learning classification project for predicting diabetes outcomes from demographic, lifestyle, and clinical features.

<p>
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-learn">
  <img src="https://img.shields.io/badge/Model-Decision%20Tree-2E7D32?style=for-the-badge" alt="Decision Tree">
  <img src="https://img.shields.io/badge/Status-Completed-2E7D32?style=for-the-badge" alt="Completed">
</p>

<p>
  <b>🎯 Task:</b> Binary Classification &nbsp;•&nbsp;
  <b>🌳 Algorithm:</b> Decision Tree &nbsp;•&nbsp;
  <b>📊 Test Accuracy:</b> 97%
</p>

</div>

📌 Overview

This project implements an end-to-end supervised machine learning classification workflow to predict the diabetes outcome from demographic, lifestyle, and clinical features.

The project focuses on building an interpretable Decision Tree Classifier, handling class imbalance, performing exploratory data analysis, tuning model hyperparameters with cross-validation, evaluating the final model, and analyzing feature importance.

⚠️ Educational project only: This model is not clinically validated and must not be used as a medical diagnostic or decision-making system.

🚀 Project Highlights

Area

Implementation

📥 Data

Diabetes Prediction Dataset

🧹 Data Cleaning

Duplicate and missing-value checks

📈 EDA

Distribution, comparison, and correlation analysis

🔤 Encoding

One-hot encoding for categorical features

✂️ Data Split

80/20 stratified train-test split

🌳 Model

Decision Tree Classifier

🔧 Tuning

GridSearchCV with 5-fold cross-validation

⚖️ Imbalance

Class-weight experimentation

📊 Evaluation

Accuracy, Precision, Recall, F1-score

🔍 Interpretability

Decision Tree feature importance

💾 Deployment Artifact

Pickle model

📂 Dataset

The project uses the Diabetes Prediction Dataset downloaded through KaggleHub.

Dataset identifier used in the notebook:

mariamelghareeb/diabeties-prediction

Original file:

diabetes_prediction_dataset.csv

Features

Feature

Description

Type

gender

Gender category

Categorical

age

Age

Numerical

hypertension

Presence of hypertension

Binary

heart_disease

Presence of heart disease

Binary

smoking_history

Smoking history category

Categorical

bmi

Body Mass Index

Numerical

HbA1c_level

HbA1c measurement

Numerical

blood_glucose_level

Blood glucose measurement

Numerical

diabetes

Prediction target

Binary

🎯 Target Variable

0 → No Diabetes
1 → Diabetes

🔄 Machine Learning Workflow

                    ┌─────────────────────┐
                    │      Dataset        │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ Data Inspection     │
                    │ & Cleaning          │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ Exploratory Data    │
                    │ Analysis            │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ One-Hot Encoding    │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ Stratified Train /  │
                    │ Test Split          │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ Decision Tree       │
                    │ Classifier          │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ Hyperparameter      │
                    │ Tuning              │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ Final Evaluation    │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ Feature Importance  │
                    └─────────────────────┘

🧹 Data Preprocessing

1. Duplicate Removal

The original dataset contained 100,000 records.

Duplicate records were checked and removed, leaving:

96,146 records

2. Missing Values

Missing values were checked using:

db.isna().sum()

No missing values were present in the modeling dataset.

3. Outlier Filtering

The notebook applies the IQR method to:

bmi

blood_glucose_level

After preprocessing and outlier filtering, the modeling dataset contained:

89,070 records

Important: Statistical outlier removal does not automatically mean an observation is medically invalid. Extreme clinical values can be genuine. This preprocessing choice should therefore be reconsidered before using the workflow for any real clinical application.

📊 Exploratory Data Analysis

The project investigates the dataset using multiple visualizations.

Visualizations included

Diabetes class distribution

Age distribution by diabetes status

BMI distribution by diabetes status

Blood glucose distribution by diabetes status

Feature correlation analysis

Feature importance from the trained Decision Tree

The class distribution showed substantial imbalance between the 0 and 1 classes, making metrics such as recall, precision, and F1-score important alongside accuracy.

🌳 Model Development

Initial Decision Tree

The initial model was configured as:

DecisionTreeClassifier(
    criterion='gini',
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features=None,
    class_weight='balanced',
    random_state=42
)

Initial Results

Class

Precision

Recall

F1-score

0

0.99

0.85

0.92

1

0.29

0.90

0.44

Accuracy





0.86

The class-weighted model achieved high recall for the diabetes class but produced many false-positive predictions.

🔧 Hyperparameter Optimization

Instead of relying on the initial model configuration, GridSearchCV was used to search for a better Decision Tree configuration.

Parameters searched

params = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 4, 5, 6, 7, 8, 10],
    'min_samples_split': [2, 5, 10, 20, 50],
    'min_samples_leaf': [1, 2, 5, 10, 20],
    'class_weight': [None, 'balanced']
}

Cross-validation

CV = 5 folds
Scoring = F1-score

Best Parameters

{
    'class_weight': None,
    'criterion': 'gini',
    'max_depth': 10,
    'min_samples_leaf': 20,
    'min_samples_split': 50
}

Best Cross-Validation F1

0.7258

🏆 Final Model Performance

The tuned model was evaluated on the held-out test set.

Classification Report

Class

Precision

Recall

F1-score

Support

0 — No Diabetes

0.97

1.00

0.99

16,715

1 — Diabetes

0.97

0.58

0.73

1,099

Overall Accuracy





0.97

17,814

Summary Metrics

Metric

Score

🎯 Accuracy

97%

📌 Macro Precision

0.97

📌 Macro Recall

0.79

📌 Macro F1

0.86

📌 Weighted F1

0.97

🩺 Diabetes Precision

0.97

🩺 Diabetes Recall

0.58

🩺 Diabetes F1

0.73

What these results actually mean

The 97% accuracy should not be interpreted as clinical accuracy.

The dataset is imbalanced, and the model's class-specific metrics tell a more useful story:

The model has very high precision (97%) for the diabetes class.

The model has 58% recall for the diabetes class.

Therefore, the model is relatively selective when predicting diabetes but still misses a meaningful number of positive cases.

The model should not be presented as a clinically validated diabetes screening or diagnostic system.

This distinction is important when interpreting an imbalanced classification model.

🔍 Feature Importance

The Decision Tree's feature_importances_ attribute is used to understand which input features contributed most to the tree's splitting decisions.

Example:

importance = pd.Series(
    model.feature_importances_,
    index=x_train.columns
).sort_values(ascending=False)

The resulting feature-importance visualization helps explain which features the trained Decision Tree relied on most.

Feature importance indicates model reliance, not causation. A high importance score does not mean that the feature independently causes diabetes.

💾 Saved Model

The final trained model is stored as:

Diabetes-Prediction-Project-Model.pkl

Load the model

import pickle

with open("Diabetes-Prediction-Project-Model.pkl", "rb") as file:
    model = pickle.load(file)

Generate a prediction

prediction = model.predict(input_data)

Generate prediction probabilities

probability = model.predict_proba(input_data)

The input must contain the same feature structure and preprocessing used during model training.

📁 Repository Structure

diabetes-prediction-decision-tree/
│
├── 📓 Diabetics_Prediction_DT.ipynb
├── 🤖 Diabetes-Prediction-Project-Model.pkl
├── 📄 README.md
└── 📦 requirements.txt

If you later add a web application:

diabetes-prediction-decision-tree/
│
├── 📓 Diabetics_Prediction_DT.ipynb
├── 🤖 Diabetes-Prediction-Project-Model.pkl
├── 🌐 app.py
├── 📦 requirements.txt
└── 📄 README.md

🛠️ Tech Stack

Programming

🐍 Python

Data Science

Pandas

NumPy

Visualization

Matplotlib

Seaborn

Machine Learning

Scikit-learn

Decision Tree Classifier

GridSearchCV

Model Persistence

Pickle

Development

Jupyter Notebook

KaggleHub

📚 Skills Demonstrated

This project demonstrates practical experience with:

Data Collection
      ↓
Data Inspection
      ↓
Data Cleaning
      ↓
Duplicate Detection
      ↓
Outlier Detection
      ↓
Exploratory Data Analysis
      ↓
Categorical Encoding
      ↓
Class Imbalance Analysis
      ↓
Train/Test Split
      ↓
Decision Tree Modeling
      ↓
Hyperparameter Tuning
      ↓
Cross-Validation
      ↓
Model Evaluation
      ↓
Feature Importance
      ↓
Model Serialization

⚠️ Limitations

This project has several limitations that should be understood before interpreting the results:

The dataset is not a clinically validated dataset.

The model has not been externally validated on an independent population.

The target is based on the dataset's existing labels.

The classes are imbalanced.

The final model has 58% recall for the diabetes class.

The preprocessing pipeline removes statistical outliers from BMI and blood glucose.

HbA1c_level and blood_glucose_level are highly informative clinical measurements.

The model should not be treated as a long-term diabetes risk calculator.

Performance on this dataset does not guarantee equivalent performance on real-world clinical data.

🔮 Future Improvements

Model Improvements

Compare with Logistic Regression

Compare with Random Forest

Compare with XGBoost

Perform threshold tuning

Evaluate ROC-AUC

Evaluate Precision-Recall AUC

Perform probability calibration

Test alternative imbalance-handling methods

Engineering Improvements

Create a reusable preprocessing pipeline

Add input validation

Build a Streamlit application

Add automated tests

Add model/version tracking

Create reproducible training scripts

Validation Improvements

Evaluate on an independent external dataset

Investigate population/generalization differences

Reassess the treatment of extreme clinical values

Evaluate clinically meaningful performance thresholds

🧪 Reproducibility

Recommended environment:

Python 3.x

Install dependencies:

pip install -r requirements.txt

Run the notebook:

jupyter notebook Diabetics_Prediction_DT.ipynb

🩺 Disclaimer

This project is strictly for educational and machine learning demonstration purposes.

It is not a medical diagnostic tool, has not been clinically validated, and should not be used to diagnose, treat, screen, or make medical decisions about diabetes.

Always consult a qualified healthcare professional for medical evaluation.

👤 Author

Diabetes Classification — Machine Learning Portfolio Project

Built to demonstrate an end-to-end classification workflow using Decision Trees, from data preprocessing and EDA through hyperparameter optimization, evaluation, interpretation, and model serialization.

<div align="center">

⭐ If you found this project useful, consider starring the repository.

Built with Python • Pandas • Scikit-learn • Matplotlib • Seaborn

</div>
