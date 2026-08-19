# Diabetes Risk Prediction
A machine learning system that predicts a patient's diabetes risk using demographic, behavioral, and clinical data.

## Business Impact
This project demonstrates how machine learning can support data-driven healthcare decisions by identifying high-risk individuals, improving screening efficiency, and enabling earlier intervention through predictive analytics.

## Table of Contents
- [Overview](#overview)
- [Business Problem](#business-problem)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Machine Learning Models](#machine-learning-models)
- [Model Evaluation](#model-evaluation)
- [Key Insights](#key-insights)
- [Limitations & Future Improvements](#limitations--future-improvements)
- [Technologies Used](#technologies-used)
- [How to Run](#how-to-run)
- [Project Files](#project-files)
- [Skills Demonstrated](#skills-demonstrated)

## Overview
This project builds and evaluates a classification model that predicts diabetes status from routinely collected patient data. It covers the full pipeline: exploratory data analysis, data cleaning, feature preparation, handling class imbalance, training multiple models, hyperparameter tuning, and evaluating results against business-relevant metrics.

## Business Problem
Early identification of at-risk patients allows healthcare providers and insurers to intervene before complications develop, but manual screening does not scale across large populations. A predictive model that flags high-risk individuals using standard clinical and demographic data (age, BMI, blood glucose, HbA1c, hypertension, heart disease, smoking history) supports:

- **Earlier intervention** — timely follow-up testing and treatment for likely diabetic patients
- **Resource prioritization** — directing clinical attention to the patients most at risk
- **Cost reduction** — fewer missed diagnoses (false negatives) and fewer unnecessary follow-ups (false positives)

Because a missed diagnosis can delay treatment, **recall** (catching true diabetic cases) is prioritized alongside **precision** (avoiding unnecessary false alarms) when selecting the final model.

## Dataset
The dataset contains records for **100,000 individuals**, combining demographic, behavioral, and clinical attributes relevant to diabetes risk:

| Category | Fields |
|---|---|
| Demographics | age, gender, race, location |
| Clinical measurements | BMI, HbA1c level, blood glucose level |
| Health history | hypertension, heart disease, smoking history |
| Target | diabetes status (0 = non-diabetic, 1 = diabetic) |

The dataset is **imbalanced**: approximately 91,500 non-diabetic cases vs. 8,500 diabetic cases (~8.5% positive class).

## Project Workflow

### 1. Data Exploration
- Inspected structure, data types, and dimensions
- Checked for missing values and duplicate records
- Reviewed summary statistics (mean, median, spread) for numerical features
- Analyzed class distributions across the target and key categorical features
- Built a correlation heatmap to identify the strongest predictors of diabetes

<p align="center">
  <img src="images/correlation_heatmap.png" alt="Correlation heatmap of numeric features" width="600">
</p>

Blood glucose level, HbA1c level, BMI, hypertension, and heart disease showed the strongest positive correlations with diabetes status.

### 2. Data Cleaning & Preprocessing
- Renamed columns for clarity
- Removed duplicate rows
- Replaced an ambiguous `"Other"` gender category with the most frequent value
- Dropped columns with limited predictive value or unusable content (`year`, `location`, unstructured clinical notes)
- Reviewed numerical features for outliers and retained clinically plausible extreme values
- One-hot encoded categorical variables (smoking history, gender)
- Standardized numerical features (age, BMI, HbA1c, blood glucose) using `StandardScaler`

<p align="center">
  <img src="images/outlier_boxplots.png" alt="Boxplots of numerical features for outlier detection" width="700">
</p>

<p align="center">
  <img src="images/age_by_diabetes_boxplot.png" alt="Age distribution by diabetes status" width="400">
  <img src="images/age_distribution.png" alt="Age distribution across patients" width="400">
</p>

### 3. Feature Engineering & Selection
- Tested an age × BMI interaction feature; excluded it after confirming it added negligible correlation with the target variable

### 4. Model Development
- Split data into training and test sets (80/20)
- Addressed class imbalance in the training set using **SMOTE** oversampling
- Trained three classification models: Decision Tree, Random Forest, and XGBoost
- Performed hyperparameter tuning on the Decision Tree using `RandomizedSearchCV` with class weighting, optimizing for F1-score

### 5. Model Evaluation
- Compared all models on accuracy, precision, recall, and F1-score for the diabetic class
- Selected the final model based on the best overall balance between catching true diabetic cases and minimizing false positives

## Machine Learning Models

| Model | Why It Was Used |
|---|---|
| **Decision Tree** | Simple, interpretable baseline for classification |
| **Random Forest** | Ensemble of decision trees to reduce overfitting and improve generalization |
| **XGBoost** | Gradient-boosted ensemble, well-suited to structured/tabular data and imbalanced classification tasks |

## Model Evaluation

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Decision Tree | 0.9461 | 0.6613 | 0.7608 | 0.7075 |
| Random Forest | 0.9578 | 0.7516 | 0.7590 | 0.7553 |
| **XGBoost** | **0.9712** | **0.9209** | 0.7264 | **0.8121** |
| Decision Tree (tuned) | 0.9542 | 0.7295 | 0.7410 | 0.7352 |

*Metrics reported for the diabetic class on the held-out test set (1,714 diabetic patients out of 19,998 total test records).*

**Selected model: XGBoost.** Decision Tree and Random Forest achieved marginally higher recall, but XGBoost delivered substantially higher precision, the highest F1-score, and the highest overall accuracy — making it the most reliable model for real-world screening use.

**In practical terms:** on the test set, the XGBoost model correctly flags approximately **1,245 of the 1,714 diabetic patients (~73%)**, while roughly **92% of all patients it flags as diabetic are true positives** (around 107 false alarms across ~1,352 flagged cases). This balance minimizes missed diagnoses without overwhelming clinical staff with false positives.

Hyperparameter tuning improved the Decision Tree's accuracy, precision, and F1-score over its baseline version, but it still did not surpass XGBoost's performance.

## Key Insights
- Blood glucose level, HbA1c level, BMI, hypertension, and heart disease are the strongest predictors of diabetes in this dataset — consistent with established clinical risk factors.
- Diabetic patients are, on average, older and have higher BMI, HbA1c, and blood glucose levels than non-diabetic patients.
- Race showed minimal correlation with diabetes status, indicating it is not a meaningful predictor here.
- Class imbalance (~8.5% positive cases) required SMOTE resampling during training so the models could learn meaningful patterns for the minority class.
- XGBoost's high precision (0.92) combined with solid recall (0.73) makes it the most operationally practical model — it limits false negatives without generating excessive false positives.

## Limitations & Future Improvements
- The dataset's exact source/provenance is not documented within the notebook; results should be validated against real-world clinical data before any operational use.
- Evaluation relies on a single train/test split rather than cross-validated performance across multiple splits.
- The default 0.5 classification threshold was used; adjusting the decision threshold could further improve recall for the diabetic class if minimizing missed diagnoses is the priority.
- A feature importance analysis (e.g., from the trained XGBoost model) would add further interpretability and is a natural next step.
- A complementary BI dashboard (e.g., Power BI/Tableau) summarizing risk segments and model outputs would make these findings more accessible to non-technical stakeholders.

## Technologies Used
- **Language:** Python
- **Data manipulation:** pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine learning:** scikit-learn (Decision Tree, Random Forest, preprocessing, `RandomizedSearchCV`)
- **Gradient boosting:** XGBoost
- **Imbalanced data handling:** imbalanced-learn (SMOTE)

## How to Run
1. Clone this repository and open `Final_ML.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab.
2. Install the required libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn
   ```
3. Ensure the dataset file (`diabetes_dataset.csv`) is in the same directory as the notebook.
4. Run all cells sequentially to reproduce the data preparation, model training, and evaluation steps.

## Project Files
```
├── Final_ML.ipynb     # Full notebook: EDA, preprocessing, feature engineering,
│                       # model training, tuning, and evaluation
└── images/             # Exported charts referenced in this README
```

## Skills Demonstrated
- End-to-end machine learning workflow design and execution
- Exploratory data analysis and statistical interpretation
- Data cleaning, encoding, and feature scaling
- Handling imbalanced classification problems with SMOTE
- Training and comparing multiple classification algorithms (Decision Tree, Random Forest, XGBoost)
- Hyperparameter tuning with `RandomizedSearchCV`
- Model evaluation using accuracy, precision, recall, and F1-score, with business-oriented interpretation of trade-offs
- Data visualization for communicating patterns and results
- Clear technical documentation for non-technical stakeholders
