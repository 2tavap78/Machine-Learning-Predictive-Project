# Machine-Learning-Predictive-Project

# Reach for Change — Predicting Donor Response to Optimize Outreach

A binary classification project to predict whether a potential donor will respond positively to a fundraising campaign, enabling a nonprofit federation to target outreach more precisely and reduce unnecessary contact.

---

## Context

The **Civic Support Alliance (CSA)** is a federation representing multiple humanitarian and social aid programs. As donor fatigue has become an increasing challenge for nonprofits, the organisation sought to move away from blanket outreach and instead identify — in advance — which individuals are likely to donate if contacted.

This project builds a predictive system to answer a single core question: **will this person donate if contacted?**

---

## Dataset

The dataset contains historical records from 13,561 individuals contacted in a previous fundraising campaign, including:

- **Demographic features** — age, neighbourhood-level socioeconomic indicators (median home value, household income), military affiliation
- **Donation history** — lifetime gift amounts, recency, frequency, average gift size
- **Campaign interaction** — number of past promotions, response rates, card response counts
- **Target variable** — `TARGET_B`: 1 if the person donated in the last campaign, 0 otherwise

The dataset exhibits significant **class imbalance** (~25% donors, ~75% non-donors), which drove several modelling decisions.

---

## Project Structure

```
.
├── Donors_Predictive_Final.ipynb   # Main notebook
├── donors_train_target.csv         # Training labels
├── test.csv                        # Test set (no labels)
├── sample_submission.csv           # Submission format reference
└── submission_lr.csv               # Final Kaggle submission
```

---

## Methodology

### 1. Exploratory Data Analysis & Preprocessing
- Identified and handled missing values through imputation
- Detected and corrected outliers, misclassifications, and data incoherencies
- Analysed class imbalance and its implications for model selection

### 2. Feature Selection
A multi-method approach was applied to identify the most informative features:

| Method | Variable Type |
|---|---|
| Chi-Square test | Categorical |
| Variance Threshold | Numerical |
| Spearman Correlation | Numerical |
| Decision Tree Importance | Numerical |
| Recursive Feature Elimination (RFE) | Numerical |
| Lasso Regularisation | Numerical |

### 3. Model Assessment Strategy
A **holdout split** was used for model evaluation. Five classifiers were trained and compared using **F1 Score** (class 1) as the primary metric, given the class imbalance:

| Model | Notes |
|---|---|
| K-Nearest Neighbours | Susceptible to class imbalance; distance metric less optimal for this feature space |
| Decision Tree | Prone to overfitting; lower stability |
| Random Forest | Strong in theory but unable to surpass Logistic Regression after tuning |
| Neural Network (MLP) | Better suited for larger datasets; struggled with imbalance |
| **Logistic Regression** ✓ | **Best performer — selected as final model** |

### 4. Optimisation
Hyperparameter tuning was performed via `GridSearchCV`. The final Logistic Regression configuration achieved:

- **Class 1 F1**: 0.40
- **Class 1 Recall**: 0.55
- **Kaggle F1 Score**: 0.4279

The high recall for class 1 was a deliberate trade-off — correctly identifying real donors matters more than avoiding false positives in this context.

---

## Key Results

Logistic Regression outperformed all other candidate models on every relevant metric. The final model correctly identified **373 true donors** on the test holdout, compared to 68 for KNN and 240 for Random Forest.

The model's tendency to produce more false positives (non-donors flagged as donors) is an acceptable cost given that the goal is to *not miss* likely donors rather than to achieve perfect precision.

---

## Tech Stack

- **Python 3.10**
- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `scikit-learn` — preprocessing, feature selection, modelling, evaluation
- `scipy` — statistical tests (Chi-Square, Spearman)
- Google Colab

---

## Running the Notebook

1. Clone the repository
2. Place `donors_train_target.csv` and `test.csv` in the same directory as the notebook (or adjust `lab_root` if running in Colab)
3. Run all cells top to bottom — the final cell exports `submission_lr.csv`

> **Note:** If using Google Colab, mount your Drive and set `lab_root` to point to your data folder before running.
